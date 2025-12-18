"""
Helpers for hard-sectored flux captures (e.g., 32 holes + index).

Greaseweazle's `--hardsector --raw` mode records one revolution per hole
tick, so an 8" disk with 32 sector holes plus one index hole yields
33 entries per full rotation. We group those, detect the index split,
and merge the two shortest adjacent entries to recover 32 uniform
intervals per rotation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

from .fm import (
    DEFAULT_CLOCK_ADJ,
    FMPhaseCandidate,
    SectorGuess,
    _select_fm_phase_candidates,
    bits_to_bytes,
    decode_fm_bytes,
    estimate_cell_ticks,
    pll_decode_bits,
    scan_fm_sectors,
)
from .scp import SCPImage, TrackData


@dataclass(frozen=True)
class HoleCapture:
    hole_index: int
    revolution_indices: tuple[int, ...]
    index_ticks: int
    flux_count: int
    logical_sector_index: Optional[int] = None

    @property
    def revolution_index(self) -> int:
        """
        Primary revolution index for compatibility with single-entry callers.
        """

        return self.revolution_indices[0]


@dataclass(frozen=True)
class HardSectorGrouping:
    groups: List[List[HoleCapture]]
    sectors_per_rotation: int
    index_holes_per_rotation: int = 1
    rotated_by: int = 0
    index_confidence: float = 0.0
    index_aligned_flag: Optional[bool] = None
    short_pair_positions: List[Optional[int]] | None = None
    pulses_per_rotation: int = 33
    index_pair_position: Optional[int] = None
    short_intervals: tuple[int, ...] | None = None
    detection_notes: str | None = None

    @property
    def rotations(self) -> int:
        return len(self.groups)

    @property
    def physical_revolutions(self) -> int:
        return len(self.groups)


@dataclass(frozen=True)
class SectorWindowDecode:
    bytes_out: bytes
    phase: int
    bit_shift: int
    score: float
    bitcells: tuple[int, ...]
    candidates: tuple[FMPhaseCandidate, ...] | None = None
    encoding: str = "fm"


FORMAT_PRESETS = {
    "cpm-16x256": {"expected_sectors": 16, "sector_size": 256, "encoding": "fm"},
    "cpm-26x128": {"expected_sectors": 26, "sector_size": 128, "encoding": "fm"},
    "ibm-9x512-mfm": {"expected_sectors": 9, "sector_size": 512, "encoding": "mfm"},
    "wang-ois-hs32-fm-16x256": {
        "expected_sectors": 16,
        "sector_size": 256,
        "encoding": "fm",
        "logical_sectors": 16,
        "physical_sectors": 32,
    },
}

FILL_BYTES = {0xFF, 0xFE, 0xFB, 0xF7, 0xEF, 0xDF, 0xFD, 0xBF, 0x7F}


def payload_metrics(data: bytes) -> tuple[float, float]:
    if not data:
        return 1.0, 0.0
    fill_ratio = sum(1 for b in data if b in FILL_BYTES) / len(data)
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    entropy = 0.0
    for count in freq:
        if count:
            p = count / len(data)
            entropy -= p * math.log2(p)
    return fill_ratio, entropy


def best_payload_windows(
    data: bytes, window_size: int, top_n: int = 1
) -> list[tuple[int, bytes, float, float]]:
    """
    Return the top-N windows ranked by lowest fill ratio then highest entropy.
    """

    if window_size <= 0 or not data:
        return []

    results: list[tuple[float, float, int, bytes]] = []
    for start in range(0, max(1, len(data) - window_size + 1)):
        window = data[start : start + window_size]
        fill, entropy = payload_metrics(window)
        results.append((fill, -entropy, start, window))

    results.sort()
    trimmed = results[:top_n]
    return [
        (start, window, fill, -entropy_neg)
        for fill, entropy_neg, start, window in trimmed
    ]


def _detect_short_pair_position(
    intervals: Sequence[int], threshold_scale: float = 0.75
) -> tuple[Optional[int], float, tuple[int, ...]]:
    """
    Locate two consecutive short intervals that represent the split index hole.

    Returns (start_index, median_interval, short_indices). The start_index is
    chosen from adjacent short intervals whose combined duration is closest to
    the median interval length.
    """
    if not intervals:
        return None, 0.0, ()

    med = median(intervals)
    cutoff = med * threshold_scale
    short_indices = tuple(idx for idx, val in enumerate(intervals) if val < cutoff)
    if len(short_indices) < 2:
        return None, med, short_indices

    best: tuple[float, int] | None = None
    for idx in short_indices:
        nxt = (idx + 1) % len(intervals)
        if nxt not in short_indices:
            continue
        combined = intervals[idx] + intervals[nxt]
        closeness = abs(combined - med)
        if best is None or closeness < best[0]:
            best = (closeness, idx)
    if best is None:
        return None, med, short_indices
    return best[1], med, short_indices


def _rotate_list(values: Sequence, offset: int) -> list:
    if not values:
        return []
    adj = offset % len(values)
    return list(values[adj:] + values[:adj])


def _score_candidate_bytes(candidate: bytes) -> float:
    if not candidate:
        return -math.inf
    fill, entropy = payload_metrics(candidate[:1024])
    mark_score = sum(
        candidate.count(mark) for mark in (0xFE, 0xFB, 0xFA, 0xA1, 0x4E, 0xF6)
    )
    return mark_score * 4 + entropy - fill * 2.0


def normalize_rotation(
    hole_captures: Sequence[HoleCapture],
    expected_pair_start: int | None = None,
    threshold_scale: float = 0.75,
) -> Tuple[List[HoleCapture], Optional[int]]:
    """
    Merge the index split in a 33-hole capture into 32 uniform intervals.

    The Wang HS32 captures show one short+short adjacent pair caused by the
    index pulse splitting a hard-sector window. We detect candidates shorter
    than 75% of the median interval, locate the adjacent pair among them, and
    return a new list with the pair merged (flux concatenated, ticks summed).
    """

    if len(hole_captures) < 3:
        return list(hole_captures), None

    short_pair_start = expected_pair_start
    if short_pair_start is None:
        short_pair_start, _, _ = _detect_short_pair_position(
            [h.index_ticks for h in hole_captures], threshold_scale=threshold_scale
        )
    merged: List[HoleCapture] = []
    skip: set[int] = set()
    for idx, hole in enumerate(hole_captures):
        if idx in skip:
            continue
        if short_pair_start is not None and idx == short_pair_start:
            partner = (idx + 1) % len(hole_captures)
            partner_hole = hole_captures[partner]
            combined_flux_count = hole.flux_count + partner_hole.flux_count
            combined_ticks = hole.index_ticks + partner_hole.index_ticks
            combined_indices = hole.revolution_indices + partner_hole.revolution_indices
            merged.append(
                HoleCapture(
                    hole_index=len(merged),
                    revolution_indices=combined_indices,
                    index_ticks=combined_ticks,
                    flux_count=combined_flux_count,
                )
            )
            skip.add(partner)
            continue
        merged.append(
            HoleCapture(
                hole_index=len(merged),
                revolution_indices=hole.revolution_indices,
                index_ticks=hole.index_ticks,
                flux_count=hole.flux_count,
            )
        )

    return merged, short_pair_start


def _flatten_flux(
    track: TrackData,
    hole_capture: HoleCapture,
    invert_flux: bool = False,
    clock_scale: float = 1.0,
) -> list[int]:
    flux: list[int] = []
    for idx in hole_capture.revolution_indices:
        part = track.decode_flux(idx)
        if invert_flux:
            part = list(reversed(part))
        if clock_scale != 1.0:
            part = [max(1, int(x * clock_scale)) for x in part]
        flux.extend(part)
    return flux


def decode_sector_window_bytes(
    image: SCPImage,
    track: TrackData,
    hole_capture: HoleCapture,
    encoding: str = "fm",
    invert_bits: bool = False,
    invert_flux: bool = False,
    initial_clock_ticks: float | None = None,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
    clock_scale: float = 1.0,
) -> SectorWindowDecode:
    """
    Decode one normalized sector window and score both FM phases.

    The decode score blends FM phase heuristics with byte-level entropy/mark
    counts so each sector window can be ranked independently.
    """

    flux = _flatten_flux(
        track, hole_capture, invert_flux=invert_flux, clock_scale=clock_scale
    )
    bitcells = pll_decode_bits(
        flux,
        sample_freq_hz=image.sample_freq_hz,
        index_ticks=hole_capture.index_ticks,
        clock_adjust=clock_adjust,
        initial_clock_ticks=initial_clock_ticks,
        invert=invert_bits,
    )

    if encoding.lower() == "mfm":
        best: SectorWindowDecode | None = None
        for phase in (0, 1):
            data_bits = bitcells[phase::2]
            decoded = bits_to_bytes(data_bits)
            score = _score_candidate_bytes(decoded)
            candidate = SectorWindowDecode(
                bytes_out=decoded,
                phase=phase,
                bit_shift=phase,
                score=score,
                bitcells=tuple(bitcells),
                candidates=None,
                encoding="mfm",
            )
            if best is None or candidate.score > best.score:
                best = candidate
        assert best is not None
        return best

    best_phase, candidates = _select_fm_phase_candidates(bitcells)
    score = _score_candidate_bytes(best_phase.bytes_out) + best_phase.score
    return SectorWindowDecode(
        bytes_out=best_phase.bytes_out,
        phase=best_phase.phase,
        bit_shift=best_phase.bit_shift,
        score=score,
        bitcells=tuple(bitcells),
        candidates=candidates,
        encoding="fm",
    )


def pair_holes(holes: Sequence[HoleCapture], phase: int = 0) -> List[HoleCapture]:
    """
    Combine adjacent holes into logical sectors (two holes per logical sector).

    The `phase` argument controls pairing start: phase=0 pairs (0,1),(2,3)...;
    phase=1 pairs (1,2),(3,4)... with wraparound.
    """

    if not holes:
        return []

    paired: List[HoleCapture] = []
    count = len(holes)
    for pair_idx in range(count // 2):
        first = (pair_idx * 2 + phase) % count
        second = (first + 1) % count
        h0 = holes[first]
        h1 = holes[second]
        paired.append(
            HoleCapture(
                hole_index=pair_idx,
                revolution_indices=h0.revolution_indices + h1.revolution_indices,
                index_ticks=h0.index_ticks + h1.index_ticks,
                flux_count=h0.flux_count + h1.flux_count,
                logical_sector_index=pair_idx,
            )
        )
    return paired


def group_hard_sectors(
    track: TrackData,
    sectors_per_rotation: int = 32,
    index_aligned: bool = True,
    require_strong_index: bool = False,
    hs_normalize: bool = True,
    threshold_scale: float = 0.75,
) -> HardSectorGrouping:
    """
    Chunk hard-sector pulses into physical rotations.

    Each SCP "revolution" is a pulse interval between hard-sector holes. For
    HS32 captures, there are 33 intervals per physical rotation (32 sector
    holes plus an index hole that splits one interval into two short spans).

    This function identifies the split index interval (two consecutive short
    entries), rotates the capture so each rotation begins at that index, and
    optionally merges the index-split pair to yield 32 sector windows.
    """
    pulses_per_rotation = sectors_per_rotation + 1
    revs = list(track.revolutions)
    ordered_indices = list(range(len(revs)))

    base_intervals = [rev.index_ticks for rev in revs]
    detection_window = (
        base_intervals[:pulses_per_rotation]
        if len(base_intervals) >= pulses_per_rotation
        else base_intervals
    )
    short_pair, med_interval, short_indices = _detect_short_pair_position(
        detection_window, threshold_scale=threshold_scale
    )
    rotated_by = short_pair % pulses_per_rotation if short_pair is not None else 0
    detection_notes = None
    if short_pair is None:
        detection_notes = "No adjacent short intervals found; using raw capture order."
        rotated_by = 0

    confidence = 0.0
    if med_interval and short_pair is not None and base_intervals:
        confidence = 1.0 - (
            abs(base_intervals[rotated_by] - (med_interval / 2)) / max(med_interval, 1)
        )

    if rotated_by and (not index_aligned or not require_strong_index):
        revs = _rotate_list(revs, rotated_by)
        ordered_indices = _rotate_list(ordered_indices, rotated_by)

    groups: List[List[HoleCapture]] = []
    short_pairs: List[Optional[int]] = []
    for _rot_idx, base in enumerate(range(0, len(revs), pulses_per_rotation)):
        chunk = revs[base : base + pulses_per_rotation]
        if len(chunk) < pulses_per_rotation:
            break
        chunk_indices = ordered_indices[base : base + pulses_per_rotation]

        local_intervals = [rev.index_ticks for rev in chunk]
        local_pair, _med, _shorts = _detect_short_pair_position(
            local_intervals, threshold_scale=threshold_scale
        )
        pair_start = local_pair if local_pair is not None else short_pair

        captures: List[HoleCapture] = []
        for idx, rev in enumerate(chunk):
            captures.append(
                HoleCapture(
                    hole_index=idx,
                    revolution_indices=(chunk_indices[idx],),
                    index_ticks=rev.index_ticks,
                    flux_count=rev.flux_count,
                )
            )
        normalized = captures
        if hs_normalize:
            normalized, pair_start = normalize_rotation(
                captures,
                expected_pair_start=pair_start,
                threshold_scale=threshold_scale,
            )
        short_pairs.append(pair_start)
        groups.append(normalized)
    return HardSectorGrouping(
        groups=groups,
        sectors_per_rotation=sectors_per_rotation,
        rotated_by=rotated_by,
        index_confidence=confidence,
        index_aligned_flag=index_aligned,
        short_pair_positions=short_pairs,
        pulses_per_rotation=pulses_per_rotation,
        index_pair_position=short_pair,
        short_intervals=short_indices,
        detection_notes=detection_notes,
    )


def decode_hole(
    image: SCPImage,
    track: TrackData,
    hole_capture: HoleCapture,
    use_pll: bool = False,
    require_sync: bool = False,
    encoding: str = "fm",
    initial_clock_ticks: float | None = None,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
    invert_bits: bool = False,
    invert_flux: bool = False,
    clock_scale: float = 1.0,
) -> Optional[SectorGuess]:
    """
    Decode one hole's worth of flux into bytes using FM heuristics or PLL.
    """
    if not use_pll and encoding.lower() == "fm":
        decoded_bytes = decode_fm_bytes(
            _flatten_flux(
                track,
                hole_capture,
                invert_flux=invert_flux,
                clock_scale=clock_scale,
            )
        ).bytes_out
        decoded_score = _score_candidate_bytes(decoded_bytes)
    else:
        decoded = decode_sector_window_bytes(
            image,
            track,
            hole_capture,
            encoding=encoding,
            invert_bits=invert_bits,
            invert_flux=invert_flux,
            initial_clock_ticks=initial_clock_ticks,
            clock_adjust=clock_adjust,
            clock_scale=clock_scale,
        )
        decoded_bytes = decoded.bytes_out
        decoded_score = decoded.score

    sync_bytes = (0xA1, 0x4E) if encoding.lower() == "mfm" else (0xA1,)
    sectors = scan_fm_sectors(
        decoded_bytes, require_sync=require_sync, sync_bytes=sync_bytes
    )
    if not sectors:
        return None
    sectors[0].decode_score = decoded_score
    return sectors[0]


def compute_flux_index_deltas(
    track: TrackData, max_entries: Optional[int] = None
) -> List[int]:
    """
    Compare recorded index_ticks to the sum of flux intervals for each entry.

    Large deltas can indicate chopped intervals around hard-sector holes or
    mis-parsed offsets.
    """
    deltas: List[int] = []
    limit = (
        track.revolution_count
        if max_entries is None
        else min(max_entries, track.revolution_count)
    )
    for idx in range(limit):
        flux = track.decode_flux(idx)
        delta = track.revolutions[idx].index_ticks - sum(flux)
        deltas.append(delta)
    return deltas


def compute_flux_index_diagnostics(
    track: TrackData, grouping: HardSectorGrouping, rotation_index: int = 0
) -> List[dict]:
    """
    Return per-hole diagnostics comparing index_ticks to summed flux intervals.

    Each entry includes the revolution indices used, summed flux, raw delta,
    and ratio (flux/index).
    """

    if rotation_index >= grouping.rotations:
        return []

    diagnostics: List[dict] = []
    for hole in grouping.groups[rotation_index]:
        flux_total = 0
        for idx in hole.revolution_indices:
            flux_total += sum(track.decode_flux(idx))
        delta = hole.index_ticks - flux_total
        ratio = (flux_total / hole.index_ticks) if hole.index_ticks else 0.0
        diagnostics.append(
            {
                "hole_index": hole.hole_index,
                "revolution_indices": hole.revolution_indices,
                "index_ticks": hole.index_ticks,
                "flux_total": flux_total,
                "delta": delta,
                "ratio": ratio,
            }
        )
    return diagnostics


def stitch_rotation_flux(
    track: TrackData,
    grouping: HardSectorGrouping,
    rotation_index: int,
    compensate_gaps: bool = False,
) -> Tuple[List[int], int]:
    """
    Concatenate all hole flux intervals for a given rotation into a single list.

    If compensate_gaps is True, insert a no-transition interval equal to the
    difference between index_ticks and summed flux for each hole.
    Returns (flux_intervals, approx_index_ticks_for_rotation).
    """
    if rotation_index >= grouping.rotations:
        return [], 0
    stitched: List[int] = []
    holes = grouping.groups[rotation_index]
    total_index_ticks = sum(h.index_ticks for h in holes)
    for hole in holes:
        hole_flux_total = 0
        for idx in hole.revolution_indices:
            flux = track.decode_flux(idx)
            hole_flux_total += sum(flux)
            stitched.extend(flux)
        if compensate_gaps:
            delta = hole.index_ticks - hole_flux_total
            if delta > 0:
                stitched.append(delta)
    return stitched, total_index_ticks


def decode_hole_bytes(
    image: SCPImage,
    track: TrackData,
    hole_capture: HoleCapture,
    use_pll: bool = False,
    encoding: str = "fm",
    initial_clock_ticks: float | None = None,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
    invert_bits: bool = False,
    invert_flux: bool = False,
    clock_scale: float = 1.0,
) -> bytes:
    """
    Decode one hole's flux and return raw decoded bytes without sector framing.
    """
    if not use_pll and encoding.lower() == "fm":
        flux = _flatten_flux(
            track,
            hole_capture,
            invert_flux=invert_flux,
            clock_scale=clock_scale,
        )
        return decode_fm_bytes(flux).bytes_out

    decoded = decode_sector_window_bytes(
        image,
        track,
        hole_capture,
        encoding=encoding,
        invert_bits=invert_bits,
        invert_flux=invert_flux,
        initial_clock_ticks=initial_clock_ticks,
        clock_adjust=clock_adjust,
        clock_scale=clock_scale,
    )
    return decoded.bytes_out


def assemble_rotation(
    image: SCPImage,
    track: TrackData,
    grouping: HardSectorGrouping,
    rotation_index: int,
    use_pll: bool = False,
    require_sync: bool = False,
    encoding: str = "fm",
    calibrate_rotation: bool = False,
    synthetic_from_hole: bool = False,
    expected_sectors: int = 16,
    expected_size: int = 256,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
) -> List[SectorGuess]:
    """
    Decode all holes in a given rotation and return any sector guesses found.
    """
    if rotation_index >= grouping.rotations:
        return []
    guesses: List[SectorGuess] = []
    initial_clock = None
    if calibrate_rotation and grouping.groups[rotation_index]:
        first_flux = track.decode_flux(
            grouping.groups[rotation_index][0].revolution_index
        )
        half, _, _ = estimate_cell_ticks(first_flux)
        initial_clock = half * 2

    for hole in grouping.groups[rotation_index]:
        guess = decode_hole(
            image,
            track,
            hole,
            use_pll=use_pll,
            require_sync=require_sync,
            encoding=encoding,
            initial_clock_ticks=initial_clock,
            clock_adjust=clock_adjust,
        )
        if guess and guess.length:
            guesses.append(guess)
            continue
        if synthetic_from_hole:
            raw = decode_hole_bytes(
                image,
                track,
                hole,
                use_pll=use_pll,
                encoding=encoding,
                initial_clock_ticks=initial_clock,
                clock_adjust=clock_adjust,
            )
            windows = best_payload_windows(raw, expected_size, top_n=1)
            if windows:
                offset, payload, _, _ = windows[0]
            else:
                offset, payload = 0, raw[:expected_size]
            decode_score = _score_candidate_bytes(payload)
            guesses.append(
                SectorGuess(
                    offset=offset,
                    track=track.track_number,
                    head=0,
                    sector_id=hole.hole_index % expected_sectors,
                    size_code=0,
                    length=len(payload),
                    crc_ok=False,
                    id_crc_ok=False,
                    data_crc_ok=False,
                    data=payload,
                    decode_score=decode_score,
                )
            )
    return guesses


def decode_track_best_map(
    image: SCPImage,
    track_number: int,
    sectors_per_rotation: int = 32,
    expected_sectors: int = 16,
    expected_size: int = 256,
    encoding: str = "fm",
    use_pll: bool = False,
    require_sync: bool = False,
    calibrate_rotation: bool = False,
    synthetic_from_holes: bool = False,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
    hs_normalize: bool = True,
) -> Dict[int, SectorGuess]:
    track = image.read_track(track_number)
    if not track:
        return {}
    grouping = group_hard_sectors(
        track,
        sectors_per_rotation=sectors_per_rotation,
        index_aligned=bool(image.header.flags & 0x01),
        hs_normalize=hs_normalize,
    )
    all_rotations = [
        assemble_rotation(
            image,
            track,
            grouping,
            r,
            use_pll=use_pll,
            require_sync=require_sync,
            encoding=encoding,
            calibrate_rotation=calibrate_rotation,
            synthetic_from_hole=synthetic_from_holes,
            expected_sectors=expected_sectors,
            expected_size=expected_size,
            clock_adjust=clock_adjust,
        )
        for r in range(grouping.rotations)
    ]
    return best_sector_map(
        all_rotations,
        expected_track=track_number,
        expected_head=0,
        expected_sector_count=expected_sectors,
        expected_size=expected_size,
    )


def build_raw_image(
    track_maps: Dict[int, Dict[int, SectorGuess]],
    track_order: List[int],
    expected_sectors: int,
    expected_size: int,
    fill_byte: int = 0x00,
) -> bytes:
    """
    Assemble a flat image from per-track sector maps.

    Missing sectors are filled with the given fill_byte.
    """
    fill_block = bytes([fill_byte]) * expected_size
    parts: List[bytes] = []
    for track_num in track_order:
        sector_map = track_maps.get(track_num, {})
        for sid in range(expected_sectors):
            guess = sector_map.get(sid)
            if guess and guess.data:
                parts.append(guess.data)
            else:
                parts.append(fill_block)
    return b"".join(parts)


def best_sector_map(
    rotations: List[List[SectorGuess]],
    expected_track: int,
    expected_head: int = 0,
    expected_sector_count: int = 16,
    expected_size: int = 256,
) -> Dict[int, SectorGuess]:
    """
    Pick the best copy for each sector id across rotations.
    Preference order: valid CRCs, then first occurrence.
    """
    best: Dict[int, SectorGuess] = {}
    for rotation in rotations:
        for guess in rotation:
            if guess.track != expected_track or guess.head != expected_head:
                continue
            if guess.length != expected_size:
                continue
            existing = best.get(guess.sector_id)
            existing_score = (
                getattr(existing, "decode_score", None) if existing else None
            )
            raw_new_score = getattr(guess, "decode_score", None)
            new_score = raw_new_score if raw_new_score is not None else -math.inf
            current_score = existing_score if existing_score is not None else -math.inf
            choose = (
                existing is None
                or (existing is not None and not existing.crc_ok and guess.crc_ok)
                or (
                    existing is not None
                    and existing.crc_ok == guess.crc_ok
                    and new_score > current_score
                )
            )
            if choose:
                best[guess.sector_id] = guess
    return {sid: g for sid, g in best.items() if sid < expected_sector_count}
