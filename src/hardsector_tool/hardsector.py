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
    SectorGuess,
    decode_fm_bytes,
    decode_mfm_bytes,
    estimate_cell_ticks,
    pll_decode_fm_bytes,
    scan_fm_sectors,
)
from .scp import RevolutionEntry, SCPImage, TrackData


@dataclass(frozen=True)
class HoleCapture:
    hole_index: int
    revolution_indices: tuple[int, ...]
    index_ticks: int
    flux_count: int

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

    @property
    def rotations(self) -> int:
        return len(self.groups)


FORMAT_PRESETS = {
    "cpm-16x256": {"expected_sectors": 16, "sector_size": 256, "encoding": "fm"},
    "cpm-26x128": {"expected_sectors": 26, "sector_size": 128, "encoding": "fm"},
    "ibm-9x512-mfm": {"expected_sectors": 9, "sector_size": 512, "encoding": "mfm"},
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


def _detect_index_hole_offset(
    revs: Sequence[RevolutionEntry], holes_per_rotation: int
) -> Tuple[int, float]:
    """
    Heuristically locate the index-hole entry within the first rotation.
    Returns (offset, confidence_ratio).
    """
    if not revs:
        return 0, 0.0
    window = revs[:holes_per_rotation]
    ticks = [rev.index_ticks for rev in window]
    max_idx = max(range(len(ticks)), key=ticks.__getitem__)
    med = median(ticks) if len(ticks) > 1 else ticks[0] if ticks else 0.0
    confidence = (ticks[max_idx] / med) if med else 0.0
    return max_idx, confidence


def _rotate_list(values: Sequence, offset: int) -> list:
    if not values:
        return []
    adj = offset % len(values)
    return list(values[adj:] + values[:adj])


def _merge_index_splits(
    revs: Sequence[RevolutionEntry], ordered_indices: Sequence[int]
) -> tuple[list[tuple[tuple[int, ...], int, int]], Optional[int]]:
    """
    Merge the two shortest, adjacent holes (index split) into one interval.

    Returns (merged_entries, short_pair_start) where merged_entries is a list of
    tuples: (revolution_indices, combined_index_ticks, combined_flux_count).
    If no adjacent minimum pair is found, the original ordering is preserved and
    short_pair_start is None.
    """

    if len(revs) < 3:
        merged = [
            ((ordered_indices[i],), revs[i].index_ticks, revs[i].flux_count)
            for i in range(len(revs))
        ]
        return merged, None

    med = median(r.index_ticks for r in revs)
    cutoff = med * 0.7 if med else 0
    candidates: list[tuple[int, int]] = []
    for idx, rev in enumerate(revs):
        nxt = (idx + 1) % len(revs)
        if rev.index_ticks < cutoff and revs[nxt].index_ticks < cutoff:
            candidates.append((rev.index_ticks + revs[nxt].index_ticks, idx))

    short_pair_start = min(candidates)[1] if candidates else None
    merged: list[tuple[tuple[int, ...], int, int]] = []
    visited: set[int] = set()

    for idx in range(len(revs)):
        if idx in visited:
            continue
        if short_pair_start is not None and idx == short_pair_start:
            partner = (idx + 1) % len(revs)
            indices = (ordered_indices[idx], ordered_indices[partner])
            merged_ticks = revs[idx].index_ticks + revs[partner].index_ticks
            merged_flux = revs[idx].flux_count + revs[partner].flux_count
            merged.append((indices, merged_ticks, merged_flux))
            visited.update({idx, partner})
            continue
        merged.append(
            ((ordered_indices[idx],), revs[idx].index_ticks, revs[idx].flux_count)
        )
        visited.add(idx)

    return merged, short_pair_start


def group_hard_sectors(
    track: TrackData,
    sectors_per_rotation: int = 32,
    index_aligned: bool = True,
    require_strong_index: bool = False,
) -> HardSectorGrouping:
    """
    Chunk revolutions into rotations based on the expected hole count.

    If the SCP FLAGS indicate capture did not begin immediately after index
    (index_aligned=False), we rotate the revolution list so each grouping
    starts with the entry whose index_ticks stands out as the index hole.
    """
    holes_per_rotation = sectors_per_rotation + 1
    revs = list(track.revolutions)
    ordered_indices = list(range(len(track.revolutions)))
    rotated_by = 0
    confidence = 0.0

    groups: List[List[HoleCapture]] = []
    short_pairs: List[Optional[int]] = []
    for rot_idx, base in enumerate(range(0, len(revs), holes_per_rotation)):
        chunk = revs[base : base + holes_per_rotation]
        if len(chunk) < holes_per_rotation:
            break
        chunk_indices = ordered_indices[base : base + holes_per_rotation]

        offset, chunk_conf = _detect_index_hole_offset(chunk, holes_per_rotation)
        if rot_idx == 0:
            confidence = chunk_conf
            rotated_by = offset if offset else 0
        should_rotate = (not index_aligned and offset != 0) or (
            offset != 0 and chunk_conf > 1.1 and not require_strong_index
        )
        if should_rotate:
            chunk = _rotate_list(chunk, offset)
            chunk_indices = _rotate_list(chunk_indices, offset)

        merged, short_pair = _merge_index_splits(chunk, chunk_indices)
        short_pairs.append(short_pair)
        captures: List[HoleCapture] = []
        for idx, (rev_indices, ticks, flux_count) in enumerate(merged):
            captures.append(
                HoleCapture(
                    hole_index=idx,
                    revolution_indices=tuple(rev_indices),
                    index_ticks=ticks,
                    flux_count=flux_count,
                )
            )
        groups.append(captures)
    return HardSectorGrouping(
        groups=groups,
        sectors_per_rotation=sectors_per_rotation,
        rotated_by=rotated_by,
        index_confidence=confidence,
        index_aligned_flag=index_aligned,
        short_pair_positions=short_pairs,
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
) -> Optional[SectorGuess]:
    """
    Decode one hole's worth of flux into bytes using FM heuristics or PLL.
    """
    flux_parts = [track.decode_flux(idx) for idx in hole_capture.revolution_indices]
    flux: list[int] = []
    for part in flux_parts:
        flux.extend(part)
    if encoding.lower() == "mfm":
        decoded = decode_mfm_bytes(
            flux,
            sample_freq_hz=image.sample_freq_hz,
            index_ticks=hole_capture.index_ticks,
            initial_clock_ticks=initial_clock_ticks,
            clock_adjust=clock_adjust,
        )
    elif use_pll:
        decoded = pll_decode_fm_bytes(
            flux,
            sample_freq_hz=image.sample_freq_hz,
            index_ticks=hole_capture.index_ticks,
            initial_clock_ticks=initial_clock_ticks,
            clock_adjust=clock_adjust,
        )
    else:
        decoded = decode_fm_bytes(flux)

    sync_bytes = (0xA1, 0x4E) if encoding.lower() == "mfm" else (0xA1,)
    sectors = scan_fm_sectors(
        decoded.bytes_out, require_sync=require_sync, sync_bytes=sync_bytes
    )
    return sectors[0] if sectors else None


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
) -> bytes:
    """
    Decode one hole's flux and return raw decoded bytes without sector framing.
    """
    flux: list[int] = []
    for idx in hole_capture.revolution_indices:
        flux.extend(track.decode_flux(idx))
    if encoding.lower() == "mfm":
        decoded = decode_mfm_bytes(
            flux,
            sample_freq_hz=image.sample_freq_hz,
            index_ticks=hole_capture.index_ticks,
            initial_clock_ticks=initial_clock_ticks,
            clock_adjust=clock_adjust,
        )
        return decoded.bytes_out
    if use_pll:
        decoded = pll_decode_fm_bytes(
            flux,
            sample_freq_hz=image.sample_freq_hz,
            index_ticks=hole_capture.index_ticks,
            initial_clock_ticks=initial_clock_ticks,
            clock_adjust=clock_adjust,
        )
        return decoded.bytes_out
    return decode_fm_bytes(flux).bytes_out


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
) -> Dict[int, SectorGuess]:
    track = image.read_track(track_number)
    if not track:
        return {}
    grouping = group_hard_sectors(
        track,
        sectors_per_rotation=sectors_per_rotation,
        index_aligned=bool(image.header.flags & 0x01),
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
            if not existing or (not existing.crc_ok and guess.crc_ok):
                best[guess.sector_id] = guess
    return {sid: g for sid, g in best.items() if sid < expected_sector_count}
