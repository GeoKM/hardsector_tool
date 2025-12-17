"""
Decoders and heuristics for Wang OIS-100 HS32 FM captures.

The format appears to use 32 hard-sector holes per rotation with an index
pulse splitting one interval (33 recorded entries). Logical sectors span two
hole windows and carry a simple header/check pair rather than IBM-style IDAM
framing.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, Sequence, Tuple

from .fm import FMPhaseCandidate, crc16_ibm, decode_fm_bytes, pll_decode_fm_bytes
from .hardsector import HoleCapture, group_hard_sectors, pair_holes
from .scp import SCPImage, TrackData


@dataclass(frozen=True)
class WangSector:
    track: int
    sector_id: int
    offset: int
    payload: bytes
    checksum: bytes
    checksum_algorithms: tuple[str, ...]


@dataclass(frozen=True)
class DecodedStream:
    bytes_out: bytes
    fm_phase: int
    phase_candidates: dict[int, FMPhaseCandidate]
    invert: bool
    clock_scale: float
    bitcells: tuple[int, ...] | None
    clock_ticks: float | None
    half_cell_ticks: float | None


@dataclass
class RotationSimilarity:
    rotation_index: int
    mean: float
    minimum: float
    maximum: float
    shift: int | None = None
    similarity_to_reference: float | None = None


@dataclass
class SectorReconstruction:
    sector_id: int
    wang_sector: WangSector
    consensus: bytes
    confidence: list[float]
    payload_offset: int
    window_score: float
    window_similarity: float
    payload_entropy: float
    payload_fill_ratio: float
    stream_length: int
    sector_size: int
    hole_shift: int
    kept_rotations: list[int]
    dropped_rotations: list[int]
    rotation_similarity: list[RotationSimilarity]
    reference_rotation: int
    shifts: dict[int, int]
    mean_similarity_kept: float
    phase_consensus: dict[int, bytes]
    chosen_fm_phase: int
    best_transform: "TransformResult | None"
    transform_results: list["TransformResult"]
    phase_window_stats: dict[int, tuple[float, float]]
    phase_warning: str | None


@dataclass(frozen=True)
class ChecksumHit:
    algorithm: str
    stored_endian: str
    reflected: bool
    position: str
    value: int


@dataclass(frozen=True)
class TransformResult:
    phase: int
    invert: bool
    bit_reverse: bool
    payload: bytes
    entropy: float
    fill_ratio: float
    preview_head: str
    preview_tail: str
    checksum_hits: tuple[ChecksumHit, ...]


def bit_reverse_byte(value: int) -> int:
    out = 0
    for i in range(8):
        out = (out << 1) | ((value >> i) & 0x01)
    return out


def bit_reverse_bytes(data: bytes) -> bytes:
    return bytes(bit_reverse_byte(b) for b in data)


def run_length_histogram(bitcells: Iterable[int], limit: int = 16) -> dict[str, int]:
    hist: Counter[int] = Counter()
    prev: int | None = None
    run = 0
    for bit in bitcells:
        if bit == prev:
            run += 1
        else:
            if prev is not None:
                hist[min(run, limit)] += 1
            prev = bit
            run = 1
    if prev is not None:
        hist[min(run, limit)] += 1
    return {str(k): v for k, v in sorted(hist.items())}


def _byte_entropy(values: bytes) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    entropy = 0.0
    for count in counts.values():
        p = count / len(values)
        entropy -= p * math.log2(p)
    return entropy


def _byte_fill_ratio(values: bytes) -> float:
    if not values:
        return 1.0
    counts = Counter(values)
    zero_ff = counts.get(0x00, 0) + counts.get(0xFF, 0)
    top = max(counts.values())
    return max(zero_ff, top) / len(values)


def _top_frequencies(values: bytes, limit: int = 6) -> list[tuple[str, int]]:
    counts = Counter(values)
    return [(f"{val:02x}", count) for val, count in counts.most_common(limit)]


def _preview_hex(values: bytes, span: int = 32) -> tuple[str, str]:
    head = values[:span]
    tail = values[-span:] if len(values) > span else b""
    return head.hex(), tail.hex()


def _crc16_non_reflected(data: bytes, poly: int, initial: int, xorout: int = 0) -> int:
    crc = initial
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc ^ xorout


def _crc16_reflected(data: bytes, poly: int, initial: int, xorout: int = 0) -> int:
    crc = initial
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
        crc &= 0xFFFF
    return crc ^ xorout


def sweep_16bit_checks(payload: bytes) -> list[ChecksumHit]:
    if len(payload) < 2:
        return []
    body = payload[:-2]
    stored = payload[-2:]
    stored_last1 = payload[-1]
    stored_fields = [
        ("last2", "big", int.from_bytes(stored, "big")),
        ("last2-swapped", "little", int.from_bytes(stored, "little")),
    ]
    algorithms: list[tuple[str, int, bool]] = [
        ("crc16-ccitt-false", _crc16_non_reflected(body, 0x1021, 0xFFFF), False),
        ("crc16-xmodem", _crc16_non_reflected(body, 0x1021, 0x0000), False),
        ("crc16-ibm-arc", _crc16_reflected(body, 0xA001, 0x0000), True),
        ("crc16-x25", _crc16_reflected(body, 0x8408, 0xFFFF, 0xFFFF), True),
        ("crc16-ccitt-ibm", crc16_ibm(body), False),
        ("sum16", sum(body) & 0xFFFF, False),
        ("sum16-ones-complement", (~sum(body)) & 0xFFFF, False),
        ("xor16", bytes_to_int_xor(body), False),
    ]
    hits: list[ChecksumHit] = []
    for position, endian, stored_value in stored_fields:
        for name, computed, reflected in algorithms:
            if computed == stored_value:
                hits.append(
                    ChecksumHit(
                        algorithm=name,
                        stored_endian=endian,
                        reflected=reflected,
                        position=position,
                        value=computed,
                    )
                )
    lrc8 = ((-sum(body)) & 0xFF) if body else 0
    if lrc8 == stored_last1:
        hits.append(
            ChecksumHit(
                algorithm="lrc8",
                stored_endian="byte",
                reflected=False,
                position="last1",
                value=lrc8,
            )
        )
    xor8 = 0
    for b in body:
        xor8 ^= b
    if xor8 == stored_last1:
        hits.append(
            ChecksumHit(
                algorithm="xor8",
                stored_endian="byte",
                reflected=False,
                position="last1",
                value=xor8,
            )
        )
    return hits


def bytes_to_int_xor(data: bytes) -> int:
    value = 0
    for b in data:
        value ^= b
    return value & 0xFFFF


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    crc = initial
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def checksum_hits(body: bytes, expected: bytes) -> tuple[str, ...]:
    if len(expected) != 2:
        return ()
    payload = body + expected
    hits = sweep_16bit_checks(payload)
    return tuple({hit.algorithm for hit in hits})


def _window_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    entropy = 0.0
    for count in counts.values():
        p = count / len(data)
        entropy -= p * math.log2(p)
    return entropy


def _fill_ratio_ff00(data: bytes) -> float:
    if not data:
        return 1.0
    counts = Counter(data)
    zero_ff = counts.get(0x00, 0) + counts.get(0xFF, 0)
    top = max(counts.values())
    return max(zero_ff, top) / len(data)


def _phase_ones_ratio_near_offset(
    stream: DecodedStream,
    candidate: FMPhaseCandidate,
    byte_offset: int,
    window_bytes: int = 64,
) -> tuple[float, float] | None:
    if stream.bitcells is None or byte_offset < 0:
        return None
    data_bit_index = candidate.data_bit_offset + byte_offset * 8
    if data_bit_index < 0:
        return None
    total_bits = max(1, window_bytes * 8)
    start_data = candidate.phase + 2 * data_bit_index
    end_data = start_data + total_bits * 2
    if start_data >= len(stream.bitcells):
        return None
    data_slice = stream.bitcells[start_data:end_data:2]
    clock_start = (candidate.phase ^ 1) + 2 * data_bit_index
    clock_slice = stream.bitcells[clock_start : clock_start + total_bits * 2 : 2]
    if not data_slice or not clock_slice:
        return None
    return (sum(data_slice) / len(data_slice), sum(clock_slice) / len(clock_slice))


def similarity(a: bytes, b: bytes) -> float:
    """Return fraction of matching bytes over the overlapping region.

    A small overlap yields 0.0 to avoid noisy agreements.
    """

    overlap = min(len(a), len(b))
    if overlap < 4:
        return 0.0
    matches = sum(1 for i in range(overlap) if a[i] == b[i])
    return matches / overlap if overlap else 0.0


def best_shift(ref: bytes, candidate: bytes, max_shift: int = 64) -> tuple[int, float]:
    """Find the shift that maximizes similarity between two streams.

    The returned shift moves ``candidate`` relative to ``ref``: a positive
    value means the candidate starts later. The similarity is measured only on
    the overlapping region after applying the shift.
    """

    best = (0, similarity(ref, candidate))
    for shift in range(-max_shift, max_shift + 1):
        ref_start = max(0, shift)
        cand_start = max(0, -shift)
        overlap = min(len(ref) - ref_start, len(candidate) - cand_start)
        if overlap <= 0:
            continue
        matches = sum(
            1 for i in range(overlap) if ref[ref_start + i] == candidate[cand_start + i]
        )
        score = matches / overlap if overlap else 0.0
        if score > best[1]:
            best = (shift, score)
    return best


def scan_wang_frames(
    stream: bytes, track: int, expected_sectors: int = 16, sector_size: int = 256
) -> list[WangSector]:
    frames: list[WangSector] = []
    max_offset = max(0, len(stream) - (sector_size + 4))
    for offset in range(max_offset + 1):
        trk = stream[offset]
        sec = stream[offset + 1]
        if sec >= expected_sectors:
            continue
        if abs(trk - track) > 1:
            continue
        data_start = offset + 2
        data_end = data_start + sector_size
        check_end = data_end + 2
        if check_end > len(stream):
            break
        payload = stream[data_start:data_end]
        checksum = stream[data_end:check_end]
        body = stream[offset : check_end - 2]
        hits = checksum_hits(body, checksum)
        frames.append(
            WangSector(
                track=trk,
                sector_id=sec,
                offset=offset,
                payload=payload,
                checksum=checksum,
                checksum_algorithms=hits,
            )
        )
    return frames


def _decode_capture(
    image: SCPImage,
    track: TrackData,
    capture: HoleCapture,
    use_pll: bool = True,
    clock_adjust: float = 0.10,
    invert: bool = False,
    clock_scale: float = 1.0,
) -> DecodedStream:
    flux: list[int] = []
    for idx in capture.revolution_indices:
        flux.extend(track.decode_flux(idx))
    if clock_scale != 1.0:
        flux = [max(1, int(interval * clock_scale)) for interval in flux]

    if use_pll:
        result = pll_decode_fm_bytes(
            flux,
            sample_freq_hz=image.sample_freq_hz,
            index_ticks=capture.index_ticks,
            clock_adjust=clock_adjust,
            invert=invert,
        )
        candidates = result.phase_candidates or ()
        candidate_map = {cand.phase: cand for cand in candidates}
        if not candidate_map:
            sample = result.bytes_out[:1024]
            entropy = _window_entropy(sample)
            fill = _fill_ratio_ff00(sample)
            candidate_map = {
                result.fm_phase: FMPhaseCandidate(
                    phase=result.fm_phase,
                    bit_shift=result.bit_shift,
                    data_bit_offset=0,
                    bytes_out=result.bytes_out,
                    entropy=entropy,
                    fill_ratio=fill,
                    data_ones_ratio=fill,
                    clock_ones_ratio=fill,
                    score=entropy - fill * 2.0,
                    window_bits=len(sample) * 8,
                    window_start_bit=0,
                    sample_size=len(sample),
                )
            }
        return DecodedStream(
            bytes_out=result.bytes_out,
            fm_phase=result.fm_phase,
            phase_candidates=candidate_map,
            invert=invert,
            clock_scale=clock_scale,
            bitcells=result.bitcells,
            clock_ticks=result.clock_ticks,
            half_cell_ticks=result.half_cell_ticks,
        )

    decoded = decode_fm_bytes(flux)
    sample = decoded.bytes_out[:1024]
    entropy = _window_entropy(sample)
    fill_ratio = _fill_ratio_ff00(sample)
    fallback = FMPhaseCandidate(
        phase=0,
        bit_shift=decoded.bit_shift,
        data_bit_offset=0,
        bytes_out=decoded.bytes_out,
        entropy=entropy,
        fill_ratio=fill_ratio,
        data_ones_ratio=fill_ratio,
        clock_ones_ratio=fill_ratio,
        score=entropy - fill_ratio * 2.0,
        window_bits=len(sample) * 8,
        window_start_bit=0,
        sample_size=len(sample),
    )
    return DecodedStream(
        bytes_out=decoded.bytes_out,
        fm_phase=0,
        phase_candidates={0: fallback},
        invert=invert,
        clock_scale=clock_scale,
        bitcells=None,
        clock_ticks=None,
        half_cell_ticks=None,
    )


def _pairwise_similarity_matrix(streams: Sequence[DecodedStream]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for i, ref in enumerate(streams):
        row: list[float] = []
        for j, other in enumerate(streams):
            if i == j:
                row.append(1.0)
                continue
            _, score = best_shift(ref.bytes_out, other.bytes_out)
            row.append(score)
        matrix.append(row)
    return matrix


def _rotation_similarity(matrix: list[list[float]]) -> list[RotationSimilarity]:
    stats: list[RotationSimilarity] = []
    for idx, row in enumerate(matrix):
        others = [v for j, v in enumerate(row) if j != idx]
        if not others:
            stats.append(RotationSimilarity(idx, 0.0, 0.0, 0.0))
            continue
        stats.append(
            RotationSimilarity(
                rotation_index=idx,
                mean=sum(others) / len(others),
                minimum=min(others),
                maximum=max(others),
            )
        )
    return stats


def _apply_shift(candidate: bytes, shift: int) -> tuple[int, bytes]:
    if shift >= 0:
        return shift, candidate
    trimmed = candidate[-shift:]
    return 0, trimmed


def consensus_with_confidence(
    streams: Sequence[tuple[int, bytes]],
) -> tuple[bytes, list[float]]:
    if not streams:
        return b"", []
    max_len = max(offset + len(data) for offset, data in streams)
    consensus = bytearray(max_len)
    confidence: list[float] = [0.0] * max_len
    for pos in range(max_len):
        counter: Counter[int] = Counter()
        total = 0
        for offset, data in streams:
            rel = pos - offset
            if 0 <= rel < len(data):
                counter[data[rel]] += 1
                total += 1
        if counter:
            byte, count = counter.most_common(1)[0]
            consensus[pos] = byte
            confidence[pos] = count / total if total else 0.0
        else:
            consensus[pos] = 0
            confidence[pos] = 0.0
    return bytes(consensus), confidence


WINDOW_ENTROPY_WEIGHT = 0.4
WINDOW_FILL_WEIGHT = 1.0
WINDOW_FILL_REJECTION = 0.85


def _window_similarity(
    consensus: bytes,
    aligned_streams: Sequence[tuple[int, bytes]],
    start: int,
    window: int,
) -> float:
    target = consensus[start : start + window]
    if not target:
        return 0.0
    sims: list[float] = []
    for offset, data in aligned_streams:
        rel_start = start - offset
        rel_end = rel_start + window
        if rel_end <= 0 or rel_start >= len(data):
            continue
        data_start = max(rel_start, 0)
        data_end = min(rel_end, len(data))
        consensus_start = data_start - rel_start
        overlap = data_end - data_start
        if overlap <= 0 or consensus_start >= len(target):
            continue
        segment = target[consensus_start : consensus_start + overlap]
        if not segment:
            continue
        sims.append(similarity(segment, data[data_start:data_end]))
    return sum(sims) / len(sims) if sims else 0.0


def _select_payload_window(
    consensus: bytes, aligned_streams: Sequence[tuple[int, bytes]], window: int
) -> tuple[int, float, float, float, float]:
    if window <= 0 or not consensus:
        return 0, 0.0, 0.0, 0.0, 1.0

    best_accept: tuple[float, int, float, float, float] | None = None
    best_reject: tuple[float, int, float, float, float] | None = None

    for start in range(0, max(1, len(consensus) - window + 1)):
        window_bytes = consensus[start : start + window]
        fill_ratio = _fill_ratio_ff00(window_bytes)
        entropy = _window_entropy(window_bytes)
        similarity_score = _window_similarity(consensus, aligned_streams, start, window)
        score = (
            similarity_score
            + WINDOW_ENTROPY_WEIGHT * entropy
            - WINDOW_FILL_WEIGHT * fill_ratio
        )
        bucket = best_accept if fill_ratio <= WINDOW_FILL_REJECTION else best_reject
        if bucket is None or score > bucket[0]:
            entry = (score, start, similarity_score, entropy, fill_ratio)
            if fill_ratio <= WINDOW_FILL_REJECTION:
                best_accept = entry
            else:
                best_reject = entry

    chosen = best_accept or best_reject or (0.0, 0, 0.0, 0.0, 1.0)
    _, offset, similarity_score, entropy, fill_ratio = chosen
    return offset, chosen[0], similarity_score, entropy, fill_ratio


def _reconstruct_sector(
    streams: Sequence[DecodedStream],
    track_number: int,
    sector_id: int,
    sector_size: int,
    similarity_threshold: float,
    keep_best: int,
    hole_shift: int,
) -> tuple[WangSector | None, SectorReconstruction | None]:
    if not streams:
        return None, None

    matrix = _pairwise_similarity_matrix(streams)
    similarity_stats = _rotation_similarity(matrix)
    reference = max(similarity_stats, key=lambda s: s.mean)
    ref_idx = reference.rotation_index
    ref_stream = streams[ref_idx].bytes_out

    aligned_streams: list[tuple[int, bytes]] = []
    shifts: dict[int, int] = {}
    kept: list[int] = []
    dropped: list[int] = []
    sims_to_ref: dict[int, float] = {}
    for idx, stream in enumerate(streams):
        shift, sim = best_shift(ref_stream, stream.bytes_out)
        sims_to_ref[idx] = sim
        similarity_stats[idx].shift = shift
        similarity_stats[idx].similarity_to_reference = sim
        if idx == ref_idx or sim >= similarity_threshold:
            kept.append(idx)
        else:
            dropped.append(idx)
        shifts[idx] = shift

    if len(kept) < min(len(streams), keep_best):
        ranked = sorted(sims_to_ref.items(), key=lambda kv: kv[1], reverse=True)
        kept = sorted({idx for idx, _ in ranked[:keep_best]} | {ref_idx})
        dropped = [idx for idx in range(len(streams)) if idx not in kept]

    for idx in kept:
        offset, aligned = _apply_shift(streams[idx].bytes_out, shifts[idx])
        aligned_streams.append((offset, aligned))

    consensus, confidence = consensus_with_confidence(aligned_streams)
    phase_consensus: dict[int, bytes] = {}
    for phase in (0, 1):
        phase_streams: list[tuple[int, bytes]] = []
        for idx in kept:
            candidate = streams[idx].phase_candidates.get(phase)
            if not candidate or not candidate.bytes_out:
                continue
            offset, aligned_phase = _apply_shift(candidate.bytes_out, shifts[idx])
            phase_streams.append((offset, aligned_phase))
        if phase_streams:
            consensus_phase, _ = consensus_with_confidence(phase_streams)
            phase_consensus[phase] = consensus_phase
    payload_offset, window_score, window_similarity, payload_entropy, payload_fill = (
        _select_payload_window(consensus, aligned_streams, sector_size)
    )
    payload = consensus[payload_offset : payload_offset + sector_size]
    if len(payload) < sector_size:
        payload = payload.ljust(sector_size, b"\x00")
        payload_entropy = _window_entropy(payload)
        payload_fill = _fill_ratio_ff00(payload)

    phase_window_scores: dict[int, list[tuple[float, float]]] = {}
    phase_warning: str | None = None
    for idx in kept:
        stream = streams[idx]
        rel_offset = payload_offset - shifts[idx]
        for phase, candidate in stream.phase_candidates.items():
            ratios = _phase_ones_ratio_near_offset(
                stream, candidate, rel_offset, window_bytes=64
            )
            if ratios is None:
                continue
            data_ratio, clock_ratio = ratios
            phase_window_scores.setdefault(phase, []).append((data_ratio, clock_ratio))
            if data_ratio > 0.95 and clock_ratio > 0.95:
                phase_warning = (
                    "WARNING: likely still in gap/preamble; need better offset"
                )

    phase_window_stats = {
        phase: (
            statistics.mean(v[0] for v in ratios),
            statistics.mean(v[1] for v in ratios),
        )
        for phase, ratios in phase_window_scores.items()
        if ratios
    }

    rotation_similarity = [
        RotationSimilarity(
            rotation_index=stat.rotation_index,
            mean=stat.mean,
            minimum=stat.minimum,
            maximum=stat.maximum,
            shift=stat.shift,
            similarity_to_reference=stat.similarity_to_reference,
        )
        for stat in similarity_stats
    ]

    chosen_phase = (
        Counter(streams[idx].fm_phase for idx in kept).most_common(1)[0][0]
        if kept
        else 0
    )
    if phase_window_stats:
        chosen_phase = max(
            phase_window_stats.items(),
            key=lambda item: (item[1][1] - item[1][0], item[1][1]),
        )[0]

    phase_payloads: dict[int, bytes] = {}
    for phase, stream_bytes in phase_consensus.items():
        segment = stream_bytes[payload_offset : payload_offset + sector_size]
        if len(segment) < sector_size:
            segment = segment.ljust(sector_size, b"\x00")
        phase_payloads[phase] = segment
    if chosen_phase in phase_payloads:
        payload = phase_payloads[chosen_phase]
        payload_entropy = _window_entropy(payload)
        payload_fill = _fill_ratio_ff00(payload)

    transform_results: list[TransformResult] = []
    for phase, segment in phase_payloads.items():
        for inv in (False, True):
            for br in (False, True):
                transformed = segment
                if inv:
                    transformed = bytes(~b & 0xFF for b in transformed)
                if br:
                    transformed = bit_reverse_bytes(transformed)
                head, tail = _preview_hex(transformed)
                transform_results.append(
                    TransformResult(
                        phase=phase,
                        invert=inv,
                        bit_reverse=br,
                        payload=transformed,
                        entropy=_window_entropy(transformed),
                        fill_ratio=_fill_ratio_ff00(transformed),
                        preview_head=head,
                        preview_tail=tail,
                        checksum_hits=tuple(sweep_16bit_checks(transformed)),
                    )
                )

    def _transform_score(result: TransformResult) -> tuple[float, float, float]:
        return (
            float(len(result.checksum_hits)),
            -result.fill_ratio,
            result.entropy,
        )

    best_transform = (
        max(transform_results, key=_transform_score) if transform_results else None
    )
    checksum_algorithms: tuple[str, ...] = ()
    if best_transform and best_transform.checksum_hits:
        checksum_algorithms = tuple(
            {
                f"p{best_transform.phase}_inv{int(best_transform.invert)}_br"
                f"{int(best_transform.bit_reverse)}:{hit.algorithm}"
                for hit in best_transform.checksum_hits
            }
        )

    sector = WangSector(
        track=track_number,
        sector_id=sector_id,
        offset=payload_offset,
        payload=payload,
        checksum=b"",
        checksum_algorithms=checksum_algorithms,
    )
    reconstruction = SectorReconstruction(
        sector_id=sector_id,
        wang_sector=sector,
        consensus=consensus,
        confidence=confidence,
        payload_offset=payload_offset,
        window_score=window_score,
        window_similarity=window_similarity,
        payload_entropy=payload_entropy,
        payload_fill_ratio=payload_fill,
        stream_length=len(consensus),
        sector_size=sector_size,
        hole_shift=hole_shift,
        kept_rotations=kept,
        dropped_rotations=dropped,
        rotation_similarity=rotation_similarity,
        reference_rotation=ref_idx,
        shifts=shifts,
        mean_similarity_kept=(
            sum(sims_to_ref[i] for i in kept) / len(kept) if kept else 0.0
        ),
        phase_consensus=phase_consensus,
        chosen_fm_phase=chosen_phase,
        best_transform=best_transform,
        transform_results=transform_results,
        phase_window_stats=phase_window_stats,
        phase_warning=phase_warning,
    )
    return reconstruction.wang_sector, reconstruction


def reconstruct_track(
    image: SCPImage,
    track_number: int,
    sector_size: int = 256,
    sector_sizes: Sequence[int] | None = None,
    logical_sectors: int = 16,
    pair_phase: int = 0,
    clock_adjust: float = 0.10,
    similarity_threshold: float = 0.75,
    keep_best: int = 5,
    dump_raw: Path | None = None,
    hole_shifts: Sequence[int] | None = None,
    invert: bool = False,
    clock_scale: float = 1.0,
) -> tuple[Dict[int, WangSector], dict[int, SectorReconstruction], float]:
    track = image.read_track(track_number)
    if not track:
        return {}, {}, 0.0
    grouping = group_hard_sectors(
        track,
        sectors_per_rotation=32,
        index_aligned=bool(image.header.flags & 0x01),
    )
    if not grouping.groups:
        return {}, {}, 0.0

    size_candidates = list(dict.fromkeys(sector_sizes or (128, 256, sector_size)))
    shift_candidates = list(
        dict.fromkeys(hole_shifts or range(grouping.sectors_per_rotation))
    )
    decoded_cache: dict[tuple[int, tuple[int, ...], float, bool], DecodedStream] = {}

    def decode_pair(rotation_index: int, pair: HoleCapture) -> DecodedStream:
        key = (rotation_index, tuple(pair.revolution_indices), clock_scale, invert)
        if key not in decoded_cache:
            decoded_cache[key] = _decode_capture(
                image,
                track,
                pair,
                use_pll=True,
                clock_adjust=clock_adjust,
                invert=invert,
                clock_scale=clock_scale,
            )
        return decoded_cache[key]

    def reconstruct_for_size(
        streams: dict[int, list[DecodedStream]],
        size: int,
        hole_shift_val: int,
        sector_limit: int | None = None,
    ) -> tuple[Dict[int, WangSector], dict[int, SectorReconstruction], float]:
        results: Dict[int, WangSector] = {}
        reconstructions: dict[int, SectorReconstruction] = {}
        total = 0.0
        processed = 0
        for sid in sorted(streams):
            if sector_limit is not None and processed >= sector_limit:
                break
            sector_streams = streams[sid]
            if not sector_streams:
                continue
            sector, reconstruction = _reconstruct_sector(
                sector_streams,
                track_number=track_number,
                sector_id=sid,
                sector_size=size,
                similarity_threshold=similarity_threshold,
                keep_best=keep_best,
                hole_shift=hole_shift_val,
            )
            if sector and reconstruction:
                results[sid] = sector
                reconstructions[sid] = reconstruction
                total += reconstruction.window_score
                processed += 1
        return results, reconstructions, total

    best_shift_streams: dict[int, list[DecodedStream]] | None = None
    best_shift_score = -math.inf
    best_shift_sector_count = -1
    best_shift = 0
    shift_score_size = (
        sector_size if sector_size in size_candidates else size_candidates[0]
    )

    for hole_shift in shift_candidates:
        per_sector_streams: dict[int, list[DecodedStream]] = {
            i: [] for i in range(logical_sectors)
        }
        for rot_idx, rotation in enumerate(grouping.groups):
            assert (
                len(rotation) == grouping.sectors_per_rotation
            ), "normalize_rotation must yield merged holes"
            paired = pair_holes(
                rotation[hole_shift:] + rotation[:hole_shift], phase=pair_phase
            )
            for pair in paired:
                decoded = decode_pair(rot_idx, pair)
                per_sector_streams[pair.logical_sector_index or 0].append(decoded)

        shift_results, _, shift_score = reconstruct_for_size(
            per_sector_streams,
            shift_score_size,
            hole_shift,
            sector_limit=min(logical_sectors, 4),
        )
        shift_sector_count = len(shift_results)
        if shift_score > best_shift_score or (
            math.isclose(shift_score, best_shift_score)
            and shift_sector_count > best_shift_sector_count
        ):
            best_shift_score = shift_score
            best_shift_sector_count = shift_sector_count
            best_shift_streams = per_sector_streams
            best_shift = hole_shift

    if best_shift_streams is None:
        return {}, {}, 0.0

    best_results: Dict[int, WangSector] | None = None
    best_recon: dict[int, SectorReconstruction] | None = None
    best_streams: dict[int, list[DecodedStream]] | None = None
    best_score = -math.inf
    best_len = -1
    best_size = size_candidates[0]

    for size in size_candidates:
        results, reconstructions, score = reconstruct_for_size(
            best_shift_streams, size, best_shift
        )
        sector_count = len(results)
        if score > best_score or (
            math.isclose(score, best_score) and sector_count > best_len
        ):
            best_score = score
            best_len = sector_count
            best_results = results
            best_recon = reconstructions
            best_streams = best_shift_streams
            best_size = size

    if best_results is None or best_recon is None or best_streams is None:
        return {}, {}, 0.0

    all_streams = [stream for streams in best_streams.values() for stream in streams]
    phase_buckets: dict[int, list[FMPhaseCandidate]] = {0: [], 1: []}
    for stream in all_streams:
        for phase, candidate in stream.phase_candidates.items():
            phase_buckets.setdefault(phase, []).append(candidate)
    phase_stats = {}
    for phase, candidates in phase_buckets.items():
        if not candidates:
            phase_stats[phase] = {
                "entropy": 0.0,
                "fill": 1.0,
                "data_ones": 1.0,
                "clock_ones": 1.0,
            }
            continue
        phase_stats[phase] = {
            "entropy": sum(c.entropy for c in candidates) / len(candidates),
            "fill": sum(c.fill_ratio for c in candidates) / len(candidates),
            "data_ones": sum(c.data_ones_ratio for c in candidates) / len(candidates),
            "clock_ones": sum(c.clock_ones_ratio for c in candidates) / len(candidates),
        }
    median_length = int(
        median(len(stream.bytes_out) for stream in all_streams) if all_streams else 0
    )
    chosen_phase = (
        Counter(stream.fm_phase for stream in all_streams).most_common(1)[0][0]
        if all_streams
        else 0
    )

    def fmt_phase_stats(phase: int) -> str:
        stats = phase_stats.get(
            phase,
            {"entropy": 0.0, "fill": 1.0, "data_ones": 1.0, "clock_ones": 1.0},
        )
        return (
            f"entropy={stats['entropy']:.2f},"
            f" fill={stats['fill']:.3f},"
            f" data_ones={stats['data_ones']:.3f},"
            f" clock_ones={stats['clock_ones']:.3f}"
        )

    print(
        "  FM phase selection: "
        f"chosen={chosen_phase} "
        f"phase0({fmt_phase_stats(0)}) "
        f"phase1({fmt_phase_stats(1)}) "
        f"decoded_len_med={median_length}"
    )
    print(
        f"  Selected sector_size={best_size} hole_shift={best_shift} "
        f"score={best_score:.2f} sectors={best_len}"
    )
    sample_bytes = b"".join(
        stream.phase_candidates.get(chosen_phase, stream).bytes_out[:512]
        for stream in all_streams
    )
    decoded_fill = _fill_ratio_ff00(sample_bytes)
    decoded_entropy = _window_entropy(sample_bytes)
    data_phase_ones = phase_stats.get(chosen_phase, {}).get("data_ones", 1.0)
    clock_phase_ones = phase_stats.get(chosen_phase, {}).get("clock_ones", 1.0)
    print(
        "  Decoded bytes stats: "
        f"fill00ff={decoded_fill:.3f} entropy={decoded_entropy:.2f} "
        f"ones_ratio_data_phase={data_phase_ones:.3f} "
        f"ones_ratio_clock_phase={clock_phase_ones:.3f}"
    )
    if decoded_fill > 0.85 and decoded_entropy < 1.0:
        print(
            "   WARNING: decoded bytes look like gap/clock (mostly FF/00); "
            "likely wrong phase or wrong window offset"
        )

    if dump_raw:
        prefix = dump_raw
        if dump_raw.is_dir():
            prefix = dump_raw / f"track{track_number:03d}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        reference_sector = min(best_recon) if best_recon else None
        dump_stream: DecodedStream | None = None
        if reference_sector is not None and best_streams:
            sector_streams = best_streams.get(reference_sector, [])
            ref_rotation = best_recon[reference_sector].reference_rotation
            if 0 <= ref_rotation < len(sector_streams):
                dump_stream = sector_streams[ref_rotation]
        if dump_stream and dump_stream.bitcells:
            bitcells = dump_stream.bitcells
            phase0 = bitcells[0::2]
            phase1 = bitcells[1::2]
            Path(f"{prefix}_phase0_bitcells.bin").write_bytes(bytes(phase0))
            Path(f"{prefix}_phase1_bitcells.bin").write_bytes(bytes(phase1))
            stats = {
                "ones_ratio_phase0": sum(phase0) / len(phase0) if phase0 else 0.0,
                "ones_ratio_phase1": sum(phase1) / len(phase1) if phase1 else 0.0,
                "runlen_histogram": run_length_histogram(bitcells),
                "estimated_halfcell_us": (
                    (dump_stream.half_cell_ticks / image.sample_freq_hz) * 1_000_000
                    if dump_stream.half_cell_ticks
                    else None
                ),
                "clock_scale": clock_scale,
                "bitcell_count": len(bitcells),
            }
            Path(f"{prefix}_phase_stats.json").write_text(json.dumps(stats, indent=2))
            for phase in (0, 1):
                candidate = dump_stream.phase_candidates.get(phase)
                if not candidate:
                    continue
                base_bytes = candidate.bytes_out
                Path(f"{prefix}_phase{phase}_data_bytes.bin").write_bytes(base_bytes)
                for inv in (0, 1):
                    for br in (0, 1):
                        transformed = base_bytes
                        if inv:
                            transformed = bytes(~b & 0xFF for b in transformed)
                        if br:
                            transformed = bit_reverse_bytes(transformed)
                        Path(
                            f"{prefix}_p{phase}_inv{inv}_br{br}_data_bytes.bin"
                        ).write_bytes(transformed)
                        head, tail = _preview_hex(transformed)
                        Path(
                            f"{prefix}_p{phase}_inv{inv}_br{br}_stats.json"
                        ).write_text(
                            json.dumps(
                                {
                                    "length": len(transformed),
                                    "entropy": _byte_entropy(transformed),
                                    "fill_ratio_00ff": _byte_fill_ratio(transformed),
                                    "top_frequencies": _top_frequencies(transformed),
                                    "preview_head": head,
                                    "preview_tail": tail,
                                },
                                indent=2,
                            )
                        )
        dump_dir = dump_raw if dump_raw.is_dir() else dump_raw.parent
        for sid, reconstruction in best_recon.items():
            base = dump_dir / f"track{track_number:03d}_sector{sid:02d}"
            base.parent.mkdir(parents=True, exist_ok=True)
            (base.parent / f"{base.name}_consensus.bin").write_bytes(
                reconstruction.consensus
            )
            (base.parent / f"{base.name}_payload.bin").write_bytes(
                reconstruction.wang_sector.payload
            )
            conf_text = "\n".join(f"{c:.3f}" for c in reconstruction.confidence)
            (base.parent / f"{base.name}_confidence.txt").write_text(conf_text)
            (base.parent / f"{base.name}_stream.bin").write_bytes(
                reconstruction.consensus
            )
            for phase, stream_bytes in reconstruction.phase_consensus.items():
                (base.parent / f"{base.name}_stream_phase{phase}.bin").write_bytes(
                    stream_bytes
                )
            meta = {
                "track": track_number,
                "sector": sid,
                "sector_size": reconstruction.sector_size,
                "hole_shift": reconstruction.hole_shift,
                "chosen_fm_phase": reconstruction.chosen_fm_phase,
                "invert": invert,
                "clock_scale": clock_scale,
                "payload_offset": reconstruction.payload_offset,
                "window_score": reconstruction.window_score,
                "window_similarity": reconstruction.window_similarity,
                "payload_entropy": reconstruction.payload_entropy,
                "payload_fill_ratio": reconstruction.payload_fill_ratio,
                "stream_length": reconstruction.stream_length,
                "phase_window_stats": reconstruction.phase_window_stats,
                "phase_warning": reconstruction.phase_warning,
                "best_transform": (
                    {
                        "phase": reconstruction.best_transform.phase,
                        "invert": reconstruction.best_transform.invert,
                        "bit_reverse": reconstruction.best_transform.bit_reverse,
                        "hits": [
                            hit.algorithm
                            for hit in reconstruction.best_transform.checksum_hits
                        ],
                    }
                    if reconstruction.best_transform
                    else None
                ),
            }
            (base.parent / f"{base.name}_meta.json").write_text(
                json.dumps(meta, indent=2)
            )

    return best_results, best_recon, best_score


def summarize_wang_map(wang_map: Dict[int, WangSector]) -> str:
    if not wang_map:
        return "no Wang/OIS sectors"
    alg_hits: Counter[str] = Counter()
    for sector in wang_map.values():
        for alg in sector.checksum_algorithms:
            alg_hits[alg] += 1
    alg_part = (
        ", checksums: "
        + ", ".join(f"{alg} x{count}" for alg, count in alg_hits.most_common())
        if alg_hits
        else ""
    )
    ids = ",".join(str(s) for s in sorted(wang_map))
    return f"sectors {ids}{alg_part}"


def detect_header_candidates(
    reconstructions: dict[int, SectorReconstruction],
    logical_sectors: int,
    track_number: int,
) -> dict[str, list[int]]:
    if not reconstructions:
        return {}
    lengths = [len(rec.consensus) for rec in reconstructions.values() if rec.consensus]
    if not lengths:
        return {}
    min_len = min(lengths)
    if min_len == 0:
        return {}
    track_offsets: list[int] = []
    sector_offsets: list[int] = []
    for offset in range(min_len):
        values = [
            rec.consensus[offset]
            for rec in reconstructions.values()
            if len(rec.consensus) > offset
        ]
        if not values:
            continue
        if track_number > 0 and all(v == track_number for v in values):
            track_offsets.append(offset)
        uniq = set(values)
        if uniq.issubset(set(range(logical_sectors))) and len(uniq) >= min(
            4, logical_sectors
        ):
            sector_offsets.append(offset)
    return {"track": track_offsets, "sector": sector_offsets}
