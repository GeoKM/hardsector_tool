"""
Decoders and heuristics for Wang OIS-100 HS32 FM captures.

The format appears to use 32 hard-sector holes per rotation with an index
pulse splitting one interval (33 recorded entries). Logical sectors span two
hole windows and carry a simple header/check pair rather than IBM-style IDAM
framing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

from .fm import crc16_ibm, decode_fm_bytes, pll_decode_fm_bytes
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
    kept_rotations: list[int]
    dropped_rotations: list[int]
    rotation_similarity: list[RotationSimilarity]
    reference_rotation: int
    shifts: dict[int, int]
    mean_similarity_kept: float


ChecksumFunc = Tuple[str, Callable[[bytes], int]]


CHECKSUM_CANDIDATES: tuple[ChecksumFunc, ...] = (
    ("sum16-be", lambda data: sum(data) & 0xFFFF),
    ("sum16-le", lambda data: int.from_bytes(sum(data).to_bytes(2, "little"), "big")),
    ("ones-complement", lambda data: (~sum(data)) & 0xFFFF),
    ("xor16", lambda data: bytes_to_int_xor(data)),
    ("crc16-ibm", lambda data: crc16_ibm(data)),
    ("crc16-ccitt", lambda data: crc16_ccitt(data)),
)


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
    exp_val = int.from_bytes(expected, "big")
    hits: list[str] = []
    for name, func in CHECKSUM_CANDIDATES:
        try:
            if func(body) == exp_val:
                hits.append(name)
        except Exception:
            continue
    return tuple(hits)


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
) -> bytes:
    if use_pll:
        result = pll_decode_fm_bytes(
            [
                interval
                for idx in capture.revolution_indices
                for interval in track.decode_flux(idx)
            ],
            sample_freq_hz=image.sample_freq_hz,
            index_ticks=capture.index_ticks,
            clock_adjust=clock_adjust,
        )
        return result.bytes_out
    return decode_fm_bytes(
        [
            interval
            for idx in capture.revolution_indices
            for interval in track.decode_flux(idx)
        ]
    ).bytes_out


def _pairwise_similarity_matrix(streams: Sequence[bytes]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for i, ref in enumerate(streams):
        row: list[float] = []
        for j, other in enumerate(streams):
            if i == j:
                row.append(1.0)
                continue
            _, score = best_shift(ref, other)
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


def _best_confidence_window(
    confidence: Sequence[float], window: int
) -> tuple[int, float]:
    if not confidence or window <= 0:
        return 0, 0.0
    best_off, best_score = 0, -1.0
    for start in range(0, max(1, len(confidence) - window + 1)):
        window_conf = confidence[start : start + window]
        score = sum(window_conf) / window
        if score > best_score:
            best_score = score
            best_off = start
    return best_off, best_score


def _reconstruct_sector(
    streams: Sequence[bytes],
    track_number: int,
    sector_id: int,
    sector_size: int,
    similarity_threshold: float,
    keep_best: int,
) -> tuple[WangSector | None, SectorReconstruction | None]:
    if not streams:
        return None, None

    matrix = _pairwise_similarity_matrix(streams)
    similarity_stats = _rotation_similarity(matrix)
    reference = max(similarity_stats, key=lambda s: s.mean)
    ref_idx = reference.rotation_index
    ref_stream = streams[ref_idx]

    aligned_streams: list[tuple[int, bytes]] = []
    shifts: dict[int, int] = {}
    kept: list[int] = []
    dropped: list[int] = []
    sims_to_ref: dict[int, float] = {}
    for idx, stream in enumerate(streams):
        shift, sim = best_shift(ref_stream, stream)
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
        offset, aligned = _apply_shift(streams[idx], shifts[idx])
        aligned_streams.append((offset, aligned))

    consensus, confidence = consensus_with_confidence(aligned_streams)
    payload_offset, window_score = _best_confidence_window(confidence, sector_size)
    payload = consensus[payload_offset : payload_offset + sector_size]
    if len(payload) < sector_size:
        payload = payload.ljust(sector_size, b"\x00")

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

    sector = WangSector(
        track=track_number,
        sector_id=sector_id,
        offset=payload_offset,
        payload=payload,
        checksum=b"",
        checksum_algorithms=(),
    )
    reconstruction = SectorReconstruction(
        sector_id=sector_id,
        wang_sector=sector,
        consensus=consensus,
        confidence=confidence,
        payload_offset=payload_offset,
        window_score=window_score,
        kept_rotations=kept,
        dropped_rotations=dropped,
        rotation_similarity=rotation_similarity,
        reference_rotation=ref_idx,
        shifts=shifts,
        mean_similarity_kept=(
            sum(sims_to_ref[i] for i in kept) / len(kept) if kept else 0.0
        ),
    )
    return reconstruction.wang_sector, reconstruction


def reconstruct_track(
    image: SCPImage,
    track_number: int,
    sector_size: int = 256,
    logical_sectors: int = 16,
    pair_phase: int = 0,
    clock_adjust: float = 0.10,
    similarity_threshold: float = 0.75,
    keep_best: int = 5,
    dump_raw: Path | None = None,
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

    per_sector_streams: dict[int, list[bytes]] = {i: [] for i in range(logical_sectors)}
    for rot_idx, rotation in enumerate(grouping.groups):
        short_pair = None
        if grouping.short_pair_positions:
            short_pair = grouping.short_pair_positions[rot_idx]
        pair_label = (
            f"({short_pair},{(short_pair + 1) % (grouping.sectors_per_rotation + 1)})"
            if short_pair is not None
            else "None"
        )
        print(
            f"  rotation {rot_idx}: short_pair={pair_label} merged_len={len(rotation)} (expected 32)"
        )
        assert len(rotation) == 32, "normalize_rotation must yield 32 merged holes"
        paired = pair_holes(rotation, phase=pair_phase)
        for pair in paired:
            raw_bytes = _decode_capture(
                image,
                track,
                pair,
                use_pll=True,
                clock_adjust=clock_adjust,
            )
            per_sector_streams[pair.logical_sector_index or 0].append(raw_bytes)

    results: Dict[int, WangSector] = {}
    reconstructions: dict[int, SectorReconstruction] = {}
    total_score = 0.0
    for sid, streams in per_sector_streams.items():
        sector, reconstruction = _reconstruct_sector(
            streams,
            track_number=track_number,
            sector_id=sid,
            sector_size=sector_size,
            similarity_threshold=similarity_threshold,
            keep_best=keep_best,
        )
        if sector and reconstruction:
            results[sid] = sector
            reconstructions[sid] = reconstruction
            total_score += reconstruction.window_score
            if dump_raw:
                base = dump_raw / f"track{track_number:03d}_sector{sid:02d}"
                base.parent.mkdir(parents=True, exist_ok=True)
                (base.with_suffix("_consensus.bin")).write_bytes(
                    reconstruction.consensus
                )
                (base.with_suffix("_payload.bin")).write_bytes(sector.payload)
                conf_text = "\n".join(f"{c:.3f}" for c in reconstruction.confidence)
                (base.with_suffix("_confidence.txt")).write_text(conf_text)
    return results, reconstructions, total_score


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
