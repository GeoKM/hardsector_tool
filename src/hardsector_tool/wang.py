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


def majority_vote_byte_columns(columns: Sequence[bytes]) -> bytes:
    if not columns:
        return b""
    max_len = max(len(col) for col in columns)
    voted = bytearray()
    for i in range(max_len):
        counter: Counter[int] = Counter()
        for col in columns:
            if i < len(col):
                counter[col[i]] += 1
        if counter:
            voted.append(counter.most_common(1)[0][0])
    return bytes(voted)


def align_stream(
    stream: bytes, track: int, sector_id: int, max_offset: int = 512
) -> int:
    best_off = 0
    best_score = -1
    limit = min(max_offset, max(0, len(stream) - 4))
    for off in range(limit + 1):
        if off + 2 >= len(stream):
            break
        trk = stream[off]
        sec = stream[off + 1]
        score = 0
        if trk == track:
            score += 3
        elif abs(trk - track) <= 1:
            score += 1
        if sec == sector_id:
            score += 3
        elif sec < 32:
            score += 1
        score += min(2, len(stream) - off)
        if score > best_score:
            best_score = score
            best_off = off
    return best_off


def reconstruct_track(
    image: SCPImage,
    track_number: int,
    sector_size: int = 256,
    logical_sectors: int = 16,
    pair_phase: int = 0,
    clock_adjust: float = 0.10,
) -> Dict[int, WangSector]:
    track = image.read_track(track_number)
    if not track:
        return {}
    grouping = group_hard_sectors(
        track,
        sectors_per_rotation=32,
        index_aligned=bool(image.header.flags & 0x01),
    )
    if not grouping.groups:
        return {}

    per_sector_streams: dict[int, list[bytes]] = {i: [] for i in range(logical_sectors)}
    for rotation in grouping.groups:
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
    for sid, streams in per_sector_streams.items():
        if not streams:
            continue
        offsets = [align_stream(stream, track_number, sid) for stream in streams]
        aligned_chunks: list[bytes] = []
        for stream, off in zip(streams, offsets, strict=False):
            chunk = stream[off : off + sector_size + 4]
            if chunk:
                aligned_chunks.append(chunk)
        if not aligned_chunks:
            continue
        voted = majority_vote_byte_columns(aligned_chunks)
        frames = scan_wang_frames(
            voted,
            track_number,
            expected_sectors=logical_sectors,
            sector_size=sector_size,
        )
        if not frames:
            continue
        best = max(frames, key=lambda f: (len(f.checksum_algorithms), -f.offset))
        results[sid] = best
    return results


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
