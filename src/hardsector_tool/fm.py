"""
Lightweight FM decoder utilities for flux timings.

The goal is to extract a plausible byte stream from FM-encoded flux data
without hardware timing context. This is intentionally heuristic: it
estimates half/full cell durations from the flux histogram and then
classifies each interval as either a full cell (data 0) or a pair of
half cells (data 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

DEFAULT_CLOCK_ADJ = 0.10
PLL_PERIOD_ADJ = 0.05
PLL_PHASE_ADJ = 0.60


@dataclass(frozen=True)
class FMDecodeResult:
    bytes_out: bytes
    bit_shift: int
    half_cell_ticks: float
    full_cell_ticks: float
    threshold_ticks: float
    method: str = "heuristic"


@dataclass(frozen=True)
class PLLDecodeResult:
    bytes_out: bytes
    bit_shift: int
    initial_clock_ticks: float
    clock_ticks: float
    method: str = "pll"


@dataclass(frozen=True)
class MFMDecodeResult:
    bytes_out: bytes
    bit_shift: int
    method: str = "mfm"


@dataclass(frozen=True)
class SectorGuess:
    offset: int
    track: int
    head: int
    sector_id: int
    size_code: int
    length: int
    crc_ok: bool
    id_crc_ok: bool
    data_crc_ok: bool
    data: Optional[bytes] = None


def scan_data_marks(
    byte_stream: bytes, marks: Sequence[int] = (0xFB, 0xFA)
) -> List[Tuple[int, int]]:
    """
    Scan for data mark bytes (e.g., 0xFB/0xFA) without CRC validation.
    Returns list of (offset, mark) pairs.
    """
    hits: List[Tuple[int, int]] = []
    for i, b in enumerate(byte_stream):
        if b in marks:
            hits.append((i, b))
    return hits


def scan_bit_patterns(
    bits: Sequence[int], patterns: Sequence[int] = (0xFB, 0xFA, 0xA1, 0xFE)
) -> List[Tuple[int, int, int]]:
    """
    Scan bitcell stream for byte patterns across all bit shifts.
    Returns list of (shift, byte_offset, value).
    """
    hits: List[Tuple[int, int, int]] = []
    for shift in range(8):
        as_bytes = bits_to_bytes(bits[shift:])
        for idx, b in enumerate(as_bytes):
            if b in patterns:
                hits.append((shift, idx, b))
    return hits


def _kmeans_1d(values: Sequence[float]) -> Tuple[float, float]:
    """
    Simple two-cluster k-means for 1D values.

    Returns the two cluster centers sorted ascending.
    """

    if not values:
        raise ValueError("Cannot cluster empty input")

    low, high = min(values), max(values)
    if math.isclose(low, high):
        return low, high

    centers = [low, high]
    for _ in range(8):
        clusters = ([], [])
        for val in values:
            idx = 0 if abs(val - centers[0]) <= abs(val - centers[1]) else 1
            clusters[idx].append(val)
        new_centers = []
        for idx, cluster in enumerate(clusters):
            if cluster:
                new_centers.append(sum(cluster) / len(cluster))
            else:
                new_centers.append(centers[idx])
        if all(math.isclose(a, b) for a, b in zip(centers, new_centers, strict=True)):
            break
        centers = new_centers
    return tuple(sorted(centers))  # type: ignore[return-value]


def estimate_cell_ticks(flux: Sequence[int]) -> Tuple[float, float, float]:
    """
    Estimate half and full cell durations from a flux interval distribution.

    Hard-sector slices often include long gaps that skew naive medians. We
    trim long tails and cluster log-scaled intervals into half/full peaks.
    """
    if not flux:
        raise ValueError("Flux list is empty")

    values = sorted(flux)
    if len(values) > 8:
        trim = max(1, int(len(values) * 0.2))
        values = values[: len(values) - trim]

    logs = [math.log(v) for v in values if v > 0]
    if len(logs) < 2:
        half_med = full_med = float(values[0])
    else:
        c1, c2 = _kmeans_1d(logs)
        half_med, full_med = sorted((math.exp(c1), math.exp(c2)))

    threshold = (
        (half_med + full_med) / 2.0 if half_med and full_med else float(values[0])
    )
    return half_med, full_med, threshold


def find_sync_bytes(
    bits: Sequence[int], pattern: int = 0xA1, window: int = 32
) -> List[int]:
    """
    Locate occurrences of a sync/address mark byte (e.g., 0xA1) that may
    include missing clock bits, by matching against bit-pattern windows.

    This is a heuristic: we scan the bits for the pattern and allow one
    missing clock (i.e., a repeated '0') within the last two bitcells.
    """
    indices: List[int] = []
    for i in range(0, max(0, len(bits) - 8)):
        window_bits = bits[i : i + 8]
        as_byte = 0
        for b in window_bits:
            as_byte = (as_byte << 1) | (1 if b else 0)
        if as_byte == pattern:
            indices.append(i)
            continue
        # Allow a single missing clock by tolerating a doubled zero near the end.
        if len(window_bits) >= 8:
            adjusted = window_bits.copy()
            for j in range(len(adjusted) - 2, len(adjusted)):
                if adjusted[j] == 0:
                    adjusted[j] = 1
                    as_adj = 0
                    for b in adjusted:
                        as_adj = (as_adj << 1) | (1 if b else 0)
                    if as_adj == pattern:
                        indices.append(i)
                        break
    return indices


def mfm_bytes_from_bitcells(bitcells: Sequence[int]) -> Tuple[int, bytes]:
    """
    Convert a bitcell stream into data bytes by trying both even/odd
    alignments (clock/data alternation) and picking the one that yields
    more sync-like bytes (0xA1/0x4E/0xF6).
    """
    sync_candidates = {0xA1, 0x4E, 0xF6}
    best_shift = 0
    best_bytes = b""
    best_score = -1

    for start in (0, 1):
        data_bits = bitcells[start::2]
        candidate = bits_to_bytes(data_bits)
        score = sum(candidate.count(val) for val in sync_candidates)
        if score > best_score:
            best_score = score
            best_shift = start
            best_bytes = candidate
    return best_shift, best_bytes


def decode_mfm_bytes(
    flux: Sequence[int],
    sample_freq_hz: float,
    index_ticks: int | None = None,
    initial_clock_ticks: float | None = None,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
) -> MFMDecodeResult:
    """
    Decode MFM data by PLL to bitcells then collapsing alternating clock/data bits.
    """
    bitcells = pll_decode_bits(
        flux,
        sample_freq_hz,
        index_ticks=index_ticks,
        initial_clock_ticks=initial_clock_ticks,
        clock_adjust=clock_adjust,
    )
    shift, decoded = mfm_bytes_from_bitcells(bitcells)
    return MFMDecodeResult(bytes_out=decoded, bit_shift=shift)


def decode_fm_bits(flux: Sequence[int], threshold: float | None = None) -> List[int]:
    """
    Decode flux intervals into FM data bits.

    Intervals shorter than the threshold are treated as half cells; pairs of
    half cells map to a data bit of 1. Longer intervals are treated as full
    cells (data bit 0). This is a heuristic and may misclassify noisy samples.
    """
    half_med, full_med, auto_threshold = estimate_cell_ticks(flux)
    use_threshold = threshold if threshold is not None else auto_threshold

    bits: List[int] = []
    i = 0
    while i < len(flux):
        span = flux[i]
        if span >= use_threshold:
            bits.append(0)
            i += 1
        # Attempt to pair with the next short span to represent a data 1.
        elif i + 1 < len(flux) and flux[i + 1] < use_threshold * 1.2:
            bits.append(1)
            i += 2
        else:
            bits.append(0)
            i += 1
    return bits


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    out = bytearray()
    for idx in range(0, len(bits) - 7, 8):
        b = 0
        for bit in bits[idx : idx + 8]:
            b = (b << 1) | (1 if bit else 0)
        out.append(b)
    return bytes(out)


def _longest_constant_run(values: bytes) -> int:
    longest = 0
    current = 0
    prev: int | None = None
    for val in values:
        if val == prev:
            current += 1
        else:
            current = 1
            prev = val
        longest = max(longest, current)
    return longest


def _entropy(values: bytes) -> float:
    if not values:
        return 0.0
    freq = [0] * 256
    for b in values:
        freq[b] += 1
    entropy = 0.0
    for count in freq:
        if count:
            p = count / len(values)
            entropy -= p * math.log2(p)
    return entropy


def best_aligned_bytes(bits: Sequence[int]) -> Tuple[int, bytes]:
    """
    Try all bit shifts (0-7) and pick the byte stream with the most stable content.

    Stability is measured by the dominant fill byte (0x00 or 0xFF), the longest
    constant run, and overall entropy. This avoids bias toward 0xFF-only streams
    when inverted captures produce long 0x00 runs.
    """

    def score_candidate(candidate: bytes) -> Tuple[int, int, float]:
        fill_score = max(candidate.count(0x00), candidate.count(0xFF))
        run_score = _longest_constant_run(candidate)
        entropy_score = -_entropy(candidate)
        return fill_score, run_score, entropy_score

    best_shift = 0
    best_bytes = b""
    best_score: Tuple[int, int, float] = (-1, -1, -math.inf)
    for shift in range(8):
        candidate = bits_to_bytes(bits[shift:])
        score = score_candidate(candidate)
        if score > best_score:
            best_score = score
            best_shift = shift
            best_bytes = candidate
    return best_shift, best_bytes


def decode_fm_bytes(flux: Sequence[int]) -> FMDecodeResult:
    half_med, full_med, threshold = estimate_cell_ticks(flux)
    bits = decode_fm_bits(flux, threshold=threshold)
    shift, decoded = best_aligned_bytes(bits)
    return FMDecodeResult(
        bytes_out=decoded,
        bit_shift=shift,
        half_cell_ticks=half_med,
        full_cell_ticks=full_med,
        threshold_ticks=threshold,
        method="heuristic",
    )


def pll_decode_bits(
    flux: Sequence[int],
    sample_freq_hz: float,
    index_ticks: int | None = None,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
    initial_clock_ticks: float | None = None,
    invert: bool = False,
) -> List[int]:
    """
    Decode flux intervals into bitcells using a simple PLL.

    Adapted from Greaseweazle's public-domain bitcell PLL, but simplified
    for single-chunk processing. Returns 1-bit-per-cell presence of a
    transition; FM/MFM framing happens separately.
    """
    if not flux:
        return []

    half_med, _, _ = estimate_cell_ticks(flux)
    clock_ticks = initial_clock_ticks if initial_clock_ticks else half_med * 2
    clock = clock_ticks / sample_freq_hz
    clock_min = clock * (1 - clock_adjust)
    clock_max = clock * (1 + clock_adjust)
    ticks = 0.0

    to_index = (index_ticks / sample_freq_hz) if index_ticks else None
    bits: List[int] = []

    for interval in flux:
        ticks += interval / sample_freq_hz
        if ticks < clock / 2:
            continue

        zeros = 0
        while True:
            if to_index is not None:
                to_index -= clock
                if to_index < 0:
                    to_index = index_ticks / sample_freq_hz

            ticks -= clock
            if ticks >= clock / 2:
                zeros += 1
                bits.append(0)
            else:
                bits.append(1)
                break

        if zeros <= 3:
            clock += ticks * PLL_PERIOD_ADJ
        else:
            clock += ((half_med * 2) / sample_freq_hz - clock) * PLL_PERIOD_ADJ
        clock = min(max(clock, clock_min), clock_max)
        new_ticks = ticks * (1 - PLL_PHASE_ADJ)
        ticks = new_ticks

    return [0 if b else 1 for b in bits] if invert else bits


def pll_decode_fm_bytes(
    flux: Sequence[int],
    sample_freq_hz: float,
    index_ticks: int | None = None,
    initial_clock_ticks: float | None = None,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
    invert: bool = False,
) -> PLLDecodeResult:
    """
    Decode FM by PLL to bitcells, then map clock/data phases into bytes.
    """

    bitcells = pll_decode_bits(
        flux,
        sample_freq_hz,
        index_ticks=index_ticks,
        initial_clock_ticks=initial_clock_ticks,
        clock_adjust=clock_adjust,
        invert=invert,
    )
    shift, decoded = fm_bytes_from_bitcells(bitcells)
    half_med, _, _ = estimate_cell_ticks(flux)
    clock_ticks = initial_clock_ticks if initial_clock_ticks else half_med * 2
    return PLLDecodeResult(
        bytes_out=decoded,
        bit_shift=shift,
        initial_clock_ticks=clock_ticks,
        clock_ticks=clock_ticks,
    )


def fm_bytes_from_bitcells(bitcells: Sequence[int]) -> Tuple[int, bytes]:
    """
    Convert PLL bitcells into FM data bytes.

    We try both possible clock/data phases, prefer the one with denser clock
    transitions, and then apply byte-alignment heuristics to the resulting
    data-bit stream.
    """

    if not bitcells:
        return 0, b""

    best_phase = 0
    best_shift = 0
    best_bytes = b""
    best_score: Tuple[int, float, int] = (-1, -math.inf, 0)

    for phase in (0, 1):
        clock_bits = bitcells[phase::2]
        data_bits = bitcells[1 - phase :: 2]
        clock_score = sum(clock_bits)
        shift, aligned = best_aligned_bytes(data_bits)
        entropy = _entropy(aligned)
        score = (clock_score, entropy, -_longest_constant_run(aligned))
        if score > best_score:
            best_score = score
            best_phase = phase
            best_shift = shift
            best_bytes = aligned

    overall_shift = (1 - best_phase + best_shift) % 8
    return overall_shift, best_bytes


def brute_force_mark_payloads(
    bits: Sequence[int],
    payload_bytes: int = 256,
    patterns: Sequence[int] = (0xFB, 0xFA, 0xA1, 0xFE),
) -> List[Tuple[int, int, int, bytes]]:
    """
    Extract payload windows following mark-like bytes across all bit shifts.

    Returns a list of (shift, byte_offset, value, payload) tuples where
    payload is payload_bytes long (or shorter at the end of the stream).
    """
    results: List[Tuple[int, int, int, bytes]] = []
    for shift in range(8):
        byte_stream = bits_to_bytes(bits[shift:])
        for idx, val in enumerate(byte_stream):
            if val not in patterns:
                continue
            start_bit = shift + idx * 8
            end_bit = start_bit + payload_bytes * 8
            payload = bits_to_bytes(bits[start_bit:end_bit])
            results.append((shift, idx, val, payload))
    return results


def crc16_ibm(data: bytes) -> int:
    """
    Standard CRC-16-IBM used by IBM/WD floppy formats.
    """
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def scan_fm_sectors(
    byte_stream: bytes,
    require_sync: bool = False,
    sync_bytes: Sequence[int] = (0xA1,),
    dam_bytes: Sequence[int] = (0xFB, 0xFA, 0xF8),
    search_window: int = 512,
) -> List[SectorGuess]:
    """
    Heuristically scan for FM address marks (0xFE) and data fields.

    Assumes standard IBM FM layout with 0xFE IDAM and size codes that map
    to lengths of 128 << size_code. If require_sync is True, only accept
    IDAMs preceded by an 0xA1 sync byte within the previous three bytes.
    Data fields are located by searching for a DAM (0xFB/0xFA/0xF8) within
    `search_window` bytes after the IDAM CRC.
    """
    gap_bytes = {0xFF, 0x00, 0x4E, 0xA1}
    guesses: List[SectorGuess] = []
    i = 0

    def has_gap_leadin(offset: int, min_run: int = 6) -> bool:
        window = byte_stream[max(0, offset - 2 * min_run) : offset]
        if not window:
            return False
        run = 0
        for b in reversed(window):
            if b in gap_bytes:
                run += 1
            else:
                break
        return run >= min_run

    while i < len(byte_stream) - 8:
        mark = byte_stream[i]
        if mark != 0xFE:
            i += 1
            continue

        if not has_gap_leadin(i):
            i += 1
            continue

        if i + 6 >= len(byte_stream):
            break

        if require_sync:
            sync_window = byte_stream[max(0, i - 3) : i]
            if not any(sb in sync_window for sb in sync_bytes):
                i += 1
                continue

        track, head, sector_id, size_code = (
            byte_stream[i + 1],
            byte_stream[i + 2],
            byte_stream[i + 3],
            byte_stream[i + 4],
        )
        if size_code >= 7:
            i += 1
            continue

        expected_len = 128 << size_code
        id_crc_val = int.from_bytes(byte_stream[i + 5 : i + 7], "big")
        id_crc_ok = crc16_ibm(byte_stream[i : i + 5]) == id_crc_val

        data_crc_ok = False
        data_bytes: Optional[bytes] = None
        dam_pos = None
        search_start = i + 7
        search_end = min(len(byte_stream), search_start + search_window)
        for pos in range(search_start, search_end):
            if byte_stream[pos] not in dam_bytes:
                continue
            if require_sync:
                sync_win = byte_stream[max(0, pos - 3) : pos]
                if not any(sb in sync_win for sb in sync_bytes):
                    continue
            dam_pos = pos
            break

        if dam_pos is not None:
            data_start = dam_pos + 1
            data_end = data_start + expected_len + 2
            if data_end <= len(byte_stream):
                data_bytes = byte_stream[data_start : data_end - 2]
                data_crc_val = int.from_bytes(
                    byte_stream[data_end - 2 : data_end], "big"
                )
                data_crc_ok = (
                    crc16_ibm(byte_stream[dam_pos : data_end - 2]) == data_crc_val
                )

        guesses.append(
            SectorGuess(
                offset=i,
                track=track,
                head=head,
                sector_id=sector_id,
                size_code=size_code,
                length=expected_len,
                crc_ok=id_crc_ok and data_crc_ok,
                id_crc_ok=id_crc_ok,
                data_crc_ok=data_crc_ok,
                data=data_bytes,
            )
        )
        i += 1

    return guesses
