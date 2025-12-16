"""
Lightweight FM decoder utilities for flux timings.

The goal is to extract a plausible byte stream from FM-encoded flux data
without hardware timing context. This is intentionally heuristic: it
estimates half/full cell durations from the flux histogram and then
classifies each interval as either a full cell (data 0) or a pair of
half cells (data 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, List, Optional, Sequence, Tuple

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


def estimate_cell_ticks(flux: Sequence[int]) -> Tuple[float, float, float]:
    """
    Estimate half and full cell durations from a flux interval distribution.
    """
    if not flux:
        raise ValueError("Flux list is empty")

    values = sorted(flux)
    mid = len(values) // 2
    half_med = median(values[:mid]) if mid else float(values[0])
    full_med = median(values[mid:]) if mid else float(values[0])
    threshold = (half_med + full_med) / 2.0
    return half_med, full_med, threshold


def find_sync_bytes(bits: Sequence[int], pattern: int = 0xA1, window: int = 32) -> List[int]:
    """
    Locate occurrences of a sync/address mark byte (e.g., 0xA1) that may
    include missing clock bits, by matching against bit-pattern windows.

    This is a heuristic: we scan the bits for the pattern and allow one
    missing clock (i.e., a repeated '0') within the last two bitcells.
    """
    indices: List[int] = []
    target = f"{pattern:08b}"
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
        else:
            # Attempt to pair with the next short span to represent a data 1.
            if i + 1 < len(flux) and flux[i + 1] < use_threshold * 1.2:
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


def best_aligned_bytes(bits: Sequence[int]) -> Tuple[int, bytes]:
    """
    Try all bit shifts (0-7) and pick the byte stream with the most 0xFF bytes.
    This favors alignments that capture sync/preamble runs common in FM.
    """
    best_shift = 0
    best_bytes = b""
    best_score = -1
    for shift in range(8):
        candidate = bits_to_bytes(bits[shift:])
        score = candidate.count(0xFF)
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
    clock = (half_med * 2) / sample_freq_hz
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
                    to_index = (index_ticks / sample_freq_hz)

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

    return bits


def pll_decode_fm_bytes(
    flux: Sequence[int],
    sample_freq_hz: float,
    index_ticks: int | None = None,
) -> PLLDecodeResult:
    bitcells = pll_decode_bits(flux, sample_freq_hz, index_ticks=index_ticks)
    shift, decoded = best_aligned_bytes(bitcells)
    half_med, _, _ = estimate_cell_ticks(flux)
    clock_ticks = half_med * 2
    return PLLDecodeResult(
        bytes_out=decoded,
        bit_shift=shift,
        initial_clock_ticks=clock_ticks,
        clock_ticks=clock_ticks,
    )


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


def scan_fm_sectors(byte_stream: bytes) -> List[SectorGuess]:
    """
    Heuristically scan for FM address marks (0xFE) and data fields.

    Assumes standard IBM FM layout with 0xFE IDAM and size codes that map
    to lengths of 128 << size_code.
    """
    guesses: List[SectorGuess] = []
    i = 0
    while i < len(byte_stream) - 8:
        if byte_stream[i] != 0xFE:
            i += 1
            continue
        if i + 6 >= len(byte_stream):
            break
        track, head, sector_id, size_code = (
            byte_stream[i + 1],
            byte_stream[i + 2],
            byte_stream[i + 3],
            byte_stream[i + 4],
        )
        expected_len = 128 << size_code if size_code < 7 else 0
        id_crc_val = int.from_bytes(byte_stream[i + 5 : i + 7], "big")
        id_crc_ok = crc16_ibm(byte_stream[i : i + 5]) == id_crc_val

        data_start = i + 7
        data_end = data_start + expected_len + 2
        data_crc_ok = False
        data_bytes: Optional[bytes] = None
        if expected_len and data_end <= len(byte_stream):
            data_bytes = byte_stream[data_start : data_end - 2]
            data_crc_val = int.from_bytes(byte_stream[data_end - 2 : data_end], "big")
            data_crc_ok = crc16_ibm(data_bytes) == data_crc_val

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
