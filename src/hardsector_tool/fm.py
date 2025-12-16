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
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class FMDecodeResult:
    bytes_out: bytes
    bit_shift: int
    half_cell_ticks: float
    full_cell_ticks: float
    threshold_ticks: float


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
    )
