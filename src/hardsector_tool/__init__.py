"""
Top-level package for hard sector flux decoding utilities.

This package will collect parsers, analyzers, and CLI helpers for working with
hard-sectored floppy disk flux captures such as the ACMS80217 sample set.
"""

from .fm import (
    FMDecodeResult,
    PLLDecodeResult,
    SectorGuess,
    best_aligned_bytes,
    bits_to_bytes,
    crc16_ibm,
    decode_fm_bits,
    decode_fm_bytes,
    estimate_cell_ticks,
    pll_decode_bits,
    pll_decode_fm_bytes,
    scan_fm_sectors,
)
from .scp import SCPImage, SCPHeader, TrackData, RevolutionEntry

__all__ = [
    "__version__",
    "SCPImage",
    "SCPHeader",
    "TrackData",
    "RevolutionEntry",
    "FMDecodeResult",
    "decode_fm_bytes",
    "decode_fm_bits",
    "estimate_cell_ticks",
    "bits_to_bytes",
    "best_aligned_bytes",
    "pll_decode_bits",
    "pll_decode_fm_bytes",
    "crc16_ibm",
    "scan_fm_sectors",
    "PLLDecodeResult",
    "SectorGuess",
]

__version__ = "0.0.1"
