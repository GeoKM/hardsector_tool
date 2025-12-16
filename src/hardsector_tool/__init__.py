"""
Top-level package for hard sector flux decoding utilities.

This package will collect parsers, analyzers, and CLI helpers for working with
hard-sectored floppy disk flux captures such as the ACMS80217 sample set.
"""

from .fm import (
    FMDecodeResult,
    best_aligned_bytes,
    bits_to_bytes,
    decode_fm_bits,
    decode_fm_bytes,
    estimate_cell_ticks,
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
]

__version__ = "0.0.1"
