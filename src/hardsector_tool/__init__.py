"""
Top-level package for hard sector flux decoding utilities.

This package will collect parsers, analyzers, and CLI helpers for working with
hard-sectored floppy disk flux captures such as the ACMS80217 sample set.
"""

from .fm import (
    FMDecodeResult,
    MFMDecodeResult,
    PLLDecodeResult,
    SectorGuess,
    best_aligned_bytes,
    bits_to_bytes,
    crc16_ibm,
    decode_fm_bits,
    decode_fm_bytes,
    decode_mfm_bytes,
    fm_bytes_from_bitcells,
    brute_force_mark_payloads,
    estimate_cell_ticks,
    pll_decode_bits,
    pll_decode_fm_bytes,
    scan_data_marks,
    scan_bit_patterns,
    mfm_bytes_from_bitcells,
    scan_fm_sectors,
)
from .hardsector import (
    HardSectorGrouping,
    HoleCapture,
    best_sector_map,
    decode_hole,
    FORMAT_PRESETS,
    group_hard_sectors,
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
    "brute_force_mark_payloads",
    "crc16_ibm",
    "scan_fm_sectors",
    "PLLDecodeResult",
    "MFMDecodeResult",
    "decode_mfm_bytes",
    "fm_bytes_from_bitcells",
    "mfm_bytes_from_bitcells",
    "scan_data_marks",
    "scan_bit_patterns",
    "SectorGuess",
    "HardSectorGrouping",
    "HoleCapture",
    "group_hard_sectors",
    "decode_hole",
    "best_sector_map",
    "FORMAT_PRESETS",
]

__version__ = "0.0.1"
