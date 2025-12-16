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
    brute_force_mark_payloads,
    crc16_ibm,
    decode_fm_bits,
    decode_fm_bytes,
    decode_mfm_bytes,
    estimate_cell_ticks,
    fm_bytes_from_bitcells,
    mfm_bytes_from_bitcells,
    pll_decode_bits,
    pll_decode_fm_bytes,
    scan_bit_patterns,
    scan_data_marks,
    scan_fm_sectors,
)
from .hardsector import (
    FORMAT_PRESETS,
    HardSectorGrouping,
    HoleCapture,
    best_sector_map,
    compute_flux_index_deltas,
    compute_flux_index_diagnostics,
    decode_hole,
    group_hard_sectors,
    pair_holes,
    stitch_rotation_flux,
)
from .scp import RevolutionEntry, SCPHeader, SCPImage, TrackData
from .wang import reconstruct_track, summarize_wang_map

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
    "pair_holes",
    "group_hard_sectors",
    "decode_hole",
    "best_sector_map",
    "FORMAT_PRESETS",
    "compute_flux_index_diagnostics",
    "compute_flux_index_deltas",
    "stitch_rotation_flux",
    "reconstruct_track",
    "summarize_wang_map",
]

__version__ = "0.0.1"
