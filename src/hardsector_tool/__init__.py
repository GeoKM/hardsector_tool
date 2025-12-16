"""
Top-level package for hard sector flux decoding utilities.

This package will collect parsers, analyzers, and CLI helpers for working with
hard-sectored floppy disk flux captures such as the ACMS80217 sample set.
"""

from .scp import SCPImage, SCPHeader, TrackData, RevolutionEntry

__all__ = [
    "__version__",
    "SCPImage",
    "SCPHeader",
    "TrackData",
    "RevolutionEntry",
]

__version__ = "0.0.1"
