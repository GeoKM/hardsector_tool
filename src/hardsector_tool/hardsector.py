"""
Helpers for hard-sectored flux captures (e.g., 32 holes + index).

Greaseweazle's `--hardsector --raw` mode records one revolution per hole
tick, so an 8" disk with 32 sector holes plus one index hole yields
33 entries per full rotation. We group those to make per-rotation
analysis easier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .fm import (
    SectorGuess,
    decode_fm_bytes,
    pll_decode_fm_bytes,
    scan_fm_sectors,
)
from .scp import SCPImage, TrackData


@dataclass(frozen=True)
class HoleCapture:
    hole_index: int
    revolution_index: int
    index_ticks: int
    flux_count: int


@dataclass(frozen=True)
class HardSectorGrouping:
    groups: List[List[HoleCapture]]
    sectors_per_rotation: int
    index_holes_per_rotation: int = 1

    @property
    def rotations(self) -> int:
        return len(self.groups)


def group_hard_sectors(track: TrackData, sectors_per_rotation: int = 32) -> HardSectorGrouping:
    """
    Chunk revolutions into rotations based on the expected hole count.
    """
    holes_per_rotation = sectors_per_rotation + 1
    revs = track.revolutions
    groups: List[List[HoleCapture]] = []
    for base in range(0, len(revs), holes_per_rotation):
        chunk = revs[base : base + holes_per_rotation]
        if len(chunk) < holes_per_rotation:
            break
        captures = [
            HoleCapture(
                hole_index=i,
                revolution_index=base + i,
                index_ticks=rev.index_ticks,
                flux_count=rev.flux_count,
            )
            for i, rev in enumerate(chunk)
        ]
        groups.append(captures)
    return HardSectorGrouping(groups=groups, sectors_per_rotation=sectors_per_rotation)


def decode_hole(
    image: SCPImage,
    track: TrackData,
    hole_capture: HoleCapture,
    use_pll: bool = False,
) -> Optional[SectorGuess]:
    """
    Decode one hole's worth of flux into bytes using FM heuristics or PLL.
    """
    flux = track.decode_flux(hole_capture.revolution_index)
    if use_pll:
        decoded = pll_decode_fm_bytes(
            flux,
            sample_freq_hz=image.sample_freq_hz,
            index_ticks=hole_capture.index_ticks,
        )
    else:
        decoded = decode_fm_bytes(flux)

    sectors = scan_fm_sectors(decoded.bytes_out)
    return sectors[0] if sectors else None


def assemble_rotation(
    image: SCPImage,
    track: TrackData,
    grouping: HardSectorGrouping,
    rotation_index: int,
    use_pll: bool = False,
) -> List[SectorGuess]:
    """
    Decode all holes in a given rotation and return any sector guesses found.
    """
    if rotation_index >= grouping.rotations:
        return []
    guesses: List[SectorGuess] = []
    for hole in grouping.groups[rotation_index]:
        guess = decode_hole(image, track, hole, use_pll=use_pll)
        if guess:
            guesses.append(guess)
    return guesses
