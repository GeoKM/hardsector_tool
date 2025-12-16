"""
Helpers for hard-sectored flux captures (e.g., 32 holes + index).

Greaseweazle's `--hardsector --raw` mode records one revolution per hole
tick, so an 8" disk with 32 sector holes plus one index hole yields
33 entries per full rotation. We group those to make per-rotation
analysis easier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .fm import (
    DEFAULT_CLOCK_ADJ,
    SectorGuess,
    decode_fm_bytes,
    decode_mfm_bytes,
    estimate_cell_ticks,
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


FORMAT_PRESETS = {
    "cpm-16x256": {"expected_sectors": 16, "sector_size": 256, "encoding": "fm"},
    "cpm-26x128": {"expected_sectors": 26, "sector_size": 128, "encoding": "fm"},
    "ibm-9x512-mfm": {"expected_sectors": 9, "sector_size": 512, "encoding": "mfm"},
}


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
    require_sync: bool = False,
    encoding: str = "fm",
    initial_clock_ticks: float | None = None,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
) -> Optional[SectorGuess]:
    """
    Decode one hole's worth of flux into bytes using FM heuristics or PLL.
    """
    flux = track.decode_flux(hole_capture.revolution_index)
    if encoding.lower() == "mfm":
        decoded = decode_mfm_bytes(
            flux,
            sample_freq_hz=image.sample_freq_hz,
            index_ticks=hole_capture.index_ticks,
            initial_clock_ticks=initial_clock_ticks,
            clock_adjust=clock_adjust,
        )
    elif use_pll:
        decoded = pll_decode_fm_bytes(
            flux,
            sample_freq_hz=image.sample_freq_hz,
            index_ticks=hole_capture.index_ticks,
            initial_clock_ticks=initial_clock_ticks,
            clock_adjust=clock_adjust,
        )
    else:
        decoded = decode_fm_bytes(flux)

    sync_bytes = (0xA1, 0x4E) if encoding.lower() == "mfm" else (0xA1,)
    sectors = scan_fm_sectors(
        decoded.bytes_out, require_sync=require_sync, sync_bytes=sync_bytes
    )
    return sectors[0] if sectors else None


def decode_hole_bytes(
    image: SCPImage,
    track: TrackData,
    hole_capture: HoleCapture,
    use_pll: bool = False,
    encoding: str = "fm",
    initial_clock_ticks: float | None = None,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
) -> bytes:
    """
    Decode one hole's flux and return raw decoded bytes without sector framing.
    """
    flux = track.decode_flux(hole_capture.revolution_index)
    if encoding.lower() == "mfm":
        decoded = decode_mfm_bytes(
            flux,
            sample_freq_hz=image.sample_freq_hz,
            index_ticks=hole_capture.index_ticks,
            initial_clock_ticks=initial_clock_ticks,
            clock_adjust=clock_adjust,
        )
        return decoded.bytes_out
    if use_pll:
        decoded = pll_decode_fm_bytes(
            flux,
            sample_freq_hz=image.sample_freq_hz,
            index_ticks=hole_capture.index_ticks,
            initial_clock_ticks=initial_clock_ticks,
            clock_adjust=clock_adjust,
        )
        return decoded.bytes_out
    return decode_fm_bytes(flux).bytes_out


def assemble_rotation(
    image: SCPImage,
    track: TrackData,
    grouping: HardSectorGrouping,
    rotation_index: int,
    use_pll: bool = False,
    require_sync: bool = False,
    encoding: str = "fm",
    calibrate_rotation: bool = False,
    synthetic_from_hole: bool = False,
    expected_sectors: int = 16,
    expected_size: int = 256,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
) -> List[SectorGuess]:
    """
    Decode all holes in a given rotation and return any sector guesses found.
    """
    if rotation_index >= grouping.rotations:
        return []
    guesses: List[SectorGuess] = []
    initial_clock = None
    if calibrate_rotation and grouping.groups[rotation_index]:
        first_flux = track.decode_flux(grouping.groups[rotation_index][0].revolution_index)
        half, _, _ = estimate_cell_ticks(first_flux)
        initial_clock = half * 2

    for hole in grouping.groups[rotation_index]:
        guess = decode_hole(
            image,
            track,
            hole,
            use_pll=use_pll,
            require_sync=require_sync,
            encoding=encoding,
            initial_clock_ticks=initial_clock,
            clock_adjust=clock_adjust,
        )
        if guess and guess.length:
            guesses.append(guess)
            continue
        if synthetic_from_hole:
            raw = decode_hole_bytes(
                image,
                track,
                hole,
                use_pll=use_pll,
                encoding=encoding,
                initial_clock_ticks=initial_clock,
                clock_adjust=clock_adjust,
            )
            payload = raw[:expected_size]
            guesses.append(
                SectorGuess(
                    offset=0,
                    track=track.track_number,
                    head=0,
                    sector_id=hole.hole_index % expected_sectors,
                    size_code=0,
                    length=len(payload),
                    crc_ok=False,
                    id_crc_ok=False,
                    data_crc_ok=False,
                    data=payload,
                )
            )
    return guesses


def decode_track_best_map(
    image: SCPImage,
    track_number: int,
    sectors_per_rotation: int = 32,
    expected_sectors: int = 16,
    expected_size: int = 256,
    encoding: str = "fm",
    use_pll: bool = False,
    require_sync: bool = False,
    calibrate_rotation: bool = False,
    synthetic_from_holes: bool = False,
    clock_adjust: float = DEFAULT_CLOCK_ADJ,
) -> Dict[int, SectorGuess]:
    track = image.read_track(track_number)
    if not track:
        return {}
    grouping = group_hard_sectors(track, sectors_per_rotation=sectors_per_rotation)
    all_rotations = [
        assemble_rotation(
            image,
            track,
            grouping,
            r,
            use_pll=use_pll,
            require_sync=require_sync,
            encoding=encoding,
            calibrate_rotation=calibrate_rotation,
            synthetic_from_hole=synthetic_from_holes,
            expected_sectors=expected_sectors,
            expected_size=expected_size,
            clock_adjust=clock_adjust,
        )
        for r in range(grouping.rotations)
    ]
    return best_sector_map(
        all_rotations,
        expected_track=track_number,
        expected_head=0,
        expected_sector_count=expected_sectors,
        expected_size=expected_size,
    )


def build_raw_image(
    track_maps: Dict[int, Dict[int, SectorGuess]],
    track_order: List[int],
    expected_sectors: int,
    expected_size: int,
    fill_byte: int = 0x00,
) -> bytes:
    """
    Assemble a flat image from per-track sector maps.

    Missing sectors are filled with the given fill_byte.
    """
    fill_block = bytes([fill_byte]) * expected_size
    parts: List[bytes] = []
    for track_num in track_order:
        sector_map = track_maps.get(track_num, {})
        for sid in range(expected_sectors):
            guess = sector_map.get(sid)
            if guess and guess.data:
                parts.append(guess.data)
            else:
                parts.append(fill_block)
    return b"".join(parts)


def best_sector_map(
    rotations: List[List[SectorGuess]],
    expected_track: int,
    expected_head: int = 0,
    expected_sector_count: int = 16,
    expected_size: int = 256,
) -> Dict[int, SectorGuess]:
    """
    Pick the best copy for each sector id across rotations.
    Preference order: valid CRCs, then first occurrence.
    """
    best: Dict[int, SectorGuess] = {}
    for rotation in rotations:
        for guess in rotation:
            if guess.track != expected_track or guess.head != expected_head:
                continue
            if guess.length != expected_size:
                continue
            existing = best.get(guess.sector_id)
            if not existing or (not existing.crc_ok and guess.crc_ok):
                best[guess.sector_id] = guess
    return {sid: g for sid, g in best.items() if sid < expected_sector_count}
