from pathlib import Path

import pytest

from conftest import require_fixture
from hardsector_tool.hardsector import (
    FORMAT_PRESETS,
    best_sector_map,
    build_raw_image,
    group_hard_sectors,
    pair_holes,
)
from hardsector_tool.scp import RevolutionEntry, SCPImage, TrackData

pytestmark = pytest.mark.slow

FIXTURE = Path("tests/ACMS80217/ACMS80217-HS32.scp")
FIXTURE_221 = Path("tests/ACMS80221/ACMS80221-HS32.scp")


def test_grouping_matches_expected_rotations() -> None:
    fixture = require_fixture(FIXTURE)

    image = SCPImage.from_file(fixture)
    track = image.read_track(0)
    assert track is not None

    grouping = group_hard_sectors(track, sectors_per_rotation=32)
    # 33 holes per rotation -> 198 rev entries yields 6 rotations.
    assert grouping.rotations == 6
    assert grouping.pulses_per_rotation == 33
    assert len(grouping.groups[0]) == 32
    assert grouping.groups[0][0].hole_index == 0
    assert sum(len(h.revolution_indices) for h in grouping.groups[0]) == 33


def test_grouping_pairs_holes_into_logical_sectors() -> None:
    fixture = require_fixture(FIXTURE)

    image = SCPImage.from_file(fixture)
    track = image.read_track(0)
    assert track is not None

    grouping = group_hard_sectors(track, sectors_per_rotation=32)
    paired = pair_holes(grouping.groups[0])
    assert len(paired) == 16
    assert all(h.logical_sector_index == idx for idx, h in enumerate(paired))
    assert sum(len(h.revolution_indices) for h in paired) == 33


def test_detects_short_pair_position() -> None:
    fixture = require_fixture(FIXTURE)

    image = SCPImage.from_file(fixture)
    track = image.read_track(0)
    assert track is not None

    grouping = group_hard_sectors(track, sectors_per_rotation=32)
    assert grouping.short_pair_positions is not None
    assert grouping.short_pair_positions[0] == 0
    assert grouping.index_pair_position == 14
    assert len(grouping.groups[0]) == 32
    assert sum(len(h.revolution_indices) for h in grouping.groups[0]) == 33


def test_short_pair_position_other_fixture() -> None:
    fixture = require_fixture(FIXTURE_221)

    image = SCPImage.from_file(fixture)
    track = image.read_track(0)
    assert track is not None

    grouping = group_hard_sectors(track, sectors_per_rotation=32)
    assert grouping.short_pair_positions is not None
    assert grouping.short_pair_positions[0] == 0
    assert grouping.index_pair_position == 28


def test_best_sector_map_prefers_crc_ok() -> None:
    # Fabricate two sector guesses, one with valid CRC.
    g_ok = type(
        "G",
        (),
        {"track": 0, "head": 0, "sector_id": 1, "length": 256, "crc_ok": True},
    )
    g_bad = type(
        "G",
        (),
        {"track": 0, "head": 0, "sector_id": 1, "length": 256, "crc_ok": False},
    )
    best = best_sector_map([[g_bad], [g_ok]], expected_track=0, expected_head=0)
    assert best[1] is g_ok


def test_best_sector_map_prefers_better_score_on_ties() -> None:
    g_low = type(
        "G",
        (),
        {
            "track": 0,
            "head": 0,
            "sector_id": 2,
            "length": 256,
            "crc_ok": False,
            "decode_score": 1.0,
        },
    )
    g_high = type(
        "G",
        (),
        {
            "track": 0,
            "head": 0,
            "sector_id": 2,
            "length": 256,
            "crc_ok": False,
            "decode_score": 5.0,
        },
    )
    best = best_sector_map([[g_low], [g_high]], expected_track=0, expected_head=0)
    assert best[2] is g_high


def test_presets_include_cpm_defaults() -> None:
    preset = FORMAT_PRESETS["cpm-16x256"]
    assert preset["expected_sectors"] == 16
    assert preset["sector_size"] == 256
    assert preset["encoding"] == "fm"


def test_build_raw_image_fills_missing() -> None:
    fake_guess = type(
        "G",
        (),
        {
            "data": b"A" * 4,
            "track": 0,
            "head": 0,
            "sector_id": 0,
            "length": 4,
            "crc_ok": True,
        },
    )
    track_maps = {0: {0: fake_guess}}
    raw = build_raw_image(
        track_maps, track_order=[0], expected_sectors=2, expected_size=4, fill_byte=0x45
    )
    assert raw == b"AAAA" + bytes([0x45]) * 4


def test_merges_shortest_adjacent_intervals() -> None:
    # Two short entries (1,2) should be merged into a single hole.
    revs = [
        RevolutionEntry(index_ticks=1000, flux_count=10, data_offset=0),
        RevolutionEntry(index_ticks=100, flux_count=5, data_offset=0),
        RevolutionEntry(index_ticks=120, flux_count=6, data_offset=0),
        RevolutionEntry(index_ticks=900, flux_count=9, data_offset=0),
    ]
    track = TrackData(
        track_number=0, revolutions=revs, flux_data=b"", flux_data_offset=0
    )
    grouping = group_hard_sectors(track, sectors_per_rotation=3, index_aligned=False)
    assert grouping.groups
    merged = grouping.groups[0]
    assert len(merged) == 3
    assert any(len(h.revolution_indices) == 2 for h in merged)


def test_normalization_yields_single_index_pair_per_rotation() -> None:
    fixture = require_fixture(FIXTURE)

    image = SCPImage.from_file(fixture)
    track = image.read_track(0)
    assert track is not None

    grouping = group_hard_sectors(track, sectors_per_rotation=32)
    assert grouping.pulses_per_rotation == 33
    assert grouping.rotations == track.revolution_count // grouping.pulses_per_rotation
    assert all(len(rot) == 32 for rot in grouping.groups)
    assert all(pos == 0 for pos in grouping.short_pair_positions or [])
    for rotation in grouping.groups:
        assert sum(len(h.revolution_indices) for h in rotation) == 33
        merged = sum(1 for h in rotation if len(h.revolution_indices) > 1)
        assert merged == 1


def test_revolution_count_represents_pulses_not_physical_revs() -> None:
    revs = [
        RevolutionEntry(index_ticks=500, flux_count=5, data_offset=0) for _ in range(10)
    ]
    track = TrackData(
        track_number=0,
        revolutions=revs,
        flux_data=b"",
        flux_data_offset=0,
    )
    grouping = group_hard_sectors(
        track, sectors_per_rotation=2, index_aligned=False, hs_normalize=False
    )
    # 3 pulses per rotation -> only 3 full rotations from 10 entries.
    assert grouping.rotations == 3
