from pathlib import Path

from hardsector_tool.hardsector import (
    FORMAT_PRESETS,
    best_sector_map,
    build_raw_image,
    group_hard_sectors,
)
from hardsector_tool.scp import RevolutionEntry, SCPImage, TrackData

FIXTURE = Path("tests/ACMS80217/ACMS80217-HS32.scp")
FIXTURE_221 = Path("tests/ACMS80221/ACMS80221-HS32.scp")


def test_grouping_matches_expected_rotations() -> None:
    image = SCPImage.from_file(FIXTURE)
    track = image.read_track(0)
    assert track is not None

    grouping = group_hard_sectors(track, sectors_per_rotation=32)
    # 33 holes per rotation -> 198 rev entries yields 6 rotations.
    assert grouping.rotations == 6
    assert len(grouping.groups[0]) == 32
    assert grouping.groups[0][0].hole_index == 0
    assert sum(len(h.revolution_indices) for h in grouping.groups[0]) == 33


def test_detects_short_pair_position() -> None:
    image = SCPImage.from_file(FIXTURE)
    track = image.read_track(0)
    assert track is not None

    grouping = group_hard_sectors(track, sectors_per_rotation=32)
    assert grouping.short_pair_positions is not None
    assert grouping.short_pair_positions[0] == 14
    assert len(grouping.groups[0]) == 32
    assert sum(len(h.revolution_indices) for h in grouping.groups[0]) == 33


def test_short_pair_position_other_fixture() -> None:
    image = SCPImage.from_file(FIXTURE_221)
    track = image.read_track(0)
    assert track is not None

    grouping = group_hard_sectors(track, sectors_per_rotation=32)
    assert grouping.short_pair_positions is not None
    assert grouping.short_pair_positions[0] == 28


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
