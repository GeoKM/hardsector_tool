from pathlib import Path

from hardsector_tool.hardsector import FORMAT_PRESETS, best_sector_map, group_hard_sectors
from hardsector_tool.fm import scan_fm_sectors
from hardsector_tool.scp import SCPImage


FIXTURE = Path("tests/ACMS80217/ACMS80217-HS32.scp")


def test_grouping_matches_expected_rotations() -> None:
    image = SCPImage.from_file(FIXTURE)
    track = image.read_track(0)
    assert track is not None

    grouping = group_hard_sectors(track, sectors_per_rotation=32)
    # 33 holes per rotation -> 198 rev entries yields 6 rotations.
    assert grouping.rotations == 6
    assert len(grouping.groups[0]) == 33
    assert grouping.groups[0][0].hole_index == 0
    assert grouping.groups[0][0].revolution_index == 0


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
