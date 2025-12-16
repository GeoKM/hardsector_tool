from pathlib import Path

from hardsector_tool.hardsector import group_hard_sectors
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
