from pathlib import Path

import pytest

from hardsector_tool.scp import SCPImage


FIXTURE = Path("tests/ACMS80217/ACMS80217-HS32.scp")


@pytest.fixture(scope="session")
def scp_image() -> SCPImage:
    return SCPImage.from_file(FIXTURE)


def test_header_matches_fixture(scp_image: SCPImage) -> None:
    header = scp_image.header
    assert header.start_track == 0
    assert header.end_track == 152  # last even track holds data
    assert header.revolutions == 198
    assert len(header.non_empty_tracks) == 77
    assert header.track_offsets[0] == 688
    assert header.track_offsets[2] > header.track_offsets[0]


def test_track_zero_flux_round_trip(scp_image: SCPImage) -> None:
    track = scp_image.read_track(0)
    assert track is not None
    assert track.revolution_count == scp_image.header.revolutions

    rev0 = track.revolutions[0]
    flux = track.decode_flux(0)
    assert len(flux) == rev0.flux_count

    total_ticks = sum(flux)
    tolerance = int(rev0.index_ticks * 0.1)
    assert abs(total_ticks - rev0.index_ticks) <= tolerance
