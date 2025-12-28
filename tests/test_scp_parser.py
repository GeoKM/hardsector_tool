from pathlib import Path

import pytest

from conftest import require_fixture
from hardsector_tool.scp import SCPImage

pytestmark = pytest.mark.slow

FIXTURE = require_fixture(Path("tests/ACMS80217/ACMS80217-HS32.scp"))


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
    assert header.cell_width_code == 0
    assert header.capture_resolution == 0
    assert header.heads == 1


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


def test_track_offsets_and_flux_words(scp_image: SCPImage) -> None:
    track = scp_image.read_track(0)
    assert track is not None
    assert len(track.revolutions) == 198
    assert track.revolutions[0].data_offset == 2380
    # First few decoded flux words should be non-zero and below 16-bit carry limits
    flux_words = track.decode_flux(0)[:8]
    assert all(0 < word < 65535 for word in flux_words)
