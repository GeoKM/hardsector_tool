from pathlib import Path

import pytest

from conftest import require_fixture
from hardsector_tool.scp import SCPImage
from hardsector_tool.wang import checksum_prefixes, dominant_prefix, reconstruct_track

pytestmark = pytest.mark.slow

FIXTURE = Path("tests/ACMS80221/ACMS80221-HS32.scp")

def test_dominant_prefix_rescue() -> None:
    fixture = require_fixture(FIXTURE)

    image = SCPImage.from_file(fixture)
    sector_map, _, _ = reconstruct_track(
        image,
        track_number=0,
        sector_size=256,
        logical_sectors=16,
        pair_phase=0,
    )

    assert len(sector_map) == 16

    dom = dominant_prefix(sector_map, min_count=10)
    assert dom is not None

    occurrences = sum(
        1 for sector in sector_map.values() if dom in checksum_prefixes(sector)
    )
    assert occurrences >= 15

    sector_six = sector_map.get(6)
    assert sector_six is not None
    assert sector_six.checksum_algorithms
    assert dom in checksum_prefixes(sector_six)
