from pathlib import Path

from pathlib import Path

import pytest

from conftest import require_fixture
from hardsector_tool.fm import pll_decode_fm_bytes
from hardsector_tool.hardsector import payload_metrics
from hardsector_tool.scp import SCPImage


pytestmark = pytest.mark.slow


FIXTURES = [
    Path("tests/ACMS80217/ACMS80217-HS32.scp"),
    Path("tests/ACMS80221/ACMS80221-HS32.scp"),
]


@pytest.mark.parametrize(
    "scp_path",
    FIXTURES,
    ids=[path.parent.name for path in FIXTURES],
)
def test_clock_factor_one_increases_entropy(scp_path: Path) -> None:
    scp_path = require_fixture(scp_path)

    image = SCPImage.from_file(scp_path)
    track = image.read_track(0)
    assert track is not None

    flux = track.decode_flux(0)
    index_ticks = track.revolutions[0].index_ticks

    default = pll_decode_fm_bytes(
        flux,
        sample_freq_hz=image.sample_freq_hz,
        index_ticks=index_ticks,
    )
    tuned = pll_decode_fm_bytes(
        flux,
        sample_freq_hz=image.sample_freq_hz,
        index_ticks=index_ticks,
        clock_factor=1.0,
    )

    window_size = min(512, len(tuned.bytes_out))
    tuned_window = tuned.bytes_out[:window_size]
    default_window = default.bytes_out[:window_size]

    fill_default, entropy_default = payload_metrics(default_window)
    fill_tuned, entropy_tuned = payload_metrics(tuned_window)

    assert entropy_tuned > entropy_default or fill_tuned < fill_default
