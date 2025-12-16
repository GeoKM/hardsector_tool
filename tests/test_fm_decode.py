from pathlib import Path

from hardsector_tool.fm import decode_fm_bytes, estimate_cell_ticks
from hardsector_tool.scp import SCPImage


FIXTURE = Path("tests/ACMS80217/ACMS80217-HS32.scp")


def test_fm_decode_produces_bytes() -> None:
    scp = SCPImage.from_file(FIXTURE)
    track = scp.read_track(0)
    assert track is not None
    flux = track.decode_flux(0)

    half, full, threshold = estimate_cell_ticks(flux)
    # Expect a clear separation between half and full cells.
    assert half < full
    assert threshold > 0

    result = decode_fm_bytes(flux)
    # Heuristic decode should produce a non-trivial byte stream.
    assert len(result.bytes_out) > 100
    assert result.bytes_out.count(0xFF) >= 1
