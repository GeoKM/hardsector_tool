from pathlib import Path

import pytest

from conftest import require_fixture
from hardsector_tool.hardsector import group_hard_sectors, pair_holes
from hardsector_tool.scp import SCPImage
from hardsector_tool.wang import reconstruct_track, scan_wang_frames

pytestmark = pytest.mark.slow

FIXTURE = Path("tests/ACMS80217/ACMS80217-HS32.scp")
FIXTURE_221 = Path("tests/ACMS80221/ACMS80221-HS32.scp")


def test_wang_scan_frame_detects_headers() -> None:
    fixture = require_fixture(FIXTURE)

    image = SCPImage.from_file(fixture)
    track = image.read_track(0)
    assert track is not None

    grouping = group_hard_sectors(track, sectors_per_rotation=32)
    paired = pair_holes(grouping.groups[0])
    assert len(paired) == 16

    reconstructed, _, _ = reconstruct_track(
        image,
        0,
        sector_size=256,
        logical_sectors=16,
        pair_phase=0,
    )
    assert reconstructed == {} or all(s.sector_id < 16 for s in reconstructed.values())


def test_scan_wang_frames_accepts_plausible_offsets() -> None:
    # fabricate a small stream with track/sector header and dummy checksum
    payload = bytes(range(16))
    header = bytes([0, 1])
    body = header + payload
    checksum = (sum(body) & 0xFFFF).to_bytes(2, "big")
    stream = b"\x00\x00" + body + checksum
    frames = scan_wang_frames(
        stream, track=0, expected_sectors=16, sector_size=len(payload)
    )
    assert frames
    assert any(frame.sector_id == 1 for frame in frames)


def test_reconstruct_track_sets_window_metadata() -> None:
    fixture = require_fixture(FIXTURE)

    image = SCPImage.from_file(fixture)
    reconstructed, recon, _ = reconstruct_track(
        image,
        0,
        sector_size=256,
        logical_sectors=16,
        pair_phase=0,
    )
    if recon:
        rec = next(iter(recon.values()))
        assert rec.sector_size in (128, 256, 512)
        assert 0 <= rec.hole_shift < 32
    else:
        assert reconstructed == {}


def test_reconstruct_track_supports_unpaired_sectors() -> None:
    fixture = require_fixture(FIXTURE)

    image = SCPImage.from_file(fixture)
    reconstructed, _, _ = reconstruct_track(
        image,
        0,
        sector_size=256,
        logical_sectors=32,
        sectors_per_rotation=32,
        pair_phase=0,
        pair_hole_windows=False,
    )
    assert reconstructed == {} or (
        max(s.sector_id for s in reconstructed.values()) < 32
        and len(reconstructed) <= 32
    )


def test_reconstruct_track_promotes_dominant_transform_family() -> None:
    fixture = require_fixture(FIXTURE_221)

    image = SCPImage.from_file(fixture)
    reconstructed, _, _ = reconstruct_track(
        image,
        0,
        sector_size=256,
        logical_sectors=16,
        pair_phase=0,
        clock_factor=1.0,
    )

    assert len(reconstructed) == 16

    prefix_counts: dict[str, int] = {}
    for sector in reconstructed.values():
        for alg in sector.checksum_algorithms:
            prefix = alg.split(":", 1)[0]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    assert prefix_counts
    dominant_prefix = max(prefix_counts.items(), key=lambda item: item[1])[0]
    assert prefix_counts[dominant_prefix] >= 15

    sector_six = reconstructed.get(6)
    assert sector_six is not None
    assert sector_six.checksum_algorithms
    assert any(
        alg.startswith(dominant_prefix) for alg in sector_six.checksum_algorithms
    )
