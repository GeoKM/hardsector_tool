from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


FIXTURES = [
    Path("tests/ACMS80217/ACMS80217-HS32.scp"),
    Path("tests/ACMS80221/ACMS80221-HS32.scp"),
]


@pytest.mark.parametrize("fixture_path", FIXTURES)
def test_reconstruct_disk_track0(tmp_path: Path, fixture_path: Path) -> None:
    out_dir = tmp_path / "reconstruct"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    cmd = [
        sys.executable,
        "-m",
        "hardsector_tool",
        "reconstruct-disk",
        str(fixture_path),
        "--out",
        str(out_dir),
        "--tracks",
        "0-0",
    ]
    result = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    sectors_dir = out_dir / "sectors"
    for sector_id in range(16):
        assert (sectors_dir / f"T00_S{sector_id:02d}.bin").exists()

    manifest_path = out_dir / "manifest.json"
    report_path = out_dir / "report.txt"
    assert manifest_path.exists()
    assert report_path.exists()

    manifest = json.loads(manifest_path.read_text())
    track_entries = {t["track_number"]: t for t in manifest.get("tracks", [])}
    assert track_entries[0]["recovered_sectors"] == 16


@pytest.mark.parametrize("fixture_path", FIXTURES)
def test_reconstruct_disk_track_mapping(tmp_path: Path, fixture_path: Path) -> None:
    out_dir = tmp_path / "reconstruct_mapping"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    cmd = [
        sys.executable,
        "-m",
        "hardsector_tool",
        "reconstruct-disk",
        str(fixture_path),
        "--out",
        str(out_dir),
        "--tracks",
        "0-2",
        "--track-step",
        "auto",
    ]
    result = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())

    assert manifest["mapping"]["track_step"] == 2
    track_entries = {t["logical_track"]: t for t in manifest.get("tracks", [])}
    expected_tracks = [0, 2, 4]
    for logical_track, scp_track in enumerate(expected_tracks):
        entry = track_entries[logical_track]
        assert entry["expected_scp_track_id"] == scp_track
        assert entry["scp_track_id"] == scp_track
        assert entry["recovered_sectors"] == 16

    sectors_dir = out_dir / "sectors"
    for track_number in range(3):
        for sector_id in range(16):
            assert (sectors_dir / f"T{track_number:02d}_S{sector_id:02d}.bin").exists()
