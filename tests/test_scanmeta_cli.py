from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _build_fake_reconstruct(tmp_path: Path) -> Path:
    out_dir = tmp_path / "reconstruct"
    sectors_dir = out_dir / "sectors"
    tracks_dir = out_dir / "tracks"
    sectors_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tracks": [],
        "totals": {"expected_sectors": 48, "written_sectors": 48, "missing_sectors": 0},
    }

    for track in range(3):
        manifest["tracks"].append(
            {
                "track_number": track,
                "recovered_sectors": 16,
                "sector_size": 256,
            }
        )
        tracks_dir.joinpath(f"T{track:02d}.json").write_text(json.dumps({"sector_size": 256}))
        for sector in range(16):
            content = bytearray(b"\x00" * 256)
            if track == 0 and sector == 0:
                token = b"Office Information System"
                content[: len(token)] = token
                content[40:48] = b"FILE.TXT"
            elif track == 1 and sector == 1:
                content[10:20] = b"CONTROL."
                content[64:72] = b"pppp="
            sectors_dir.joinpath(f"T{track:02d}_S{sector:02d}.bin").write_bytes(content)

    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    (out_dir / "report.txt").write_text("dummy report")
    return out_dir


def test_scan_metadata_cli(tmp_path: Path) -> None:
    out_dir = _build_fake_reconstruct(tmp_path)
    output_json = tmp_path / "scan.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    cmd = [
        sys.executable,
        "-m",
        "hardsector_tool",
        "scan-metadata",
        str(out_dir),
        "--out",
        str(output_json),
    ]
    result = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    assert output_json.exists()
    data = json.loads(output_json.read_text())
    for key in ["summary", "signature_hits", "candidate_tables", "pointer_stats"]:
        assert key in data
