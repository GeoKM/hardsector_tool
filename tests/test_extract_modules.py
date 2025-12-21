from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _build_fake_reconstruct(tmp_path: Path) -> Path:
    out_dir = tmp_path / "reconstruct"
    sectors_dir = out_dir / "sectors"
    tracks_dir = out_dir / "tracks"
    sectors_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tracks": [{"track_number": 0, "recovered_sectors": 16, "sector_size": 256}],
        "totals": {"expected_sectors": 16, "written_sectors": 3, "missing_sectors": 13},
    }

    # Create three sectors; sector 0 contains the descriptor.
    for sector in range(3):
        content = bytearray(b"\x00" * 256)
        if sector == 0:
            descriptor = (
                b"=SYSGEN.TEST////"
                + b"\x00\x00"
                + b"\x01\x00"
                + b"\x02\x00"
                + b"\x00\x00"
            )
            content[: len(descriptor)] = descriptor
        else:
            content[:] = bytes([sector]) * 256
        sectors_dir.joinpath(f"T00_S{sector:02d}.bin").write_bytes(content)

    tracks_dir.joinpath("T00.json").write_text(json.dumps({"sector_size": 256}))
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return out_dir


def test_extract_modules_cli(tmp_path: Path) -> None:
    out_dir = _build_fake_reconstruct(tmp_path)
    derived_dir = tmp_path / "derived"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    cmd = [
        sys.executable,
        "-m",
        "hardsector_tool",
        "extract-modules",
        str(out_dir),
        "--out",
        str(derived_dir),
        "--min-refs",
        "2",
    ]

    result = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    module_path = derived_dir / "SYSGEN.TEST.bin"
    sidecar_path = derived_dir / "SYSGEN.TEST.json"
    summary_path = derived_dir / "extraction_summary.json"

    assert module_path.exists()
    assert sidecar_path.exists()
    assert summary_path.exists()

    payload = module_path.read_bytes()
    assert len(payload) == 3 * 256
    assert payload.startswith(b"=SYSGEN.TEST////")
    assert payload[256:257] == b"\x01"
    assert payload[512:513] == b"\x02"

    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["module_name"] == "SYSGEN.TEST"
    assert sidecar["refs_linear"][:3] == [0, 1, 2]
    assert sidecar["hypothesis_used"] == "H1_le16"
