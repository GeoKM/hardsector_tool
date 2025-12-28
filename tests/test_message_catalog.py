from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hardsector_tool import scanmeta


def _build_message_reconstruct(tmp_path: Path) -> Path:
    out_dir = tmp_path / "reconstruct"
    sectors_dir = out_dir / "sectors"
    tracks_dir = out_dir / "tracks"
    sectors_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tracks": [],
        "totals": {
            "expected_sectors": 32,
            "written_sectors": 32,
            "missing_sectors": 0,
        },
    }

    track = 0
    manifest["tracks"].append(
        {"track_number": track, "recovered_sectors": 16, "sector_size": 256}
    )
    tracks_dir.joinpath(f"T{track:02d}.json").write_text(
        json.dumps({"sector_size": 256})
    )

    message_sector = bytearray(b"\x00" * 256)
    strings = {
        10: b"Mount Archive Disk",
        40: b"Delete from Archive Diskette",
        90: b"Initialize Archive Diskette",
        150: b"(Rev. J-30 - 06/06/77)",
    }
    for offset, text in strings.items():
        message_sector[offset : offset + len(text)] = text
    sectors_dir.joinpath("T00_S00.bin").write_bytes(message_sector)

    filler = os.urandom(256)
    sectors_dir.joinpath("T00_S01.bin").write_bytes(filler)

    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    (out_dir / "report.txt").write_text("dummy report")
    return out_dir


def test_message_catalog_detection(tmp_path: Path) -> None:
    out_dir = _build_message_reconstruct(tmp_path)

    result = scanmeta.scan_metadata(out_dir)
    catalog = result.get("message_catalog") or {}

    assert catalog.get("candidates")
    first = catalog["candidates"][0]
    assert first["track"] == 0
    assert any("Archive" in ex["text"] for ex in first["examples"])

    revisions = catalog.get("revision_markers", [])
    assert any("Rev." in entry["text"] for entry in revisions)


def test_message_catalog_cli_tsv(tmp_path: Path) -> None:
    out_dir = _build_message_reconstruct(tmp_path)
    output_json = tmp_path / "scan.json"
    messages_out = tmp_path / "messages.tsv"

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
        "--messages-out",
        str(messages_out),
    ]
    result = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    assert messages_out.exists()
    lines = messages_out.read_text().splitlines()
    assert any("Mount Archive Disk" in line for line in lines)
