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
        elif sector == 1:
            content[:] = bytes([sector]) * 256
            content[16 : 16 + 13] = b"pppp=DOS.MENU"
        else:
            content[:] = bytes([sector]) * 256
        sectors_dir.joinpath(f"T00_S{sector:02d}.bin").write_bytes(content)

    tracks_dir.joinpath("T00.json").write_text(json.dumps({"sector_size": 256}))
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return out_dir


def _run_extract_cli(out_dir: Path, derived_dir: Path, only_prefix: str) -> dict:
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
        "--only-prefix",
        only_prefix,
    ]

    result = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    summary_path = derived_dir / "extraction_summary.json"
    assert summary_path.exists()
    return json.loads(summary_path.read_text())


def test_extract_modules_cli(tmp_path: Path) -> None:
    out_dir = _build_fake_reconstruct(tmp_path)
    derived_dir = tmp_path / "derived"
    summary = _run_extract_cli(out_dir, derived_dir, "SYSGEN.")

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
    assert sidecar["record_type"] == "descriptor"

    totals = summary["totals"]
    assert totals["descriptor_records_found"] == 1
    assert totals["descriptor_records_parsed"] == 1
    assert totals["descriptor_records_skipped"] == 0
    assert totals["name_list_hits_found"] == 1
    assert totals["unique_name_list_modules"] == 1

    assert summary["by_hypothesis"]["H1_le16"] == 1
    assert summary["by_hypothesis"]["unparsed_descriptor"] == 0

    name_list_modules = {entry["module_name"]: entry for entry in summary["name_list_modules"]}
    assert name_list_modules["DOS.MENU"]["count"] == 1
    assert name_list_modules["DOS.MENU"]["sample_locations"][0]["sector"] == 1


def test_only_prefix_normalization(tmp_path: Path) -> None:
    out_dir = _build_fake_reconstruct(tmp_path)

    summary_plain = _run_extract_cli(out_dir, tmp_path / "derived_plain", "SYSGEN.")
    summary_with_equals = _run_extract_cli(
        out_dir, tmp_path / "derived_equals", "=SYSGEN."
    )

    modules_plain = {
        mod["module_name"] for mod in summary_plain["modules"] if mod["hypothesis_used"]
    }
    modules_with_equals = {
        mod["module_name"]
        for mod in summary_with_equals["modules"]
        if mod["hypothesis_used"]
    }

    assert modules_plain == modules_with_equals == {"SYSGEN.TEST"}

    totals = summary_with_equals["totals"]
    assert totals["descriptor_records_found"] == 1
    assert totals["descriptor_records_parsed"] == 1
    assert totals["extracted_modules"] == 1
    assert totals["missing_ref_modules"] == 0
    assert totals["bytes_written_total"] > 0

    by_hypothesis = summary_with_equals["by_hypothesis"]
    assert by_hypothesis["H1_le16"] == 1
    assert by_hypothesis["unparsed_descriptor"] == 0
