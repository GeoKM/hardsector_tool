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
        "totals": {"expected_sectors": 16, "written_sectors": 4, "missing_sectors": 12},
    }

    # Create four sectors; sectors 0 and 3 contain descriptors.
    for sector in range(4):
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
            entry = b"pppp=SYSGEN.TEST"
            start_offset = 16
            content[start_offset : start_offset + len(entry)] = entry
        elif sector == 2:
            content[:] = bytes([sector]) * 256
        else:
            descriptor2 = (
                b"=SYSGEN.NOPE////"
                + b"\x00\x00"
                + b"\x01\x00"
                + b"\x02\x00"
                + b"\x00\x00"
            )
            content[: len(descriptor2)] = descriptor2
        sectors_dir.joinpath(f"T00_S{sector:02d}.bin").write_bytes(content)

    tracks_dir.joinpath("T00.json").write_text(json.dumps({"sector_size": 256}))
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return out_dir


def _build_pppp_descriptor_fixture(tmp_path: Path) -> Path:
    out_dir = tmp_path / "reconstruct_pppp"
    sectors_dir = out_dir / "sectors"
    tracks_dir = out_dir / "tracks"
    sectors_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tracks": [{"track_number": 0, "recovered_sectors": 16, "sector_size": 256}],
        "totals": {"expected_sectors": 16, "written_sectors": 5, "missing_sectors": 11},
    }

    pointer_list = b"\x00\x00\x01\x00\x02\x00"
    pppp_entry = b"pppp=CONTROL.AT////" + pointer_list + b"pppp=SECOND.NAME"

    for sector in range(5):
        content = bytearray(bytes([sector]) * 256)
        if sector == 0:
            descriptor = b"=SYSGEN.TEST////" + pointer_list + b"\x00\x00"
            content[: len(descriptor)] = descriptor
        elif sector == 1:
            content[: len(pppp_entry)] = pppp_entry
        sectors_dir.joinpath(f"T00_S{sector:02d}.bin").write_bytes(content)

    tracks_dir.joinpath("T00.json").write_text(json.dumps({"sector_size": 256}))
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return out_dir


def _build_pppp_span_fixture(tmp_path: Path) -> Path:
    out_dir = tmp_path / "reconstruct_pppp_span"
    sectors_dir = out_dir / "sectors"
    tracks_dir = out_dir / "tracks"
    sectors_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tracks": [{"track_number": 0, "recovered_sectors": 16, "sector_size": 256}],
        "totals": {"expected_sectors": 16, "written_sectors": 6, "missing_sectors": 10},
    }

    for sector in range(6):
        content = bytearray(bytes([sector]) * 256)
        if sector == 0:
            descriptor = b"=SYSGEN.TEST////" + b"\x00\x00\x01\x00\x02\x00" + b"\x00\x00"
            content[: len(descriptor)] = descriptor
        elif sector == 2:
            entry = b"pppp=CONTROL.QUEUE////"
            start_offset = 256 - len(entry)
            content[start_offset : start_offset + len(entry)] = entry
        elif sector == 3:
            pointer_bytes = b"\x00\x00\x01\x00\x04\x00"
            next_marker = b"pppp=TAIL.END"
            content[: len(pointer_bytes)] = pointer_bytes
            content[len(pointer_bytes) : len(pointer_bytes) + len(next_marker)] = (
                next_marker
            )
        sectors_dir.joinpath(f"T00_S{sector:02d}.bin").write_bytes(content)

    tracks_dir.joinpath("T00.json").write_text(json.dumps({"sector_size": 256}))
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return out_dir


def _run_extract_cli(
    out_dir: Path,
    derived_dir: Path,
    only_prefix: str,
    extra_args: list[str] | None = None,
) -> dict:
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
    if extra_args:
        cmd.extend(extra_args)

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
    module_path_nope = derived_dir / "SYSGEN.NOPE.bin"
    sidecar_path_nope = derived_dir / "SYSGEN.NOPE.json"
    summary_path = derived_dir / "extraction_summary.json"

    assert module_path.exists()
    assert sidecar_path.exists()
    assert module_path_nope.exists()
    assert sidecar_path_nope.exists()
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

    sidecar_nope = json.loads(sidecar_path_nope.read_text())
    assert sidecar_nope["module_name"] == "SYSGEN.NOPE"
    assert sidecar_nope["hypothesis_used"] == "H1_le16"

    totals = summary["totals"]
    assert totals["descriptor_records_found"] == 2
    assert totals["descriptor_records_parsed"] == 2
    assert totals["descriptor_records_filtered_by_pppp"] == 0
    assert totals["descriptor_records_skipped"] == 0
    assert totals["name_list_hits_found"] == 1
    assert totals["unique_name_list_modules"] == 1
    assert totals["pppp_descriptor_found"] == 0
    assert totals["pppp_descriptor_parsed"] == 0
    assert totals["pppp_descriptor_skipped"] == 0

    descriptor_hypotheses = summary["by_hypothesis"]["descriptor"]
    assert descriptor_hypotheses["H1_le16"] == 2
    assert descriptor_hypotheses["unparsed_descriptor"] == 0

    name_list_modules = {
        entry["module_name"]: entry for entry in summary["name_list_modules"]
    }
    assert name_list_modules["SYSGEN.TEST"]["count"] == 1
    assert name_list_modules["SYSGEN.TEST"]["sample_locations"][0]["sector"] == 1


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

    assert modules_plain == modules_with_equals == {"SYSGEN.TEST", "SYSGEN.NOPE"}

    totals = summary_with_equals["totals"]
    assert totals["descriptor_records_found"] == 2
    assert totals["descriptor_records_parsed"] == 2
    assert totals["extracted_modules"] == 2
    assert totals["missing_ref_modules"] == 0
    assert totals["bytes_written_total"] > 0

    by_hypothesis = summary_with_equals["by_hypothesis"]["descriptor"]
    assert by_hypothesis["H1_le16"] == 2
    assert by_hypothesis["unparsed_descriptor"] == 0


def test_pppp_descriptor_upgrade(tmp_path: Path) -> None:
    out_dir = _build_pppp_descriptor_fixture(tmp_path)
    derived_dir = tmp_path / "derived_pppp"
    summary = _run_extract_cli(out_dir, derived_dir, "CONTROL.")

    module_path = derived_dir / "CONTROL.AT.bin"
    assert module_path.exists()
    assert module_path.stat().st_size == 3 * 256

    totals = summary["totals"]
    assert totals["descriptor_records_found"] == 1
    assert totals["descriptor_records_parsed"] == 0
    assert totals["descriptor_records_skipped"] == 1
    assert totals["pppp_descriptor_found"] == 1
    assert totals["pppp_descriptor_parsed"] == 1
    assert totals["pppp_descriptor_skipped"] == 0

    pppp_hypotheses = summary["by_hypothesis"]["pppp_descriptor"]
    assert pppp_hypotheses["H1_le16"] == 1
    assert pppp_hypotheses["unparsed_descriptor"] == 0


def test_require_name_in_pppp_list(tmp_path: Path) -> None:
    out_dir = _build_fake_reconstruct(tmp_path)
    derived_dir = tmp_path / "derived_filtered"
    summary = _run_extract_cli(
        out_dir,
        derived_dir,
        "SYSGEN.",
        ["--require-name-in-pppp-list"],
    )

    modules = {mod["module_name"] for mod in summary["modules"]}

    assert "SYSGEN.TEST" in modules
    assert "SYSGEN.NOPE" not in modules

    totals = summary["totals"]
    assert totals["descriptor_records_found"] == 2
    assert totals["descriptor_records_parsed"] == 1
    assert totals["descriptor_records_filtered_by_pppp"] == 1

    filtered = summary["descriptor_records_filtered"]
    assert any(entry["module_name"] == "SYSGEN.NOPE" for entry in filtered)


def test_pppp_descriptor_span(tmp_path: Path) -> None:
    out_dir = _build_pppp_span_fixture(tmp_path)
    derived_dir = tmp_path / "derived_pppp_span"
    summary = _run_extract_cli(
        out_dir, derived_dir, "CONTROL.", extra_args=["--pppp-span-sectors", "1"]
    )

    module_path = derived_dir / "CONTROL.QUEUE.bin"
    assert module_path.exists()
    assert module_path.stat().st_size == 3 * 256

    control_sidecar = json.loads((derived_dir / "CONTROL.QUEUE.json").read_text())
    assert control_sidecar["record_type"] == "pppp_descriptor"
    assert control_sidecar["pointer_window"]["span_sectors_used"] == 1

    totals = summary["totals"]
    assert totals["descriptor_records_found"] == 1
    assert totals["descriptor_records_parsed"] == 0
    assert totals["pppp_descriptor_found"] == 1
    assert totals["pppp_descriptor_parsed"] == 1
    assert totals["pppp_descriptor_skipped"] == 0
