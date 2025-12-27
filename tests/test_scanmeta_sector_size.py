import json
from pathlib import Path

from hardsector_tool.scanmeta import scan_metadata


def _build_recon(tmp_path: Path) -> Path:
    recon_dir = tmp_path / "recon"
    sectors_dir = recon_dir / "sectors"
    tracks_dir = recon_dir / "tracks"
    sectors_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tracks": [
            {
                "track_number": 0,
                "sector_size": 128,
                "track_score": 1.0,
            }
        ],
        "totals": {
            "expected_sectors": 16,
            "written_sectors": 16,
            "missing_sectors": 0,
        },
    }
    recon_dir.joinpath("manifest.json").write_text(json.dumps(manifest))

    for sector in range(16):
        payload = bytes([sector]) * 128
        sectors_dir.joinpath(f"T00_S{sector:02d}.bin").write_bytes(payload)

    return recon_dir


def test_scan_metadata_prefers_track_json_then_sector_files(tmp_path: Path) -> None:
    recon_dir = _build_recon(tmp_path)
    track_json = recon_dir / "tracks" / "T00.json"
    track_json.write_text(json.dumps({"selected_sector_size": 128}))

    result = scan_metadata(recon_dir)
    summary = result["summary"]
    assert summary["sector_size"] == 128
    assert summary["sector_size_inferred"] == 128
    assert summary["sector_size_source"] == "track_json"

    track_json.unlink()
    result = scan_metadata(recon_dir)
    summary = result["summary"]
    assert summary["sector_size"] == 128
    assert summary["sector_size_inferred"] == 128
    assert summary["sector_size_source"] == "sector_files"
