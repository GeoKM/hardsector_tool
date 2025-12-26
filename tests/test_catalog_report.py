import json
from pathlib import Path

from hardsector_tool import catalog_report


def _build_recon_fixture(tmp_path: Path) -> Path:
    recon_dir = tmp_path / "out_fixture"
    sectors_dir = recon_dir / "sectors"
    sectors_dir.mkdir(parents=True)

    # Craft a descriptor with three references and a matching pppp entry.
    pointer_bytes = (0).to_bytes(2, "little") + (1).to_bytes(2, "little") + (2).to_bytes(2, "little")
    sectors_dir.joinpath("T00_S00.bin").write_bytes(b"DATA=HELLO.TXT////" + pointer_bytes)
    sectors_dir.joinpath("T00_S01.bin").write_bytes(b"pppp=HELLO.TXT////" + b"X" * 16)
    sectors_dir.joinpath("T00_S02.bin").write_bytes(b"PAYLOAD-02")
    sectors_dir.joinpath("T00_S03.bin").write_bytes(b"PAYLOAD-03")

    manifest = {
        "image": {"path": "fixture.scp"},
        "mapping": {"mode": "dense", "track_step": 1, "side": 0, "present_scp_tracks": [0]},
        "tracks": [
            {
                "track_number": 0,
                "scp_track_id": 0,
                "sector_size": 16,
                "track_score": 1.0,
            }
        ],
        "totals": {"expected_sectors": 4, "written_sectors": 4, "missing_sectors": 0},
    }
    recon_dir.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2))
    return recon_dir


def test_catalog_report_produces_outputs(tmp_path: Path) -> None:
    recon_dir = _build_recon_fixture(tmp_path)
    out_dir = tmp_path / "reports"

    result = catalog_report.catalog_report(
        recon_dir,
        out_dir=out_dir,
        enable_pppp_descriptors=True,
        min_refs=3,
    )

    json_path = result["json_path"]
    txt_path = result["txt_path"]

    assert json_path.exists()
    assert txt_path.exists()

    report = json.loads(json_path.read_text())
    assert report["summary"]["totals"]["descriptors"] == 1
    assert report["descriptors"][0]["name"] == "HELLO.TXT"
    assert report["descriptors"][0]["ref_count"] >= 3
    assert "HELLO.TXT" in report["names_seen"]

    text_contents = txt_path.read_text()
    assert "HELLO.TXT" in text_contents
    assert "catalog-report" in text_contents
