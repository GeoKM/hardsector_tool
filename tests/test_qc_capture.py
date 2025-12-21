from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hardsector_tool import qc


def test_qc_outdir_missing_sector(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    sectors_dir = out_dir / "sectors"
    tracks_dir = out_dir / "tracks"
    sectors_dir.mkdir(parents=True)
    tracks_dir.mkdir(parents=True)

    manifest = {
        "tracks": [
            {"track_number": 0, "recovered_sectors": 2},
            {"track_number": 1, "recovered_sectors": 1},
        ],
        "totals": {"expected_sectors": 4, "written_sectors": 3, "missing_sectors": 1},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))

    for track in [0, 1]:
        track_meta = {"sectors": []}
        if track == 0:
            track_meta["sectors"] = [{"sector_id": 0}, {"sector_id": 1}]
            for sector in [0, 1]:
                sectors_dir.joinpath(f"T{track:02d}_S{sector:02d}.bin").write_bytes(
                    b"data"
                )
        else:
            track_meta["sectors"] = [{"sector_id": 0}]
            sectors_dir.joinpath(f"T{track:02d}_S00.bin").write_bytes(b"data")
        tracks_dir.joinpath(f"T{track:02d}.json").write_text(json.dumps(track_meta))

    report = qc.qc_from_outdir(out_dir, mode="detail")

    assert report["overall"]["status"] == "FAIL"
    assert any(entry["status"] == "MISSING" for entry in report.get("per_sector", []))


def test_qc_outdir_crc_only_warning(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    sectors_dir = out_dir / "sectors"
    tracks_dir = out_dir / "tracks"
    sectors_dir.mkdir(parents=True)
    tracks_dir.mkdir(parents=True)

    manifest = {
        "tracks": [
            {"track_number": 0, "recovered_sectors": 2},
            {"track_number": 1, "recovered_sectors": 2},
        ],
        "totals": {"expected_sectors": 4, "written_sectors": 4, "missing_sectors": 0},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))

    tracks_dir.joinpath("T00.json").write_text(
        json.dumps(
            {"sectors": [{"sector_id": 0, "status": "CRC_FAIL"}, {"sector_id": 1}]}
        )
    )
    tracks_dir.joinpath("T01.json").write_text(
        json.dumps({"sectors": [{"sector_id": 0}, {"sector_id": 1}]})
    )

    for track in [0, 1]:
        for sector in [0, 1]:
            sectors_dir.joinpath(f"T{track:02d}_S{sector:02d}.bin").write_bytes(b"ok")

    report = qc.qc_from_outdir(out_dir, mode="detail")

    assert report["overall"]["status"] == "WARN"
    assert any("CRC-like" in reason for reason in report["overall"]["reasons"])


def test_qc_scp_missing_track(monkeypatch, tmp_path: Path) -> None:
    class FakeTrack:
        def __init__(self) -> None:
            self.revolutions = [SimpleNamespace(index_ticks=1000)]
            self.revolution_count = len(self.revolutions)

        def decode_flux(self, rev_index: int):
            return [10, 12, 11, 13]

    class FakeImage:
        def __init__(self, present: list[int]):
            self._present = present
            self.header = SimpleNamespace(sides=1, revolutions=1)

        @classmethod
        def from_file(cls, path: Path):
            return cls([0])

        def list_present_tracks(self, side: int):
            return self._present

        def read_track(self, track_number: int):
            if track_number not in self._present:
                return None
            return FakeTrack()

    monkeypatch.setattr(qc, "SCPImage", FakeImage)

    report = qc.qc_from_scp(tmp_path / "fake.scp", mode="detail", tracks=[0, 1])

    assert report["overall"]["status"] == "FAIL"
    assert 1 in report["capture_qc"]["missing_tracks"]
