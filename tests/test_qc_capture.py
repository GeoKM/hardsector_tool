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


def test_scp_capture_reports_hole_interval(monkeypatch, tmp_path: Path) -> None:
    class FakeTrack:
        def __init__(self) -> None:
            self.revolutions = [
                SimpleNamespace(index_ticks=1000),
                SimpleNamespace(index_ticks=3000),
            ]
            self.revolution_count = len(self.revolutions)

        def decode_flux(self, rev_index: int):
            return [10, 12, 11, 13]

    class FakeImage:
        def __init__(self, present: list[int]):
            self._present = present
            self.header = SimpleNamespace(sides=1, revolutions=2)

        @classmethod
        def from_file(cls, path: Path):
            return cls([0])

        def list_present_tracks(self, side: int):
            return self._present

        def read_track(self, track_number: int):
            return FakeTrack()

    monkeypatch.setattr(qc, "SCPImage", FakeImage)

    report = qc.qc_from_scp(
        tmp_path / "fake.scp", mode="brief", tracks=[0], sectors_per_rotation=4, revs=2
    )
    track_entry = report["capture_qc"]["per_track"][0]

    assert track_entry["windows_captured"] == 2
    assert track_entry["expected_windows"] == 8
    assert (
        track_entry["hole_interval"].get("cv") is not None
        or track_entry.get("hole_interval_cv_noindex") is not None
    )
    assert any("hole timing" in reason for reason in report["overall"]["reasons"])


def test_capture_formatting_uses_capture_metrics(monkeypatch, tmp_path: Path) -> None:
    class FakeTrack:
        def __init__(self) -> None:
            self.revolutions = [SimpleNamespace(index_ticks=1000)]
            self.revolution_count = len(self.revolutions)

        def decode_flux(self, rev_index: int):
            return [10, 80, 12, 14, 120]

    class FakeImage:
        def __init__(self) -> None:
            self.header = SimpleNamespace(sides=1, revolutions=1)

        @classmethod
        def from_file(cls, path: Path):
            return cls()

        def list_present_tracks(self, side: int):
            return [0]

        def read_track(self, track_number: int):
            return FakeTrack()

    monkeypatch.setattr(qc, "SCPImage", FakeImage)

    report = qc.qc_from_scp(tmp_path / "fake.scp", mode="detail", tracks=[0])
    output = qc.format_detail_summary(report)

    assert "windows=" in output
    assert "Reconstruction: not run" in output


def test_brief_reasons_are_summarized(monkeypatch, tmp_path: Path) -> None:
    class FakeTrack:
        def __init__(self) -> None:
            self.revolutions = [
                SimpleNamespace(index_ticks=1000),
                SimpleNamespace(index_ticks=5000),
            ]
            self.revolution_count = len(self.revolutions)

        def decode_flux(self, rev_index: int):
            return [10, 12, 11, 13]

    class FakeImage:
        def __init__(self, present: list[int]):
            self._present = present
            self.header = SimpleNamespace(sides=1, revolutions=2)

        @classmethod
        def from_file(cls, path: Path):
            return cls(list(range(5)))

        def list_present_tracks(self, side: int):
            return self._present

        def read_track(self, track_number: int):
            return FakeTrack()

    monkeypatch.setattr(qc, "SCPImage", FakeImage)

    report = qc.qc_from_scp(tmp_path / "fake.scp", mode="brief", tracks=list(range(5)))

    reasons = report["overall"]["reasons"]
    assert len(reasons) < 5
    assert any("hole timing" in reason for reason in reasons)


def test_reconstruction_failures_listed(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    sectors_dir = out_dir / "sectors"
    tracks_dir = out_dir / "tracks"
    sectors_dir.mkdir(parents=True)
    tracks_dir.mkdir(parents=True)

    manifest = {
        "tracks": [
            {"track_number": 0, "recovered_sectors": 1},
        ],
        "totals": {"expected_sectors": 2, "written_sectors": 1, "missing_sectors": 1},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))

    tracks_dir.joinpath("T00.json").write_text(
        json.dumps({"sectors": [{"sector_id": 0, "status": "CRC_FAIL"}]})
    )
    sectors_dir.joinpath("T00_S00.bin").write_bytes(b"ok")

    report = qc.qc_from_outdir(out_dir, mode="detail")
    output = qc.format_detail_summary(report)

    assert "Sector failures:" in output
    assert "T00 S00 FAIL = CRC_FAIL" in output
    assert "T00 S01 FAIL = MISSING" in output


def _write_minimal_recon_out(out_dir: Path) -> None:
    sectors_dir = out_dir / "sectors"
    tracks_dir = out_dir / "tracks"
    sectors_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "tracks": [{"track_number": 0, "recovered_sectors": 1}],
        "totals": {"expected_sectors": 1, "written_sectors": 1, "missing_sectors": 0},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    tracks_dir.joinpath("T00.json").write_text(
        json.dumps({"sectors": [{"sector_id": 0}]})
    )
    sectors_dir.joinpath("T00_S00.bin").write_bytes(b"ok")


def test_reconstruction_logs_suppressed(monkeypatch, tmp_path: Path, capsys) -> None:
    class FakeTrack:
        def __init__(self) -> None:
            self.revolutions = [SimpleNamespace(index_ticks=1000)]
            self.revolution_count = 1

        def decode_flux(self, rev_index: int):
            return [10, 11, 12]

    class FakeImage:
        def __init__(self) -> None:
            self.header = SimpleNamespace(sides=1, revolutions=1)

        @classmethod
        def from_file(cls, path: Path):
            return cls()

        def list_present_tracks(self, side: int):
            return [0]

        def read_track(self, track_number: int):
            return FakeTrack()

    logs: list[str] = []

    class FakeReconstructor:
        def __init__(self, *, output_dir: Path, **_: object) -> None:
            self.output_dir = output_dir

        def run(self) -> None:
            print("NOISY LOG")
            logs.append("ran")
            _write_minimal_recon_out(self.output_dir)

    monkeypatch.setattr(qc, "SCPImage", FakeImage)
    monkeypatch.setattr(qc, "DiskReconstructor", FakeReconstructor)

    scp_path = tmp_path / "image.scp"
    scp_path.write_bytes(b"data")

    qc.qc_capture(
        scp_path,
        mode="brief",
        tracks=[0],
        reconstruct=True,
        reconstruct_out=tmp_path / "out_quiet",
    )
    captured = capsys.readouterr()
    assert "NOISY LOG" not in captured.out

    qc.qc_capture(
        scp_path,
        mode="brief",
        tracks=[0],
        reconstruct=True,
        reconstruct_out=tmp_path / "out_verbose",
        reconstruct_verbose=True,
        force_reconstruct=True,
    )
    captured_verbose = capsys.readouterr()
    assert "NOISY LOG" in captured_verbose.out


def test_qc_capture_pipeline_uses_cache(monkeypatch, tmp_path: Path) -> None:
    class FakeTrack:
        def __init__(self) -> None:
            self.revolutions = [SimpleNamespace(index_ticks=1000)]
            self.revolution_count = 1

        def decode_flux(self, rev_index: int):
            return [10, 11, 12]

    class FakeImage:
        def __init__(self) -> None:
            self.header = SimpleNamespace(sides=1, revolutions=1)

        @classmethod
        def from_file(cls, path: Path):
            return cls()

        def list_present_tracks(self, side: int):
            return [0]

        def read_track(self, track_number: int):
            return FakeTrack()

    runs: list[Path] = []

    class FakeReconstructor:
        def __init__(self, *, output_dir: Path, **_: object) -> None:
            self.output_dir = output_dir

        def run(self) -> None:
            runs.append(self.output_dir)
            _write_minimal_recon_out(self.output_dir)

    monkeypatch.setattr(qc, "SCPImage", FakeImage)
    monkeypatch.setattr(qc, "DiskReconstructor", FakeReconstructor)

    scp_path = tmp_path / "image.scp"
    scp_path.write_bytes(b"data")

    cache_dir = tmp_path / "cache"
    report = qc.qc_capture(scp_path, mode="detail", tracks=[0], cache_dir=cache_dir)

    assert report["pipeline"] == ["capture_qc", "reconstruct", "reconstruction_qc"]
    assert report["reconstruct"]["used_cache"] is False
    assert report["reconstruction_qc"] is not None
    assert runs

    # Second run should reuse cache without invoking reconstructor again
    report_cached = qc.qc_capture(
        scp_path, mode="detail", tracks=[0], cache_dir=cache_dir
    )

    assert report_cached["reconstruct"]["used_cache"] is True
    assert len(runs) == 1
    assert "Reconstruction: reused cache" in qc.format_detail_summary(report_cached)


def test_qc_capture_no_reconstruct(monkeypatch, tmp_path: Path) -> None:
    class FakeImage:
        def __init__(self) -> None:
            self.header = SimpleNamespace(sides=1, revolutions=1)

        @classmethod
        def from_file(cls, path: Path):
            return cls()

        def list_present_tracks(self, side: int):
            return [0]

        def read_track(self, track_number: int):
            return SimpleNamespace(
                revolutions=[SimpleNamespace(index_ticks=1000)],
                revolution_count=1,
                decode_flux=lambda _: [10, 11, 12],
            )

    monkeypatch.setattr(qc, "SCPImage", FakeImage)

    scp_path = tmp_path / "image.scp"
    scp_path.write_bytes(b"data")

    report = qc.qc_capture(scp_path, reconstruct=False, tracks=[0])

    assert report["reconstruction_qc"] is None
    assert report["pipeline"] == ["capture_qc"]
    output = qc.format_detail_summary(report)
    assert "Reconstruction QC" not in output


def test_qc_capture_out_dir_pipeline(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    _write_minimal_recon_out(out_dir)

    report = qc.qc_capture(out_dir, mode="brief")

    assert report["pipeline"] == ["reconstruction_qc"]
    assert report.get("reconstruct", {}).get("enabled") is False


def test_hole_interval_metrics_ignore_index_gap() -> None:
    revolutions = [
        SimpleNamespace(index_ticks=v) for v in [100, 102, 98, 300, 101, 99, 100, 310]
    ]

    stats = qc._hole_interval_metrics(revolutions, 4)

    assert stats["cv"] > 0.1
    assert stats["cv_noindex"] is not None
    assert stats["cv_noindex"] < stats["cv"]


def test_top_issues_empty_when_zero() -> None:
    report = {
        "overall": {
            "status": "PASS",
            "reasons": ["all checks passed"],
            "suggestions": [],
        },
        "capture_qc": {
            "per_track": [
                {
                    "track": 0,
                    "windows_captured": 1,
                    "expected_windows": 1,
                    "hole_interval": {},
                    "hole_interval_cv_noindex": None,
                    "index_gap_ratio": None,
                    "anomalies": {"dropouts": 0, "noise": 0},
                    "status": "PASS",
                }
            ],
            "missing_tracks": [],
        },
        "reconstruction_qc": {},
        "reconstruction_per_track": [
            {
                "track": 0,
                "sectors_present": 1,
                "sectors_expected": 1,
                "missing_sectors": [],
                "crc_fail": 0,
                "no_decode": 0,
                "low_confidence": 0,
            }
        ],
    }

    top_line = qc._format_top_issues_line(report)

    assert "No affected tracks" in top_line
