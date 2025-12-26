import re

from hardsector_tool import qc


def _base_report(mode: str = "brief") -> dict:
    return {
        "mode": mode,
        "overall": {"status": "PASS", "reasons": ["all checks passed"], "suggestions": []},
        "capture_qc": {},
        "reconstruction_qc": None,
    }


def test_sector_failures_render_in_brief_mode() -> None:
    report = _base_report()
    report["overall"]["status"] = "FAIL"
    report["overall"]["reasons"] = ["sector issues detected"]
    report["reconstruction_qc"] = {
        "status": "FAIL",
        "expected_sectors": 4,
        "written_sectors": 3,
        "missing_sectors": [(0, 1)],
        "crc_fail_count": 1,
        "no_decode_count": 0,
        "low_confidence_count": 0,
        "per_sector_failures": [
            {"track": 0, "sector": 1, "code": "MISSING"},
            {"track": 1, "sector": 0, "code": "CRC_FAIL", "detail": None},
        ],
    }

    rendered = qc.format_detail_summary(report)

    assert "Sector failures (2):" in rendered
    assert "T00 S01: MISSING — no sector output written" in rendered
    assert re.search(r"T01 S00: CRC_FAIL — .*integrity check", rendered)


def test_sector_failures_suppressed_when_empty() -> None:
    report = _base_report()
    report["reconstruction_qc"] = {
        "status": "PASS",
        "expected_sectors": 2,
        "written_sectors": 2,
        "missing_sectors": [],
        "crc_fail_count": 0,
        "no_decode_count": 0,
        "low_confidence_count": 0,
        "per_sector_failures": [],
    }

    rendered = qc.format_detail_summary(report)

    assert "Sector failures (" not in rendered
    assert "Sector failures: none detected." in rendered


def test_sector_failure_lines_use_expected_pattern() -> None:
    report = _base_report()
    report["reconstruction_qc"] = {
        "status": "WARN",
        "expected_sectors": 3,
        "written_sectors": 3,
        "missing_sectors": [],
        "crc_fail_count": 0,
        "no_decode_count": 1,
        "low_confidence_count": 0,
        "per_sector_failures": [
            {"track": 2, "sector": 5, "code": "NO_DECODE", "detail": "unstable"}
        ],
    }

    rendered = qc.format_detail_summary(report)

    match = re.search(r"T02 S05: NO_DECODE — could not reconstruct stable sector", rendered)
    assert match
