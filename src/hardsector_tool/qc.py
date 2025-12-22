"""Quality control reporting for SCP captures and reconstruction outputs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Iterable, Sequence

from .diskdump import map_logical_to_scp_track_id
from .scp import SCPImage

QC_VERSION = 1


_STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _worse_status(current: str, candidate: str) -> str:
    return candidate if _STATUS_ORDER[candidate] > _STATUS_ORDER[current] else current


def _base_report(input_type: str, path: Path, mode: str) -> dict:
    return {
        "tool": "hardsector_tool",
        "qc_version": QC_VERSION,
        "input": {"type": input_type, "path": str(path)},
        "mode": mode,
        "overall": {"status": "PASS", "reasons": [], "suggestions": []},
        "capture_qc": None,
        "reconstruction_qc": None,
        "per_track": [],
        "per_sector": [],
    }


TIMING_CV_THRESHOLD = 0.02


def summarize_qc(report: dict) -> str:
    overall = report.get("overall", {})
    status = overall.get("status", "PASS")
    reasons = overall.get("reasons") or []
    reason = reasons[0] if reasons else "no issues detected"
    return f"QC: {status} — {reason}"


def _derive_expected_sectors(manifest: dict) -> int | None:
    tracks = manifest.get("tracks") or []
    if not tracks:
        return None
    totals = manifest.get("totals") or {}
    expected = totals.get("expected_sectors")
    if expected:
        return expected // max(1, len(tracks))
    return None


def _collect_track_issue_counts(track_meta: dict) -> tuple[int, int, int]:
    crc_fail = int(track_meta.get("crc_fail_count") or 0)
    no_decode = int(track_meta.get("no_decode_count") or 0)
    low_confidence = int(track_meta.get("low_confidence_count") or 0)
    sectors = track_meta.get("sectors") or []
    for sector in sectors:
        status = (sector.get("status") or "").upper()
        if status == "CRC_FAIL":
            crc_fail += 1
        if status in {"NO_DECODE", "UNREADABLE"}:
            no_decode += 1
        if status == "LOW_CONFIDENCE":
            low_confidence += 1
    return crc_fail, no_decode, low_confidence


def qc_from_outdir(out_dir: Path, mode: str = "brief") -> dict:
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    report = _base_report("out_dir", out_dir, mode)

    if not manifest_path.exists():
        report["overall"] = {
            "status": "FAIL",
            "reasons": ["manifest.json is missing; reconstruction output incomplete"],
            "suggestions": ["rerun reconstruct-disk or verify output directory"],
        }
        return report

    manifest = json.loads(manifest_path.read_text())
    expected_per_track = _derive_expected_sectors(manifest)
    sectors_dir = out_dir / "sectors"
    tracks_dir = out_dir / "tracks"

    missing_tracks: list[int] = []
    missing_sectors: list[tuple[int, int]] = []
    per_track_entries: list[dict] = []
    per_sector_entries: list[dict] = []
    crc_like_total = 0
    no_decode_total = 0
    low_conf_total = 0

    for track_entry in manifest.get("tracks", []):
        track_number = int(track_entry.get("track_number") or 0)
        track_meta_path = tracks_dir / f"T{track_number:02d}.json"
        track_meta = (
            json.loads(track_meta_path.read_text()) if track_meta_path.exists() else {}
        )
        sectors_present_ids = [
            int(s.get("sector_id"))
            for s in track_meta.get("sectors", [])
            if "sector_id" in s
        ]
        recovered = int(
            track_entry.get("recovered_sectors") or len(sectors_present_ids)
        )
        expected = expected_per_track or recovered
        crc_fail, no_decode, low_conf = _collect_track_issue_counts(track_meta)

        if track_entry.get("missing_reason"):
            missing_tracks.append(track_number)

        if expected_per_track is not None:
            for sector_id in range(expected_per_track):
                sector_path = sectors_dir / f"T{track_number:02d}_S{sector_id:02d}.bin"
                if not sector_path.exists():
                    missing_sectors.append((track_number, sector_id))

        entry = {
            "track": track_number,
            "sectors_expected": expected,
            "sectors_present": recovered,
            "missing_sectors": [],
            "track_score": track_entry.get("track_score"),
            "crc_fail": crc_fail,
            "no_decode": no_decode,
            "low_confidence": low_conf,
            "notes": [],
        }

        if expected_per_track is not None:
            missing_ids = [
                i for i in range(expected_per_track) if i not in sectors_present_ids
            ]
            entry["missing_sectors"] = missing_ids
        per_track_entries.append(entry)
        crc_like_total += crc_fail
        no_decode_total += no_decode
        low_conf_total += low_conf

        if mode == "detail":
            for sector in track_meta.get("sectors", []):
                status = (sector.get("status") or "").upper()
                if status in {"CRC_FAIL", "NO_DECODE", "UNREADABLE", "LOW_CONFIDENCE"}:
                    per_sector_entries.append(
                        {
                            "track": track_number,
                            "sector": int(sector.get("sector_id", -1)),
                            "status": status,
                        }
                    )

    totals = manifest.get("totals") or {}
    missing_count = totals.get("missing_sectors") or len(missing_sectors)
    overall_status = "PASS"
    reasons: list[str] = []
    suggestions: list[str] = []

    if missing_tracks:
        overall_status = _worse_status(overall_status, "FAIL")
        reasons.append(f"Missing tracks: {missing_tracks}")
        suggestions.append("check SCP capture coverage or track mapping")

    if missing_count:
        overall_status = _worse_status(overall_status, "FAIL")
        reasons.append(f"Missing sectors detected ({missing_count})")
        suggestions.append("review reconstruct-disk output for gaps")

    if crc_like_total > 0:
        overall_status = _worse_status(overall_status, "WARN")
        reasons.append("CRC-like integrity failures on reconstructed sectors")
        suggestions.append("re-read media or verify checksums with another pass")

    if no_decode_total > 0:
        overall_status = _worse_status(overall_status, "WARN")
        reasons.append(
            "Un-decodable sectors present; capture/decoding limits suspected"
        )
        suggestions.append("inspect flux or adjust reconstruction parameters")

    classification = "unknown"
    if missing_count == 0:
        if crc_like_total > 0 and no_decode_total == 0:
            classification = "CRC-like integrity failures"
        elif no_decode_total > 0:
            classification = "deeper decode/capture issues likely"
        else:
            classification = "no anomalies detected"
    else:
        classification = "deeper decode/capture issues likely"

    reconstruction_qc = {
        "manifest_present": True,
        "missing_tracks": missing_tracks,
        "missing_sectors": missing_sectors,
        "crc_fail_count": crc_like_total,
        "no_decode_count": no_decode_total,
        "low_confidence_count": low_conf_total,
        "classification": classification,
        "expected_sectors": totals.get("expected_sectors"),
        "written_sectors": totals.get("written_sectors"),
    }

    report["overall"] = {
        "status": overall_status,
        "reasons": reasons or ["all checks passed"],
        "suggestions": suggestions,
    }
    report["reconstruction_qc"] = reconstruction_qc
    report["per_track"] = per_track_entries
    if mode == "detail":
        bad_sectors = [
            {"track": t, "sector": s, "status": "MISSING"} for t, s in missing_sectors
        ]
        report["per_sector"] = bad_sectors + per_sector_entries
    return report


def _flux_anomalies(flux: Sequence[int]) -> tuple[int, int]:
    if not flux:
        return 0, 0
    median = statistics.median(flux)
    long_thresh = median * 8 if median else max(flux)
    short_thresh = median * 0.2
    dropouts = sum(1 for v in flux if v > long_thresh)
    noise = sum(1 for v in flux if v < short_thresh)
    return dropouts, noise


def _timing_stats(values: Iterable[int]) -> dict:
    vals = list(values)
    if not vals:
        return {"mean": None, "stdev": None, "cv": None}
    mean = statistics.mean(vals)
    stdev = statistics.pstdev(vals)
    cv = stdev / mean if mean else None
    return {"mean": mean, "stdev": stdev, "cv": cv}


def qc_from_scp(
    image_path: Path,
    *,
    mode: str = "brief",
    tracks: Sequence[int] | None = None,
    side: int = 0,
    track_step: str | int = "auto",
    sectors_per_rotation: int | None = None,
    revs: int | None = None,
) -> dict:
    image_path = Path(image_path)
    image = SCPImage.from_file(image_path)
    present_tracks = image.list_present_tracks(side)
    if tracks is None:
        tracks = list(range(0, 77))

    report = _base_report("scp", image_path, mode)
    missing_tracks: list[int] = []
    per_track_entries: list[dict] = []
    reasons: list[str] = []
    suggestions: list[str] = []
    overall_status = "PASS"

    for logical_track in tracks:
        try:
            scp_track_id, mapping_mode, step, used_fallback, expected_track_id = (
                map_logical_to_scp_track_id(
                    image,
                    logical_track,
                    side=side,
                    track_step=track_step,
                    present_tracks=present_tracks,
                )
            )
        except ValueError as exc:
            missing_tracks.append(logical_track)
            per_track_entries.append(
                {
                    "track": logical_track,
                    "windows_captured": 0,
                    "expected_windows": None,
                    "timing": {"mean": None, "stdev": None, "cv": None},
                    "anomalies": {"dropouts": 0, "noise": 0},
                    "notes": [str(exc)],
                }
            )
            overall_status = _worse_status(overall_status, "FAIL")
            continue

        track_data = image.read_track(scp_track_id)
        if track_data is None:
            missing_tracks.append(logical_track)
            per_track_entries.append(
                {
                    "track": logical_track,
                    "windows_captured": 0,
                    "expected_windows": None,
                    "timing": {"mean": None, "stdev": None, "cv": None},
                    "anomalies": {"dropouts": 0, "noise": 0},
                    "notes": ["track not present in SCP"],
                }
            )
            overall_status = _worse_status(overall_status, "FAIL")
            continue

        observed_windows = None
        if sectors_per_rotation:
            observed_windows = sectors_per_rotation * track_data.revolution_count
        expected_windows = None
        if sectors_per_rotation and revs:
            expected_windows = sectors_per_rotation * revs

        timing = _timing_stats(rev.index_ticks for rev in track_data.revolutions)
        anomalies = {"dropouts": 0, "noise": 0}
        if track_data.revolutions:
            flux_segment = track_data.decode_flux(0)
            anomalies["dropouts"], anomalies["noise"] = _flux_anomalies(flux_segment)

        per_track_entries.append(
            {
                "track": logical_track,
                "windows_captured": observed_windows,
                "expected_windows": expected_windows,
                "timing": timing,
                "anomalies": anomalies,
                "notes": [],
            }
        )

        if expected_windows and observed_windows is not None:
            if observed_windows < expected_windows * 0.9:
                overall_status = _worse_status(overall_status, "WARN")
                reasons.append(
                    f"Track {logical_track} captured {observed_windows} windows; expected {expected_windows}"
                )
        if timing.get("cv") is not None and timing["cv"] > TIMING_CV_THRESHOLD:
            overall_status = _worse_status(overall_status, "WARN")
            reasons.append(
                f"Track {logical_track} shows high revolution jitter (cv={timing['cv']:.2f})"
            )
        if anomalies["dropouts"] or anomalies["noise"]:
            overall_status = _worse_status(overall_status, "WARN")
            reasons.append(
                f"Track {logical_track} has flux anomalies (dropouts={anomalies['dropouts']}, noise={anomalies['noise']})"
            )

    if missing_tracks:
        overall_status = _worse_status(overall_status, "FAIL")
        reasons.append(f"Missing tracks: {missing_tracks}")
        suggestions.append("verify capture head/side selection and track range")

    capture_qc = {
        "present_tracks": present_tracks,
        "missing_tracks": missing_tracks,
        "sectors_per_rotation": sectors_per_rotation,
        "revs_requested": revs,
    }

    report["capture_qc"] = capture_qc
    report["overall"] = {
        "status": overall_status,
        "reasons": reasons or ["all checks passed"],
        "suggestions": suggestions,
    }

    report["per_track"] = per_track_entries
    return report


def qc_capture(
    input_path: Path,
    *,
    mode: str = "brief",
    tracks: Sequence[int] | None = None,
    side: int = 0,
    track_step: str | int = "auto",
    sectors_per_rotation: int | None = None,
    revs: int | None = None,
) -> dict:
    path = Path(input_path)
    if path.suffix.lower() == ".scp":
        report = qc_from_scp(
            path,
            mode=mode,
            tracks=tracks,
            side=side,
            track_step=track_step,
            sectors_per_rotation=sectors_per_rotation,
            revs=revs,
        )
        return report

    if path.is_dir() and (path / "manifest.json").exists():
        return qc_from_outdir(path, mode=mode)

    raise ValueError("input must be an .scp file or reconstruction output directory")


def _capture_issue_score(entry: dict) -> int:
    anomalies = entry.get("anomalies") or {}
    dropouts = int(anomalies.get("dropouts") or 0)
    noise = int(anomalies.get("noise") or 0)
    windows_captured = entry.get("windows_captured") or 0
    expected_windows = entry.get("expected_windows")
    missing_windows = (
        max(expected_windows - windows_captured, 0)
        if expected_windows is not None
        else 0
    )
    timing = entry.get("timing") or {}
    timing_cv = timing.get("cv")
    jitter_flag = timing_cv is not None and timing_cv > TIMING_CV_THRESHOLD

    return dropouts * 3 + noise * 2 + (5 if jitter_flag else 0) + missing_windows * 4


def _reconstruction_issue_score(entry: dict) -> int:
    missing = len(entry.get("missing_sectors") or [])
    crc_fail = int(entry.get("crc_fail") or 0)
    no_decode = int(entry.get("no_decode") or 0)
    low_conf = int(entry.get("low_confidence") or 0)
    return missing * 5 + no_decode * 3 + crc_fail + low_conf


def format_capture_report(report: dict, limit: int = 5) -> str:
    lines: list[str] = [summarize_qc(report), "", "Capture QC:"]
    capture = report.get("capture_qc") or {}
    present_tracks = capture.get("present_tracks") or []
    missing_tracks = capture.get("missing_tracks") or []
    lines.append(
        f"  tracks present={len(present_tracks)} missing={len(missing_tracks)} sectors_per_rotation={capture.get('sectors_per_rotation')} revs_requested={capture.get('revs_requested')}"
    )

    tracks = report.get("per_track") or []
    scored = [entry for entry in tracks if _capture_issue_score(entry) > 0]
    scored.sort(key=_capture_issue_score, reverse=True)
    lines.append("Top affected tracks:")
    if not scored:
        lines.append("  No affected tracks detected.")
    else:
        for entry in scored[:limit]:
            windows_captured = entry.get("windows_captured")
            expected_windows = entry.get("expected_windows")
            window_desc = "unknown"
            missing_windows = 0
            if expected_windows is not None:
                window_desc = f"{windows_captured or 0}/{expected_windows}"
                missing_windows = max(expected_windows - (windows_captured or 0), 0)
            elif windows_captured is not None:
                window_desc = str(windows_captured)

            timing = entry.get("timing") or {}
            timing_cv = timing.get("cv")
            timing_note = "n/a"
            if timing_cv is not None:
                timing_note = (
                    f"cv={timing_cv:.3f} (WARN)"
                    if timing_cv > TIMING_CV_THRESHOLD
                    else f"cv={timing_cv:.3f}"
                )
            anomalies = entry.get("anomalies") or {}
            lines.append(
                "  "
                + f"T{entry.get('track'):02d}: windows={window_desc} missing_windows={missing_windows} "
                + f"timing={timing_note} dropouts={anomalies.get('dropouts', 0)} noise={anomalies.get('noise', 0)}"
            )

    if report.get("mode") == "detail" and tracks:
        lines.append("Details:")
        for entry in tracks:
            anomalies = entry.get("anomalies") or {}
            timing = entry.get("timing") or {}
            timing_cv = timing.get("cv")
            timing_note = (
                f"timing_cv={timing_cv:.3f} (WARN)"
                if timing_cv and timing_cv > TIMING_CV_THRESHOLD
                else (
                    f"timing_cv={timing_cv:.3f}"
                    if timing_cv is not None
                    else "timing_cv=n/a"
                )
            )
            windows_captured = entry.get("windows_captured")
            expected_windows = entry.get("expected_windows")
            missing_windows = 0
            window_desc = "unknown"
            if expected_windows is not None:
                missing_windows = max(expected_windows - (windows_captured or 0), 0)
                window_desc = f"{windows_captured or 0}/{expected_windows}"
            elif windows_captured is not None:
                window_desc = str(windows_captured)

            detail_line = (
                f"  T{entry.get('track'):02d}: windows={window_desc} missing_windows={missing_windows} "
                f"{timing_note} dropouts={anomalies.get('dropouts', 0)} noise={anomalies.get('noise', 0)}"
            )
            notes = entry.get("notes") or []
            if notes:
                detail_line += f" notes={' | '.join(notes)}"
            lines.append(detail_line)

    lines.extend(
        [
            "",
            "Capture legend:",
            "  windows_captured: number of captured hole-to-hole windows",
            "  expected_windows: sectors_per_rotation * revs (if supplied/known)",
            "  timing_cv: stdev(window_duration) / mean(window_duration)",
            "  dropouts: windows with unusually long gaps (potential weak signal)",
            "  noise: windows with unusually many very short intervals (noisy signal)",
        ]
    )
    return "\n".join(lines)


def _reconstruction_top_tracks(report: dict) -> list[dict]:
    tracks = report.get("per_track") or []
    scored = [entry for entry in tracks if _reconstruction_issue_score(entry) > 0]
    scored.sort(key=_reconstruction_issue_score, reverse=True)
    return scored


def _format_failure_entry(entry: dict) -> str:
    status = (entry.get("status") or "").upper()
    status = {"LOW_CONFIDENCE": "LOW_CONF", "UNREADABLE": "NO_DECODE"}.get(
        status, status
    )
    return f"  T{int(entry.get('track', -1)):02d} S{int(entry.get('sector', -1)):02d}: {status}"


def format_reconstruction_report(
    report: dict, limit: int = 5, failure_cap: int = 100
) -> str:
    lines: list[str] = [summarize_qc(report), "", "Reconstruction QC:"]
    recon = report.get("reconstruction_qc") or {}
    per_track = report.get("per_track") or []
    missing_count = len(recon.get("missing_sectors") or [])
    lines.append(
        "  "
        + f"tracks_present={len(per_track)} expected_sectors={recon.get('expected_sectors')} "
        + f"written_sectors={recon.get('written_sectors')} missing_sectors={missing_count} "
        + f"crc_fail={recon.get('crc_fail_count', 0)} no_decode={recon.get('no_decode_count', 0)}"
    )

    top_tracks = _reconstruction_top_tracks(report)
    lines.append("Top affected tracks:")
    if not top_tracks:
        lines.append("  No affected tracks detected.")
    else:
        for entry in top_tracks[:limit]:
            lines.append(
                "  "
                + f"T{entry.get('track'):02d}: missing={len(entry.get('missing_sectors') or [])} "
                + f"crc_fail={entry.get('crc_fail', 0)} no_decode={entry.get('no_decode', 0)} low_conf={entry.get('low_confidence', 0)}"
            )

    if report.get("mode") == "detail":
        failures = report.get("per_sector") or []
        if failures:
            lines.append("Failures:")
            for entry in failures[:failure_cap]:
                lines.append(_format_failure_entry(entry))
            if len(failures) > failure_cap:
                lines.append(f"  (+{len(failures) - failure_cap} more)")

    lines.extend(
        [
            "",
            "Reconstruction legend:",
            "  missing: sector file absent from sectors/ (incomplete reconstruction)",
            "  crc_fail: sector decoded but failed integrity check (often marginal bits)",
            "  no_decode: no consistent sector could be reconstructed (deeper issue)",
            "  low_conf: reconstructed but low consensus (if supported)",
        ]
    )
    return "\n".join(lines)


def format_detail_summary(report: dict, limit: int = 5) -> str:
    if report.get("input", {}).get("type") == "scp":
        return format_capture_report(report, limit=limit)
    return format_reconstruction_report(report, limit=limit)


def default_output_path(input_path: Path, report: dict) -> Path:
    if report["input"]["type"] == "out_dir":
        return Path(input_path) / "qc.json"
    base = Path(input_path).name
    return Path(f"qc_{base}").with_suffix(".json")


__all__ = [
    "qc_capture",
    "qc_from_outdir",
    "qc_from_scp",
    "summarize_qc",
    "format_capture_report",
    "format_reconstruction_report",
    "format_detail_summary",
    "default_output_path",
]
