"""Quality control reporting for SCP captures and reconstruction outputs."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable, Sequence

from .diskdump import _parse_sector_sizes

from .diskdump import DiskReconstructor, map_logical_to_scp_track_id
from .hardsector import group_hard_sectors
from .scp import SCPImage

QC_VERSION = 1


_STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _worse_status(current: str, candidate: str) -> str:
    return candidate if _STATUS_ORDER[candidate] > _STATUS_ORDER[current] else current


def _extract_hole_windows(
    track: object, holes_per_rotation: int | None, *, index_aligned: bool
) -> tuple[list[list[int]] | None, int | None, int | None]:
    """Return grouped hole windows for a track if hard-sector geometry is known."""

    if holes_per_rotation is None:
        return None, None, None

    grouping = group_hard_sectors(
        track,
        sectors_per_rotation=holes_per_rotation,
        index_aligned=index_aligned,
    )
    if not grouping.groups:
        return None, None, None

    intervals = [
        [hole.index_ticks for hole in rotation]
        for rotation in grouping.groups
        if rotation
    ]
    windows = sum(len(rotation) for rotation in intervals)
    return intervals, grouping.sectors_per_rotation, windows


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _base_report(input_type: str, path: Path, mode: str) -> dict:
    return {
        "tool": "hardsector_tool",
        "qc_version": QC_VERSION,
        "input": {"type": input_type, "path": str(path)},
        "mode": mode,
        "overall": {"status": "PASS", "reasons": [], "suggestions": []},
        "pipeline": [],
        "reconstruct": None,
        "capture_qc": None,
        "reconstruction_qc": None,
        "per_track": [],
        "per_sector": [],
        "reconstruction_per_track": [],
        "reconstruction_per_sector": [],
    }


HOLE_INTERVAL_WARN_THRESHOLD = 0.25
HOLE_INTERVAL_FAIL_THRESHOLD = 0.40


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


def qc_from_outdir(
    out_dir: Path, mode: str = "brief", *, show_all_tracks: bool = False
) -> dict:
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    report = _base_report("out_dir", out_dir, mode)
    report["show_all_tracks"] = show_all_tracks
    report["pipeline"] = ["reconstruction_qc"]
    report["reconstruct"] = {"enabled": False, "out_dir": str(out_dir)}

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
    per_sector_failures: list[dict] = []
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
            "scp_track_id": track_entry.get("scp_track_id"),
            "expected_scp_track_id": track_entry.get("expected_scp_track_id"),
            "track_mapping_mode": track_entry.get("track_mapping_mode"),
            "track_step": track_entry.get("track_step"),
            "used_fallback_track": track_entry.get("used_fallback_track"),
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

        for sector in track_meta.get("sectors", []):
            status = (sector.get("status") or "").upper()
            if status in {"CRC_FAIL", "NO_DECODE", "UNREADABLE", "LOW_CONFIDENCE"}:
                entry = {
                    "track": track_number,
                    "sector": int(sector.get("sector_id", -1)),
                    "status": status,
                }
                if mode == "detail":
                    per_sector_entries.append(entry)
                code = status
                sector_status = "FAIL"
                if status == "LOW_CONFIDENCE":
                    sector_status = "WARN"
                    code = "LOW_CONF"
                per_sector_failures.append(
                    {
                        "track": track_number,
                        "sector": int(sector.get("sector_id", -1)),
                        "status": sector_status,
                        "code": code,
                        "detail": sector.get("detail"),
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

    reason_payload = reasons or (
        ["all checks passed"] if overall_status == "PASS" else []
    )
    report["overall"] = {
        "status": overall_status,
        "reasons": reason_payload,
        "suggestions": suggestions,
    }
    report["reconstruction_qc"] = reconstruction_qc
    report["per_track"] = per_track_entries
    report["reconstruction_per_track"] = per_track_entries
    reconstruction_qc["per_sector_failures"] = per_sector_failures + [
        {"track": t, "sector": s, "status": "FAIL", "code": "MISSING", "detail": None}
        for t, s in missing_sectors
    ]
    if mode == "detail":
        bad_sectors = [
            {"track": t, "sector": s, "status": "MISSING"} for t, s in missing_sectors
        ]
        report["per_sector"] = bad_sectors + per_sector_entries
        report["reconstruction_per_sector"] = bad_sectors + per_sector_entries
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


def _hole_interval_stats(values: Iterable[int]) -> dict:
    vals = list(values)
    if not vals:
        return {"mean": None, "stdev": None, "cv": None}
    mean = statistics.mean(vals)
    stdev = statistics.pstdev(vals)
    cv = stdev / mean if mean else None
    return {"mean": mean, "stdev": stdev, "cv": cv}


def _hole_interval_metrics(
    intervals_by_rotation: Sequence[Sequence[int]] | Sequence[object],
    holes_per_rotation: int | None,
) -> dict:
    """Return hole interval statistics grouped by rotation.

    The input may already be grouped (list of list[int]) or a flat list of
    revolution-like objects with an ``index_ticks`` attribute.
    """

    groups: list[list[int]] = []
    if intervals_by_rotation:
        sample = intervals_by_rotation[0]
        if isinstance(sample, Sequence) and not hasattr(sample, "index_ticks"):
            groups = [list(map(int, grp)) for grp in intervals_by_rotation]
        else:
            groups = [[int(getattr(rev, "index_ticks")) for rev in intervals_by_rotation]]

    flat_intervals = [v for grp in groups for v in grp if v is not None]
    base_stats = _hole_interval_stats(flat_intervals)
    cv_noindex = None
    index_gap_ratio = None

    if holes_per_rotation and holes_per_rotation > 1 and groups:
        per_rotation_cv: list[float] = []
        per_rotation_ratios: list[float] = []
        for group in groups:
            if len(group) < 2:
                continue
            max_interval = max(group)
            median_interval = statistics.median(group)
            if median_interval:
                per_rotation_ratios.append(max_interval / median_interval)
            remaining = list(group)
            try:
                remaining.pop(remaining.index(max_interval))
            except ValueError:
                pass
            if remaining:
                per_rotation_cv.append(_hole_interval_stats(remaining).get("cv") or 0.0)

        if per_rotation_cv:
            cv_noindex = statistics.median(per_rotation_cv)
        if per_rotation_ratios:
            index_gap_ratio = statistics.median(per_rotation_ratios)

    return {
        "mean": base_stats.get("mean"),
        "stdev": base_stats.get("stdev"),
        "cv": base_stats.get("cv"),
        "cv_noindex": cv_noindex,
        "index_gap_ratio": index_gap_ratio,
    }


def qc_from_scp(
    image_path: Path,
    *,
    mode: str = "brief",
    tracks: Sequence[int] | None = None,
    side: int = 0,
    track_step: str | int = "auto",
    sectors_per_rotation: int | None = None,
    revs: int | None = None,
    show_all_tracks: bool = False,
) -> dict:
    image_path = Path(image_path)
    image = SCPImage.from_file(image_path)
    present_tracks = image.list_present_tracks(side)
    if tracks is None:
        tracks = list(range(0, 77))

    report = _base_report("scp", image_path, mode)
    report["show_all_tracks"] = show_all_tracks
    report["pipeline"] = ["capture_qc"]
    missing_tracks: list[int] = []
    per_track_entries: list[dict] = []
    suggestions: list[str] = []
    overall_status = "PASS"
    hole_interval_warn_tracks: list[tuple[int, float]] = []
    hole_interval_fail_tracks: list[tuple[int, float]] = []
    anomaly_tracks: list[tuple[int, dict]] = []
    missing_window_tracks: list[tuple[int, int, int]] = []
    index_aligned_flag = bool(getattr(image.header, "flags", 0) & 0x01)
    capture_holes_effective = sectors_per_rotation

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
                    "scp_track_id": None,
                    "expected_scp_track_id": None,
                    "track_mapping_mode": None,
                    "track_step": None,
                    "used_fallback_track": False,
                    "flux_intervals_total": 0,
                    "windows_captured": 0,
                    "expected_windows": None,
                    "hole_interval": {"mean": None, "stdev": None, "cv": None},
                    "revs_estimated": None,
                    "hole_interval_cv_noindex": None,
                    "index_gap_ratio": None,
                    "anomalies": {"dropouts": 0, "noise": 0},
                    "status": "FAIL",
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
                    "flux_intervals_total": 0,
                    "windows_captured": 0,
                    "expected_windows": None,
                    "hole_interval": {"mean": None, "stdev": None, "cv": None},
                    "hole_interval_cv_noindex": None,
                    "index_gap_ratio": None,
                    "revs_estimated": None,
                    "anomalies": {"dropouts": 0, "noise": 0},
                    "status": "FAIL",
                    "notes": ["track not present in SCP"],
                }
            )
            overall_status = _worse_status(overall_status, "FAIL")
            continue

        flux_intervals_total = track_data.revolution_count
        hole_windows, holes_effective, windows_captured = _extract_hole_windows(
            track_data,
            sectors_per_rotation,
            index_aligned=index_aligned_flag,
        )
        expected_windows = (
            holes_effective * revs if holes_effective is not None and revs else None
        )

        intervals_for_metrics = (
            hole_windows
            if hole_windows is not None
            else [[getattr(rev, "index_ticks", 0) for rev in track_data.revolutions]]
        )
        hole_interval = _hole_interval_metrics(
            intervals_for_metrics, holes_effective if hole_windows else None
        )
        if holes_effective is not None:
            capture_holes_effective = holes_effective
        revs_estimated = (
            windows_captured / holes_effective
            if holes_effective and windows_captured is not None
            else None
        )
        anomalies = {"dropouts": 0, "noise": 0}
        if track_data.revolutions:
            flux_segment = track_data.decode_flux(0)
            anomalies["dropouts"], anomalies["noise"] = _flux_anomalies(flux_segment)

        per_track_entries.append(
            {
                "track": logical_track,
                "scp_track_id": scp_track_id,
                "expected_scp_track_id": expected_track_id,
                "track_mapping_mode": mapping_mode,
                "track_step": step,
                "used_fallback_track": used_fallback,
                "flux_intervals_total": flux_intervals_total,
                "windows_captured": windows_captured,
                "expected_windows": expected_windows,
                "hole_interval": hole_interval,
                "hole_interval_cv_noindex": hole_interval.get("cv_noindex"),
                "index_gap_ratio": hole_interval.get("index_gap_ratio"),
                "revs_estimated": revs_estimated,
                "anomalies": anomalies,
                "status": "PASS",
                "notes": [],
            }
        )

        if expected_windows and windows_captured is not None:
            if windows_captured < expected_windows * 0.9:
                overall_status = _worse_status(overall_status, "WARN")
                per_track_entries[-1]["status"] = "WARN"
                missing_window_tracks.append(
                    (logical_track, windows_captured, expected_windows)
                )
        hole_interval_cv = hole_interval.get("cv_noindex") or hole_interval.get("cv")
        if hole_interval_cv is not None:
            if hole_interval_cv >= HOLE_INTERVAL_FAIL_THRESHOLD:
                overall_status = _worse_status(overall_status, "FAIL")
                per_track_entries[-1]["status"] = "FAIL"
                hole_interval_fail_tracks.append((logical_track, hole_interval_cv))
            elif hole_interval_cv >= HOLE_INTERVAL_WARN_THRESHOLD:
                overall_status = _worse_status(overall_status, "WARN")
                per_track_entries[-1]["status"] = "WARN"
                hole_interval_warn_tracks.append((logical_track, hole_interval_cv))
        if anomalies["dropouts"] or anomalies["noise"]:
            overall_status = _worse_status(overall_status, "WARN")
            per_track_entries[-1]["status"] = "WARN"
            anomaly_tracks.append((logical_track, anomalies))

    if missing_tracks:
        overall_status = _worse_status(overall_status, "FAIL")
        suggestions.append("verify capture head/side selection and track range")

    capture_qc = {
        "present_tracks": present_tracks,
        "missing_tracks": missing_tracks,
        "sectors_per_rotation": sectors_per_rotation,
        "holes_per_rotation_effective": capture_holes_effective,
        "revs_requested": revs,
        "params": {
            "tracks": list(tracks),
            "side": side,
            "track_step": str(track_step),
            "sectors_per_rotation": sectors_per_rotation,
            "revs": revs,
        },
        "per_track": per_track_entries,
    }

    report["capture_qc"] = capture_qc
    reason_summaries: list[str] = []

    if missing_tracks:
        reason_summaries.append(f"Missing tracks: {missing_tracks}")
    if missing_window_tracks:
        reason_summaries.append(
            "Missing captured windows on " + f"{len(missing_window_tracks)} track(s)"
        )
    if hole_interval_fail_tracks or hole_interval_warn_tracks:
        affected = len(hole_interval_fail_tracks) + len(hole_interval_warn_tracks)
        reason_summaries.append(
            f"hole timing jitter elevated on {affected}/{len(tracks)} track(s)"
        )
    if anomaly_tracks:
        top_anomalies = sorted(
            anomaly_tracks,
            key=lambda item: item[1].get("dropouts", 0) + item[1].get("noise", 0),
            reverse=True,
        )
        top_desc = ", ".join(
            f"T{track:02d}=" f"{issues.get('dropouts', 0) + issues.get('noise', 0)}"
            for track, issues in top_anomalies[:3]
        )
        reason_summaries.append(
            f"Noise/dropout anomalies on {len(anomaly_tracks)} track(s)"
            + (f" (top: {top_desc})" if top_desc else "")
        )

    scored = [
        (entry, _track_issue_score(entry, None))
        for entry in per_track_entries
        if _track_issue_score(entry, None) > 0
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    for entry, _ in scored[:3]:
        if mode == "brief":
            anomalies = entry.get("anomalies") or {}
            missing = 0
            expected = entry.get("expected_windows")
            if expected is not None:
                missing = max(expected - (entry.get("windows_captured") or 0), 0)
            reason_summaries.append(
                f"Track {entry.get('track')} anomalous (missing_windows={missing} "
                f"dropouts={anomalies.get('dropouts', 0)} noise={anomalies.get('noise', 0)})"
            )

    reason_payload = reason_summaries or (
        ["all checks passed"] if overall_status == "PASS" else []
    )
    report["overall"] = {
        "status": overall_status,
        "reasons": reason_payload,
        "suggestions": suggestions,
    }

    report["per_track"] = per_track_entries
    return report


def _normalize_reconstruct_params(
    *,
    tracks: Sequence[int],
    side: int,
    track_step: str | int,
    logical_sectors: int,
    sectors_per_rotation: int,
    sector_sizes: Sequence[int] | None,
    keep_best: int,
    similarity_threshold: float,
    clock_factor: float,
    dump_raw_windows: bool,
) -> dict:
    return {
        "tracks": list(tracks),
        "side": side,
        "track_step": str(track_step),
        "logical_sectors": logical_sectors,
        "sectors_per_rotation": sectors_per_rotation,
        "sector_sizes": list(sector_sizes) if sector_sizes is not None else None,
        "keep_best": keep_best,
        "similarity_threshold": similarity_threshold,
        "clock_factor": clock_factor,
        "dump_raw_windows": dump_raw_windows,
    }


def _param_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()


def _cache_dir_for_run(
    scp_path: Path, cache_root: Path, params: dict, scp_hash: str | None = None
) -> tuple[Path, str]:
    scp_hash = scp_hash or _hash_file(scp_path)
    param_hash = _param_hash(params)
    dir_name = f"{Path(scp_path).stem}_{scp_hash[:12]}_{param_hash[:8]}"
    return cache_root / dir_name, scp_hash


def _ensure_empty_dir(path: Path, allow_nonempty: bool) -> None:
    if not path.exists():
        return
    if allow_nonempty:
        return
    if any(path.iterdir()):
        raise FileExistsError(
            f"Output directory {path} is not empty; use --force or --force-reconstruct"
        )


def _merge_overall(*reports: dict) -> dict:
    status = "PASS"
    reasons: list[str] = []
    suggestions: list[str] = []
    for rep in reports:
        overall = rep.get("overall") or {}
        status = _worse_status(status, overall.get("status", "PASS"))
        reasons.extend(overall.get("reasons") or [])
        suggestions.extend(overall.get("suggestions") or [])
    reason_payload = reasons or (["all checks passed"] if status == "PASS" else [])
    return {
        "status": status,
        "reasons": reason_payload,
        "suggestions": suggestions,
    }


def _apply_capture_expectations(
    capture_qc: dict, *, holes_per_rotation: int | None, revs: int | None
) -> None:
    if not capture_qc:
        return
    if holes_per_rotation is not None:
        capture_qc["holes_per_rotation_effective"] = holes_per_rotation

    for entry in capture_qc.get("per_track") or []:
        if (
            holes_per_rotation is not None
            and entry.get("revs_estimated") is None
            and entry.get("windows_captured") is not None
        ):
            entry["revs_estimated"] = entry.get("windows_captured") / holes_per_rotation
        if (
            holes_per_rotation is not None
            and revs is not None
            and entry.get("expected_windows") is None
        ):
            entry["expected_windows"] = holes_per_rotation * revs


def _run_reconstruction(
    scp_path: Path,
    output_dir: Path,
    params: dict,
    *,
    force: bool,
    verbose: bool = True,
) -> None:
    reconstructor = DiskReconstructor(
        image_path=scp_path,
        output_dir=output_dir,
        tracks=params["tracks"],
        side=params["side"],
        logical_sectors=params["logical_sectors"],
        sectors_per_rotation=params["sectors_per_rotation"],
        sector_sizes=params["sector_sizes"],
        keep_best=params["keep_best"],
        similarity_threshold=params["similarity_threshold"],
        clock_factor=params["clock_factor"],
        dump_raw_windows=params["dump_raw_windows"],
        write_manifest=True,
        write_report=True,
        force=force,
        track_step=params["track_step"],
    )
    if verbose:
        reconstructor.run()
        return

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(stdout_buffer),
            contextlib.redirect_stderr(stderr_buffer),
        ):
            reconstructor.run()
    except Exception:
        sys.stderr.write(stdout_buffer.getvalue())
        sys.stderr.write(stderr_buffer.getvalue())
        raise


def qc_capture(
    input_path: Path,
    *,
    mode: str = "brief",
    tracks: Sequence[int] | None = None,
    side: int = 0,
    track_step: str | int = "auto",
    sectors_per_rotation: int | None = None,
    revs: int | None = None,
    reconstruct: bool | None = None,
    cache_dir: Path | None = None,
    force_reconstruct: bool = False,
    reconstruct_out: Path | None = None,
    logical_sectors: int = 16,
    recon_sectors_per_rotation: int = 32,
    sector_sizes: Sequence[int] | None = None,
    keep_best: int = 3,
    similarity_threshold: float = 0.80,
    clock_factor: float = 1.0,
    dump_raw_windows: bool = False,
    force: bool = False,
    reconstruct_verbose: bool = False,
    show_all_tracks: bool = False,
) -> dict:
    path = Path(input_path)
    cache_root = Path(cache_dir) if cache_dir is not None else Path(".qc_cache")

    if path.suffix.lower() == ".scp":
        capture_report = qc_from_scp(
            path,
            mode=mode,
            tracks=tracks,
            side=side,
            track_step=track_step,
            sectors_per_rotation=sectors_per_rotation,
            revs=revs,
            show_all_tracks=show_all_tracks,
        )
        _apply_capture_expectations(
            capture_report.get("capture_qc") or {},
            holes_per_rotation=sectors_per_rotation,
            revs=revs,
        )
        use_reconstruct = reconstruct is not False
        scp_hash = _hash_file(path)

        if not use_reconstruct:
            capture_report["pipeline"] = ["capture_qc"]
            capture_report["reconstruct"] = {
                "enabled": False,
                "out_dir": None,
                "cache_dir": str(cache_root),
                "used_cache": False,
                "scp_sha1": scp_hash,
                "params": None,
            }
            return capture_report

        recon_tracks = tracks if tracks is not None else list(range(0, 77))
        recon_sector_sizes = (
            sector_sizes if sector_sizes is not None else _parse_sector_sizes("auto")
        )
        params = _normalize_reconstruct_params(
            tracks=recon_tracks,
            side=side,
            track_step=track_step,
            logical_sectors=logical_sectors,
            sectors_per_rotation=recon_sectors_per_rotation,
            sector_sizes=recon_sector_sizes,
            keep_best=keep_best,
            similarity_threshold=similarity_threshold,
            clock_factor=clock_factor,
            dump_raw_windows=dump_raw_windows,
        )

        cache_root.mkdir(parents=True, exist_ok=True)
        output_dir, scp_hash = _cache_dir_for_run(path, cache_root, params, scp_hash)
        used_cache = False

        if reconstruct_out is not None:
            output_dir = Path(reconstruct_out)
            _ensure_empty_dir(output_dir, allow_nonempty=force or force_reconstruct)
        elif not force_reconstruct and output_dir.exists():
            manifest_path = output_dir / "manifest.json"
            if manifest_path.exists():
                used_cache = True

        if not used_cache:
            _run_reconstruction(
                path,
                output_dir,
                params,
                force=force or force_reconstruct or output_dir.exists(),
                verbose=reconstruct_verbose,
            )

        recon_report = qc_from_outdir(
            output_dir, mode=mode, show_all_tracks=show_all_tracks
        )
        effective_holes = sectors_per_rotation or recon_sectors_per_rotation
        _apply_capture_expectations(
            capture_report.get("capture_qc") or {},
            holes_per_rotation=effective_holes,
            revs=revs,
        )
        combined = _base_report("scp", path, mode)
        combined["show_all_tracks"] = show_all_tracks
        combined["pipeline"] = ["capture_qc", "reconstruct", "reconstruction_qc"]
        combined["capture_qc"] = capture_report.get("capture_qc")
        combined["reconstruction_qc"] = recon_report.get("reconstruction_qc")
        combined["per_track"] = recon_report.get("per_track")
        combined["per_sector"] = recon_report.get("per_sector")
        combined["reconstruction_per_track"] = recon_report.get(
            "reconstruction_per_track"
        )
        combined["reconstruction_per_sector"] = recon_report.get(
            "reconstruction_per_sector"
        )
        combined["capture_per_track"] = capture_report.get("per_track")
        combined["reconstruct"] = {
            "enabled": True,
            "out_dir": str(output_dir),
            "cache_dir": str(output_dir if reconstruct_out else cache_root),
            "used_cache": used_cache,
            "scp_sha1": scp_hash,
            "params": params,
        }
        combined_overall = _merge_overall(capture_report, recon_report)
        capture_status = (capture_report.get("overall") or {}).get("status", "PASS")
        recon_status = (recon_report.get("overall") or {}).get("status", "PASS")
        capture_per_track = (capture_report.get("capture_qc") or {}).get(
            "per_track"
        ) or []
        anomaly_tracks = [
            (entry.get("track"), entry.get("anomalies") or {})
            for entry in capture_per_track
            if (entry.get("anomalies") or {}).get("dropouts")
            or (entry.get("anomalies") or {}).get("noise")
        ]
        if capture_status != "PASS" and recon_status == "PASS":
            noisy_tracks = len(anomaly_tracks)
            combined_reason = (
                f"Reconstruction clean; capture shows minor flux noise on {noisy_tracks} track(s)."
                if noisy_tracks
                else "Reconstruction clean; capture warnings recorded."
            )
            combined_overall["reasons"] = [
                combined_reason,
                *[
                    r
                    for r in combined_overall.get("reasons", [])
                    if r != "all checks passed"
                ],
            ]
            if (
                _capture_noise_only(capture_report)
                and combined_overall.get("status") == "WARN"
            ):
                combined_overall.setdefault("suggestions", []).append(
                    "Suggestion: re-read and compare captures; consider head cleaning if noise persists."
                )
        combined["overall"] = combined_overall
        return combined

    if path.is_dir() and (path / "manifest.json").exists():
        return qc_from_outdir(path, mode=mode, show_all_tracks=show_all_tracks)

    raise ValueError("input must be an .scp file or reconstruction output directory")


def _track_issue_score(capture_entry: dict | None, recon_entry: dict | None) -> int:
    recon_missing = len(recon_entry.get("missing_sectors") or []) if recon_entry else 0
    crc_fail = int(recon_entry.get("crc_fail") or 0) if recon_entry else 0
    no_decode = int(recon_entry.get("no_decode") or 0) if recon_entry else 0
    low_conf = int(recon_entry.get("low_confidence") or 0) if recon_entry else 0
    anomalies = (capture_entry.get("anomalies") or {}) if capture_entry else {}
    noise = int(anomalies.get("noise") or 0)
    dropouts = int(anomalies.get("dropouts") or 0)
    hole_interval = (capture_entry or {}).get("hole_interval") or {}
    hole_interval_cv = (capture_entry or {}).get(
        "hole_interval_cv_noindex"
    ) or hole_interval.get("cv")
    timing_warn = (
        hole_interval_cv is not None
        and hole_interval_cv >= HOLE_INTERVAL_WARN_THRESHOLD
    )

    return (
        100 * recon_missing
        + 50 * no_decode
        + 10 * crc_fail
        + 2 * low_conf
        + noise
        + dropouts
        + (5 if timing_warn else 0)
    )


def _build_track_maps(report: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    capture_entries = report.get("capture_per_track") or report.get("per_track") or []
    recon_entries = report.get("reconstruction_per_track") or []
    capture_map = {
        int(entry.get("track")): entry for entry in capture_entries if "track" in entry
    }
    recon_map = {
        int(entry.get("track")): entry for entry in recon_entries if "track" in entry
    }
    return capture_map, recon_map


def _top_issue_tracks(
    report: dict, limit: int = 5
) -> list[tuple[int, int, dict | None, dict | None]]:
    capture_map, recon_map = _build_track_maps(report)
    track_ids = sorted(set(capture_map.keys()) | set(recon_map.keys()))
    scored: list[tuple[int, int, dict | None, dict | None]] = []
    for track_id in track_ids:
        capture_entry = capture_map.get(track_id)
        recon_entry = recon_map.get(track_id)
        score = _track_issue_score(capture_entry, recon_entry)
        if score > 0:
            scored.append((track_id, score, capture_entry, recon_entry))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def _has_hole_windows(capture: dict, entry: dict) -> bool:
    holes_per_rotation = capture.get("holes_per_rotation_effective")
    if holes_per_rotation is not None and entry.get("windows_captured") is not None:
        return True
    if (
        entry.get("expected_windows") is not None
        and entry.get("windows_captured") is not None
    ):
        return True
    return entry.get("revs_estimated") is not None and entry.get("windows_captured") is not None


def _capture_noise_only(capture_report: dict) -> bool:
    capture_qc = capture_report.get("capture_qc") or {}
    overall = capture_report.get("overall") or {}
    if (overall.get("status") or "PASS").upper() != "WARN":
        return False
    tracks = capture_qc.get("per_track") or capture_report.get("per_track") or []
    if capture_qc.get("missing_tracks"):
        return False
    has_missing_windows = any(
        (entry.get("expected_windows") is not None)
        and (entry.get("windows_captured") or 0) < (entry.get("expected_windows") or 0)
        for entry in tracks
    )
    if has_missing_windows:
        return False
    timing_flags = []
    for entry in tracks:
        hole_cv = entry.get("hole_interval_cv_noindex")
        if hole_cv is None:
            hole_cv = (entry.get("hole_interval") or {}).get("cv")
        if hole_cv is not None:
            timing_flags.append(hole_cv >= HOLE_INTERVAL_WARN_THRESHOLD)
        if (entry.get("status") or "PASS").upper() == "FAIL":
            return False
    if any(timing_flags):
        return False
    anomalies = [entry.get("anomalies") or {} for entry in tracks]
    return any(a.get("noise") or a.get("dropouts") for a in anomalies)


def _format_capture_detail_lines(
    capture: dict, *, status_only: bool = False, show_all: bool = False
) -> list[str]:
    tracks = capture.get("per_track") or []
    lines: list[str] = ["Capture QC (per track):"]
    if not tracks:
        lines.append("  (no capture data)")
        return lines

    for entry in tracks:
        status = (entry.get("status") or "PASS").upper()
        if status == "PASS" and not show_all:
            continue
        anomalies = entry.get("anomalies") or {}
        scp_track = entry.get("scp_track_id")
        scp_note = f" (scp={scp_track})" if scp_track is not None else ""
        expected_windows = entry.get("expected_windows")
        windows_label = "windows"
        window_desc = "n/a"
        flux_intervals = entry.get("flux_intervals_total")
        has_holes = _has_hole_windows(capture, entry)
        if has_holes:
            if expected_windows is not None:
                captured = entry.get("windows_captured")
                captured_str = str(captured) if captured is not None else "n/a"
                window_desc = f"{captured_str}/{expected_windows}"
            elif entry.get("windows_captured") is not None:
                window_desc = str(entry.get("windows_captured"))
        else:
            windows_label = "flux_intervals_total"
            if flux_intervals is not None:
                window_desc = str(flux_intervals)
        hole_cv = entry.get("hole_interval_cv_noindex")
        if hole_cv is None:
            hole_cv = (entry.get("hole_interval") or {}).get("cv")
        hole_note = (
            f"hole_cv_noindex={hole_cv:.3f}"
            if hole_cv is not None
            else "hole_cv_noindex=n/a"
        )
        index_gap = entry.get("index_gap_ratio")
        gap_note = (
            f"index_gap_ratio={index_gap:.2f}"
            if index_gap is not None
            else "index_gap_ratio=n/a"
        )
        line = (
            f"  T{int(entry.get('track', -1)):02d}{scp_note}: {windows_label}={window_desc} "
            f"{gap_note} {hole_note} noise={anomalies.get('noise', 0)} "
            f"dropouts={anomalies.get('dropouts', 0)} status={status}"
        )
        notes = entry.get("notes") or []
        if notes and not status_only:
            line += f" notes={' | '.join(notes)}"
        lines.append(line)
    if len(lines) == 1 and not show_all:
        lines.append("  (all tracks PASS; use --show-all-tracks to display)")
    return lines


def _format_reconstruction_detail_lines(
    recon: dict, failure_cap: int = 200, *, show_all: bool = False
) -> list[str]:
    per_track = recon.get("per_track") or []
    lines = ["Reconstruction QC (per track):"]
    if not per_track:
        lines.append("  (no reconstruction data)")
        return lines

    for entry in per_track:
        missing = len(entry.get("missing_sectors") or [])
        crc_fail = entry.get("crc_fail", 0)
        no_decode = entry.get("no_decode", 0)
        low_conf = entry.get("low_confidence", 0)
        status = "PASS"
        if missing or crc_fail or no_decode:
            status = "FAIL"
        elif low_conf:
            status = "WARN"
        if status == "PASS" and not show_all:
            continue
        line = (
            f"  T{int(entry.get('track', -1)):02d}: sectors={entry.get('sectors_present', 0)}/"
            f"{entry.get('sectors_expected')} missing={missing} crc_fail={entry.get('crc_fail', 0)} "
            f"no_decode={entry.get('no_decode', 0)} low_conf={entry.get('low_confidence', 0)}"
        )
        lines.append(line)

    if len(lines) == 1 and not show_all:
        lines.append("  (all tracks PASS; use --show-all-tracks to display)")

    failures = recon.get("per_sector_failures") or []
    if failures:
        lines.append("Sector failures:")
        for entry in failures[:failure_cap]:
            lines.append(
                f"  T{int(entry.get('track', -1)):02d} S{int(entry.get('sector', -1)):02d} "
                f"{entry.get('status', '').upper()} = {entry.get('code', entry.get('status'))}"
                + (f" ({entry.get('detail')})" if entry.get("detail") else "")
            )
        if len(failures) > failure_cap:
            lines.append(f"  ...and {len(failures) - failure_cap} more")
    return lines


def _summarize_reconstruction_line(report: dict) -> str:
    recon = report.get("reconstruction_qc") or {}
    if not recon:
        return "Reconstruction: not run"
    expected = recon.get("expected_sectors")
    written = recon.get("written_sectors")
    missing = len(recon.get("missing_sectors") or [])
    crc_fail = recon.get("crc_fail_count", 0)
    no_decode = recon.get("no_decode_count", 0)
    low_conf = recon.get("low_confidence_count", 0)
    return (
        "Reconstruction: "
        + f"{written}/{expected} sectors, crc_fail={crc_fail}, "
        + f"no_decode={no_decode}, missing={missing}, low_conf={low_conf}"
    )


def _summarize_capture_line(report: dict) -> str:
    capture = report.get("capture_qc") or {}
    if not capture:
        return "Capture: not run"
    tracks = capture.get("per_track") or report.get("per_track") or []
    present_count = len(tracks)
    missing_tracks = len(capture.get("missing_tracks") or [])
    windows_captured = sum(
        int(entry.get("windows_captured") or 0)
        for entry in tracks
        if entry.get("windows_captured") is not None
    )
    flux_intervals_total = sum(
        int(entry.get("flux_intervals_total") or 0) for entry in tracks
    )
    expected_list = [
        entry.get("expected_windows")
        for entry in tracks
        if entry.get("expected_windows") is not None
    ]
    expected_total = sum(expected_list) if expected_list else None
    anomalies = [entry.get("anomalies") or {} for entry in tracks]
    noise_windows = sum(int(a.get("noise") or 0) for a in anomalies)
    dropout_windows = sum(int(a.get("dropouts") or 0) for a in anomalies)
    holes_known = capture.get("holes_per_rotation_effective") is not None
    windows_known = any(entry.get("windows_captured") is not None for entry in tracks)
    window_total = windows_captured if windows_known else flux_intervals_total
    expected_str = str(expected_total) if expected_total is not None else None
    windows_label = "windows"
    if (not holes_known and expected_total is None) or not windows_known:
        windows_label = "flux_intervals_total"
        expected_str = None
    if expected_str is None:
        window_desc = str(window_total)
    else:
        window_desc = f"{window_total}/{expected_str}"
    return (
        "Capture: "
        + f"tracks present={present_count} missing={missing_tracks}, "
        + f"{windows_label}={window_desc}, noise_windows={noise_windows}, "
        + f"dropout_windows={dropout_windows}"
    )


def _format_top_issues_line(report: dict, limit: int = 5) -> str:
    top_tracks = _top_issue_tracks(report, limit=limit)
    if not top_tracks:
        return "Top issues: No affected tracks detected."
    mode = (report.get("mode") or "brief").lower()
    if mode == "detail":
        formatted = []
        for track_id, score, capture_entry, recon_entry in top_tracks:
            missing = len((recon_entry or {}).get("missing_sectors") or [])
            crc_fail = (recon_entry or {}).get("crc_fail", 0)
            no_decode = (recon_entry or {}).get("no_decode", 0)
            low_conf = (recon_entry or {}).get("low_confidence", 0)
            anomalies = (capture_entry or {}).get("anomalies") or {}
            timing_cv = (capture_entry or {}).get("hole_interval_cv_noindex")
            timing_note = "" if timing_cv is None else f", timing_cv={timing_cv:.3f}"
            formatted.append(
                f"T{track_id:02d} (score={score}; missing={missing} no_decode={no_decode} crc_fail={crc_fail} "
                f"low_conf={low_conf} noise={anomalies.get('noise', 0)} dropouts={anomalies.get('dropouts', 0)}{timing_note})"
            )
        return "Top issues: " + ", ".join(formatted)

    def _brief_issue_token(capture_entry: dict | None, recon_entry: dict | None) -> str:
        if recon_entry:
            missing = len(recon_entry.get("missing_sectors") or [])
            if missing:
                return f"missing={missing}"
            no_decode = int(recon_entry.get("no_decode") or 0)
            if no_decode:
                return f"no_decode={no_decode}"
            crc_fail = int(recon_entry.get("crc_fail") or 0)
            if crc_fail:
                return f"crc_fail={crc_fail}"
            low_conf = int(recon_entry.get("low_confidence") or 0)
            if low_conf:
                return f"low_conf={low_conf}"

        if capture_entry:
            anomalies = capture_entry.get("anomalies") or {}
            dropouts = int(anomalies.get("dropouts") or 0)
            noise = int(anomalies.get("noise") or 0)
            expected = capture_entry.get("expected_windows")
            captured = capture_entry.get("windows_captured") or 0
            if expected is not None and captured < expected:
                missing = max(expected - captured, 0)
                return f"missing_windows={missing}"
            if dropouts:
                return f"dropouts={dropouts}"
            if noise:
                return f"noise={noise}"
            hole_cv = capture_entry.get("hole_interval_cv_noindex")
            if hole_cv is None:
                hole_cv = (capture_entry.get("hole_interval") or {}).get("cv")
            if hole_cv is not None and hole_cv >= HOLE_INTERVAL_WARN_THRESHOLD:
                return f"hole_cv={hole_cv:.3f}"

        return "issue"

    compact = [
        f"T{track_id:02d} {_brief_issue_token(capture_entry, recon_entry)}"
        for track_id, _, capture_entry, recon_entry in top_tracks
    ]
    return "Top issues: " + ", ".join(compact)


def format_reconstruction_report(
    report: dict,
    limit: int = 5,
    failure_cap: int = 100,
    *,
    include_summary: bool = True,
) -> str:
    lines: list[str] = []
    show_all = bool(report.get("show_all_tracks"))
    if include_summary:
        lines.extend([summarize_qc(report), ""])
    lines.append(_summarize_reconstruction_line(report))
    recon_detail = dict(report.get("reconstruction_qc") or {})
    recon_detail.setdefault(
        "per_track",
        report.get("reconstruction_per_track") or report.get("per_track") or [],
    )
    recon_detail.setdefault(
        "per_sector_failures",
        (report.get("reconstruction_qc") or {}).get("per_sector_failures")
        or report.get("reconstruction_per_sector")
        or report.get("per_sector")
        or [],
    )
    lines.extend(
        _format_reconstruction_detail_lines(
            recon_detail, failure_cap=failure_cap, show_all=show_all
        )
    )
    lines.append(_format_top_issues_line(report, limit=limit))
    return "\n".join(lines)


def format_capture_report(
    report: dict, limit: int = 5, *, include_summary: bool = True
) -> str:
    lines: list[str] = []
    show_all = bool(report.get("show_all_tracks"))
    if include_summary:
        lines.extend([summarize_qc(report), ""])
    lines.append(_summarize_capture_line(report))
    lines.extend(
        _format_capture_detail_lines(
            report.get("capture_qc") or {}, show_all=show_all
        )
    )
    lines.append(_format_top_issues_line(report, limit=limit))
    return "\n".join(lines)


def format_detail_summary(report: dict, limit: int = 5) -> str:
    lines: list[str] = []
    show_all = bool(report.get("show_all_tracks"))
    lines.append(summarize_qc(report))
    lines.append(_summarize_reconstruction_line(report))
    lines.append(_summarize_capture_line(report))
    lines.append(_format_top_issues_line(report, limit=limit))

    if report.get("mode") == "detail":
        lines.append("")
        lines.extend(
            _format_capture_detail_lines(
                report.get("capture_qc") or {}, show_all=show_all
            )
        )
        lines.append("")
        recon_detail = dict(report.get("reconstruction_qc") or {})
        recon_detail.setdefault(
            "per_track",
            report.get("reconstruction_per_track") or report.get("per_track") or [],
        )
        recon_detail.setdefault(
            "per_sector_failures",
            (report.get("reconstruction_qc") or {}).get("per_sector_failures")
            or report.get("reconstruction_per_sector")
            or report.get("per_sector")
            or [],
        )
        recon_section = _format_reconstruction_detail_lines(
            recon_detail, failure_cap=200, show_all=show_all
        )
        lines.extend(recon_section)

    cache_note = report.get("reconstruct") or {}
    if cache_note.get("enabled"):
        cache_line = (
            "Reconstruction: reused cache at " + str(cache_note.get("out_dir"))
            if cache_note.get("used_cache")
            else "Reconstruction: generated cache at " + str(cache_note.get("out_dir"))
        )
        lines.append(cache_line)
    return "\n".join(lines)


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
