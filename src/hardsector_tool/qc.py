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
from .scp import SCPImage

QC_VERSION = 1


_STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _worse_status(current: str, candidate: str) -> str:
    return candidate if _STATUS_ORDER[candidate] > _STATUS_ORDER[current] else current


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


def qc_from_outdir(out_dir: Path, mode: str = "brief") -> dict:
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    report = _base_report("out_dir", out_dir, mode)
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
    report["reconstruction_per_track"] = per_track_entries
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
    report["pipeline"] = ["capture_qc"]
    missing_tracks: list[int] = []
    per_track_entries: list[dict] = []
    suggestions: list[str] = []
    overall_status = "PASS"
    hole_interval_warn_tracks: list[tuple[int, float]] = []
    hole_interval_fail_tracks: list[tuple[int, float]] = []
    anomaly_tracks: list[tuple[int, dict]] = []
    missing_window_tracks: list[tuple[int, int, int]] = []

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
                    "windows_captured": 0,
                    "expected_windows": None,
                    "hole_interval": {"mean": None, "stdev": None, "cv": None},
                    "revs_estimated": None,
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
                    "hole_interval": {"mean": None, "stdev": None, "cv": None},
                    "revs_estimated": None,
                    "anomalies": {"dropouts": 0, "noise": 0},
                    "notes": ["track not present in SCP"],
                }
            )
            overall_status = _worse_status(overall_status, "FAIL")
            continue

        windows_captured = track_data.revolution_count
        expected_windows = None
        if sectors_per_rotation and revs:
            expected_windows = sectors_per_rotation * revs

        hole_interval = _hole_interval_stats(
            rev.index_ticks for rev in track_data.revolutions
        )
        revs_estimated = None
        if sectors_per_rotation:
            revs_estimated = (
                windows_captured / sectors_per_rotation if sectors_per_rotation else None
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
                "windows_captured": windows_captured,
                "expected_windows": expected_windows,
                "hole_interval": hole_interval,
                "revs_estimated": revs_estimated,
                "anomalies": anomalies,
                "notes": [],
            }
        )

        if expected_windows and windows_captured is not None:
            if windows_captured < expected_windows * 0.9:
                overall_status = _worse_status(overall_status, "WARN")
                missing_window_tracks.append(
                    (logical_track, windows_captured, expected_windows)
                )
        hole_interval_cv = hole_interval.get("cv")
        if hole_interval_cv is not None:
            if hole_interval_cv >= HOLE_INTERVAL_FAIL_THRESHOLD:
                overall_status = _worse_status(overall_status, "FAIL")
                hole_interval_fail_tracks.append((logical_track, hole_interval_cv))
            elif hole_interval_cv >= HOLE_INTERVAL_WARN_THRESHOLD:
                overall_status = _worse_status(overall_status, "WARN")
                hole_interval_warn_tracks.append((logical_track, hole_interval_cv))
        if anomalies["dropouts"] or anomalies["noise"]:
            overall_status = _worse_status(overall_status, "WARN")
            anomaly_tracks.append((logical_track, anomalies))

    if missing_tracks:
        overall_status = _worse_status(overall_status, "FAIL")
        suggestions.append("verify capture head/side selection and track range")

    capture_qc = {
        "present_tracks": present_tracks,
        "missing_tracks": missing_tracks,
        "sectors_per_rotation": sectors_per_rotation,
        "holes_per_rotation_effective": sectors_per_rotation,
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
            "Missing captured windows on "
            + f"{len(missing_window_tracks)} track(s)"
        )
    if hole_interval_fail_tracks or hole_interval_warn_tracks:
        affected = len(hole_interval_fail_tracks) + len(hole_interval_warn_tracks)
        reason_summaries.append(
            f"hole_interval_cv elevated on {affected}/{len(tracks)} tracks (informational; based on hole-to-hole timing)"
        )
    if anomaly_tracks:
        top_anomalies = sorted(
            anomaly_tracks,
            key=lambda item: item[1].get("dropouts", 0) + item[1].get("noise", 0),
            reverse=True,
        )
        top_desc = ", ".join(
            f"T{track:02d}="
            f"{issues.get('dropouts', 0) + issues.get('noise', 0)}"
            for track, issues in top_anomalies[:3]
        )
        reason_summaries.append(
            f"Noise/dropout anomalies on {len(anomaly_tracks)} track(s)"
            + (f" (top: {top_desc})" if top_desc else "")
        )

    scored = [
        (entry, _capture_issue_score(entry))
        for entry in per_track_entries
        if _capture_issue_score(entry) > 0
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

    report["overall"] = {
        "status": overall_status,
        "reasons": reason_summaries or ["all checks passed"],
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
    return {
        "status": status,
        "reasons": reasons or ["all checks passed"],
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
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
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
        output_dir, scp_hash = _cache_dir_for_run(
            path, cache_root, params, scp_hash
        )
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

        recon_report = qc_from_outdir(output_dir, mode=mode)
        effective_holes = sectors_per_rotation or recon_sectors_per_rotation
        _apply_capture_expectations(
            capture_report.get("capture_qc") or {},
            holes_per_rotation=effective_holes,
            revs=revs,
        )
        combined = _base_report("scp", path, mode)
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
        combined["overall"] = _merge_overall(capture_report, recon_report)
        return combined

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
    hole_interval = entry.get("hole_interval") or {}
    hole_interval_cv = hole_interval.get("cv")
    jitter_score = 0
    if hole_interval_cv is not None:
        if hole_interval_cv >= HOLE_INTERVAL_FAIL_THRESHOLD:
            jitter_score = 5
        elif hole_interval_cv >= HOLE_INTERVAL_WARN_THRESHOLD:
            jitter_score = 1

    return dropouts * 3 + noise * 2 + jitter_score + missing_windows * 4


def _reconstruction_issue_score(entry: dict) -> int:
    missing = len(entry.get("missing_sectors") or [])
    crc_fail = int(entry.get("crc_fail") or 0)
    no_decode = int(entry.get("no_decode") or 0)
    low_conf = int(entry.get("low_confidence") or 0)
    return missing * 5 + no_decode * 3 + crc_fail + low_conf


def format_capture_report(report: dict, limit: int = 5, *, include_summary: bool = True) -> str:
    lines: list[str] = []
    if include_summary:
        lines.extend([summarize_qc(report), ""])
    lines.append("Capture QC:")
    capture = report.get("capture_qc") or {}
    present_tracks = capture.get("present_tracks") or []
    missing_tracks = capture.get("missing_tracks") or []
    tracks = capture.get("per_track") or report.get("per_track") or []
    window_counts = [
        entry.get("windows_captured")
        for entry in tracks
        if entry.get("windows_captured") is not None
    ]
    window_desc = "unknown"
    if window_counts:
        window_desc = (
            f"{int(min(window_counts))}/{statistics.median(window_counts):.1f}/"
            f"{int(max(window_counts))}"
        )
    holes_per_rotation = capture.get("holes_per_rotation_effective") or capture.get(
        "sectors_per_rotation"
    )
    revs_estimated_vals = [
        entry.get("revs_estimated")
        for entry in tracks
        if entry.get("revs_estimated") is not None
    ]
    revs_estimated_desc = (
        f"{statistics.median(revs_estimated_vals):.2f}"
        if revs_estimated_vals
        else "unknown"
    )
    lines.append(
        "  "
        + f"tracks present={len(present_tracks)} missing={len(missing_tracks)} "
        + f"windows_captured={window_desc} holes_per_rotation={holes_per_rotation} "
        + f"revs_estimated={revs_estimated_desc} revs_requested={capture.get('revs_requested')}"
    )

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

            hole_interval = entry.get("hole_interval") or {}
            hole_interval_cv = hole_interval.get("cv")
            hole_note = "n/a"
            if hole_interval_cv is not None:
                if hole_interval_cv >= HOLE_INTERVAL_FAIL_THRESHOLD:
                    hole_note = f"hole_interval_cv={hole_interval_cv:.3f} (FAIL)"
                elif hole_interval_cv >= HOLE_INTERVAL_WARN_THRESHOLD:
                    hole_note = f"hole_interval_cv={hole_interval_cv:.3f} (WARN)"
                else:
                    hole_note = f"hole_interval_cv={hole_interval_cv:.3f}"
            anomalies = entry.get("anomalies") or {}
            scp_track = entry.get("scp_track_id")
            scp_note = f" (scp={scp_track})" if scp_track is not None else ""
            lines.append(
                "  "
                + f"T{entry.get('track'):02d}{scp_note}: windows={window_desc} missing_windows={missing_windows} "
                + f"hole_interval={hole_note} dropouts={anomalies.get('dropouts', 0)} noise={anomalies.get('noise', 0)}"
            )

    if report.get("mode") == "detail" and tracks:
        lines.append("Details:")
        for entry in tracks:
            anomalies = entry.get("anomalies") or {}
            hole_interval = entry.get("hole_interval") or {}
            hole_interval_cv = hole_interval.get("cv")
            hole_note = "hole_interval_cv=n/a"
            if hole_interval_cv is not None:
                if hole_interval_cv >= HOLE_INTERVAL_FAIL_THRESHOLD:
                    hole_note = f"hole_interval_cv={hole_interval_cv:.3f} (FAIL)"
                elif hole_interval_cv >= HOLE_INTERVAL_WARN_THRESHOLD:
                    hole_note = f"hole_interval_cv={hole_interval_cv:.3f} (WARN)"
                else:
                    hole_note = f"hole_interval_cv={hole_interval_cv:.3f}"
            windows_captured = entry.get("windows_captured")
            expected_windows = entry.get("expected_windows")
            missing_windows = 0
            window_desc = "unknown"
            if expected_windows is not None:
                missing_windows = max(expected_windows - (windows_captured or 0), 0)
                window_desc = f"{windows_captured or 0}/{expected_windows}"
            elif windows_captured is not None:
                window_desc = str(windows_captured)

            scp_track = entry.get("scp_track_id")
            scp_note = f" (scp={scp_track})" if scp_track is not None else ""
            detail_line = (
                f"  T{entry.get('track'):02d}{scp_note}: windows={window_desc} missing_windows={missing_windows} "
                f"{hole_note} dropouts={anomalies.get('dropouts', 0)} noise={anomalies.get('noise', 0)}"
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
            "  hole_interval_cv: stdev(window_duration) / mean(window_duration)",
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
    report: dict, limit: int = 5, failure_cap: int = 100, *, include_summary: bool = True
) -> str:
    lines: list[str] = []
    if include_summary:
        lines.extend([summarize_qc(report), ""])
    lines.append("Reconstruction QC:")
    recon = report.get("reconstruction_qc") or {}
    per_track = (
        report.get("reconstruction_per_track")
        or report.get("per_track")
        or []
    )
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
            scp_track = entry.get("scp_track_id")
            scp_note = f" (scp={scp_track})" if scp_track is not None else ""
            lines.append(
                "  "
                + f"T{entry.get('track'):02d}{scp_note}: missing={len(entry.get('missing_sectors') or [])} "
                + f"crc_fail={entry.get('crc_fail', 0)} no_decode={entry.get('no_decode', 0)} low_conf={entry.get('low_confidence', 0)}"
            )

    if report.get("mode") == "detail":
        failures = (
            report.get("reconstruction_per_sector")
            or report.get("per_sector")
            or []
        )
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
    pipeline = report.get("pipeline") or []
    if report.get("input", {}).get("type") == "scp" and "reconstruction_qc" in pipeline:
        cache_note = report.get("reconstruct") or {}
        cache_line = None
        if cache_note.get("enabled"):
            cache_line = (
                "Reconstruction: reused cache at "
                + str(cache_note.get("out_dir"))
                if cache_note.get("used_cache")
                else "Reconstruction: generated cache at " + str(cache_note.get("out_dir"))
            )
        lines: list[str] = []
        lines.append(summarize_qc(report))
        lines.append("")
        lines.extend(format_capture_report(report, limit=limit, include_summary=False).splitlines())
        lines.append("")
        recon_lines = format_reconstruction_report(
            report, limit=limit, failure_cap=100, include_summary=False
        ).splitlines()
        lines.extend(recon_lines)
        if cache_line:
            lines.append(cache_line)
        return "\n".join(lines)
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
