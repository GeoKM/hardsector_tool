"""Evidence-backed catalog report generator."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import __version__
from .diskdump import DiskReconstructor, _parse_sector_sizes, _parse_track_range
from .extractmods import (
    PPPP_RE,
    Geometry,
    HypothesisResult,
    _build_pointer_window,
    _choose_hypothesis,
    _descriptor_pointer_window,
    _derive_geometry,
    _safe_name,
    _scan_descriptors,
    lin_to_ts,
    load_sector_db,
    read_manifest,
)
from .qc import _cache_dir_for_run, _ensure_empty_dir, _normalize_reconstruct_params


@dataclass
class CatalogDescriptor:
    name: str
    source_type: str
    location: dict
    hypothesis: str
    refs_ts: list[dict]
    refs_linear: list[int]
    missing_refs: list[int]
    inferred_size_bytes: int
    warnings: list[str]
    corroborated: bool


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sector_size_hint(manifest: dict) -> int | None:
    tracks = manifest.get("tracks") or []
    sizes: dict[int, int] = defaultdict(int)
    for entry in tracks:
        size = entry.get("sector_size")
        if size:
            sizes[int(size)] += 1
    if not sizes:
        return None
    return max(sizes, key=sizes.get)


def _collect_pppp_entries(sector_db: dict[tuple[int, int], bytes]) -> list[dict]:
    entries: list[dict] = []
    for (track, sector), payload in sector_db.items():
        for match in PPPP_RE.finditer(payload):
            entries.append(
                {
                    "record_type": "pppp_entry",
                    "module_name": match.group(1).decode("ascii", errors="ignore"),
                    "found_at": {
                        "track": track,
                        "sector": sector,
                        "offset": match.start(),
                    },
                    "name_end_offset": match.end(),
                    "source": "pppp",
                }
            )
    return entries


def _build_descriptor_entry(
    name: str,
    source_type: str,
    track: int,
    sector: int,
    offset: int,
    hypothesis: HypothesisResult,
    geometry: Geometry,
    sector_db: dict[tuple[int, int], bytes],
    name_list_modules: dict,
    name_list_upper: set[str],
) -> CatalogDescriptor:
    refs_linear = hypothesis.refs_linear
    refs_ts = [lin_to_ts(ref, geometry.sectors_per_track) for ref in refs_linear]
    missing: list[int] = []
    size_bytes = 0
    for ref_lin, (t, s) in zip(refs_linear, refs_ts):
        payload_bytes = sector_db.get((t, s))
        if payload_bytes is None:
            missing.append(ref_lin)
            continue
        size_bytes += len(payload_bytes)

    warnings: list[str] = []
    if missing:
        warnings.append("some referenced sectors were missing; confidence low")
    if name_list_modules and name not in name_list_modules:
        warnings.append("name not seen in pppp= list (if present)")

    corroborated = name.upper() in name_list_upper

    return CatalogDescriptor(
        name=name,
        source_type=source_type,
        location={"track": track, "sector": sector, "offset": offset},
        hypothesis=hypothesis.name,
        refs_ts=[{"track": t, "sector": s} for t, s in refs_ts],
        refs_linear=refs_linear,
        missing_refs=missing,
        inferred_size_bytes=size_bytes,
        warnings=warnings,
        corroborated=corroborated,
    )


def _gather_descriptors(
    manifest: dict,
    sector_db: dict[tuple[int, int], bytes],
    *,
    min_refs: int,
    max_refs: int,
    hypotheses: Iterable[str],
    enable_pppp_descriptors: bool,
    pppp_span_sectors: int,
    require_name_in_pppp_list: bool,
    only_prefix: str | None,
    only_prefix_norm: str | None,
) -> tuple[list[CatalogDescriptor], set[str], list[dict]]:
    geometry = _derive_geometry(manifest)
    enabled = {h.strip() for h in hypotheses if h.strip()}

    prefix_clean = only_prefix.strip() if only_prefix else None
    prefix_norm = (only_prefix_norm or prefix_clean or "").strip().lstrip("=")
    prefix_norm = prefix_norm or None
    prefix_norm_upper = prefix_norm.upper() if prefix_norm else None
    prefix_raw_upper = prefix_clean.upper() if prefix_clean else None

    pppp_entries = _collect_pppp_entries(sector_db)
    name_list_modules: dict[str, dict] = {}
    for hit in pppp_entries:
        entry = name_list_modules.setdefault(
            hit["module_name"],
            {"module_name": hit["module_name"], "count": 0, "sample_locations": []},
        )
        entry["count"] += 1
        if len(entry["sample_locations"]) < 5:
            entry["sample_locations"].append(hit["found_at"])

    name_list_upper = {name.upper() for name in name_list_modules}

    pppp_name_set = {entry["module_name"].upper() for entry in pppp_entries}
    descriptors: list[CatalogDescriptor] = []

    for (track, sector), payload in sector_db.items():
        for name, _, _, eq_offset, after_offset in _scan_descriptors(
            payload, track, sector
        ):
            name_upper = name.upper()
            if prefix_norm_upper and not (
                name_upper.startswith(prefix_norm_upper)
                or (prefix_raw_upper and name_upper.startswith(prefix_raw_upper))
            ):
                continue

            if require_name_in_pppp_list and name_upper not in pppp_name_set:
                continue

            pointer_bytes, _ = _descriptor_pointer_window(
                payload, track, sector, after_offset
            )
            hypothesis = _choose_hypothesis(
                pointer_bytes,
                0,
                geometry=geometry,
                min_refs=min_refs,
                max_refs=max_refs,
                enabled=enabled,
            )
            if hypothesis is None:
                continue

            descriptors.append(
                _build_descriptor_entry(
                    name,
                    "descriptor",
                    track,
                    sector,
                    eq_offset,
                    hypothesis,
                    geometry,
                    sector_db,
                    name_list_modules,
                    name_list_upper,
                )
            )

    if enable_pppp_descriptors:
        for entry in pppp_entries:
            name = entry["module_name"]
            name_upper = name.upper()
            if prefix_norm_upper and not (
                name_upper.startswith(prefix_norm_upper)
                or (prefix_raw_upper and name_upper.startswith(prefix_raw_upper))
            ):
                continue

            track = entry["found_at"]["track"]
            sector = entry["found_at"]["sector"]
            payload = sector_db.get((track, sector), b"")
            slash_start = entry["name_end_offset"]
            slash_len = 0
            for idx in range(slash_start, len(payload)):
                if payload[idx] != ord("/"):
                    break
                slash_len += 1
            if slash_len < 4:
                continue

            pointer_start = slash_start + slash_len
            pointer_bytes, pointer_window = _build_pointer_window(
                sector_db,
                geometry=geometry,
                track=track,
                sector=sector,
                start_offset=pointer_start,
                span_sectors=pppp_span_sectors,
            )
            if pointer_window["bytes_considered"] == 0:
                continue

            hypothesis = _choose_hypothesis(
                pointer_bytes,
                0,
                geometry=geometry,
                min_refs=min_refs,
                max_refs=max_refs,
                enabled=enabled,
            )
            if hypothesis is None:
                continue

            descriptors.append(
                _build_descriptor_entry(
                    name,
                    "pppp_descriptor",
                    track,
                    sector,
                    entry["found_at"].get("offset", 0),
                    hypothesis,
                    geometry,
                    sector_db,
                    name_list_modules,
                    name_list_upper,
                )
            )

    names_seen = set(pppp_name_set) | {d.name.upper() for d in descriptors}
    return descriptors, names_seen, pppp_entries


def render_text_report(
    *,
    input_path: Path,
    input_type: str,
    reconstruction_path: Path,
    geometry: Geometry,
    holes_per_rev: int | None,
    sector_size: int | None,
    descriptors: list[CatalogDescriptor],
    names_seen: set[str],
) -> str:
    lines: list[str] = []
    lines.append("hardsector_tool catalog-report")
    lines.append(f"Input: {input_path} ({input_type})")
    lines.append(f"Reconstruction: {reconstruction_path}")
    lines.append(
        "Geometry: tracks="
        + str(len(geometry.tracks))
        + f", logical_sectors={geometry.sectors_per_track}"
        + (f", sector_size={sector_size}" if sector_size else "")
        + (f", holes_per_rev={holes_per_rev}" if holes_per_rev else "")
    )
    lines.append(f"Names observed (pppp/name tables): {len(names_seen)}")
    lines.append(f"Extractable entries (descriptor-backed): {len(descriptors)}")
    lines.append("")

    lines.append("Listing (sorted by name):")
    for desc in sorted(descriptors, key=lambda d: d.name.upper()):
        loc = desc.location
        warnings = "; ".join(desc.warnings) if desc.warnings else "none"
        lines.append(
            f"{desc.name}  "
            f"size={desc.inferred_size_bytes:04d}  "
            f"refs={len(desc.refs_linear):02d}  "
            f"hypo={desc.hypothesis}  "
            f"desc_at=T{loc['track']:02d}/S{loc['sector']:02d}+0x{loc['offset']:X}  "
            f"corroborated={'YES' if desc.corroborated else 'NO'}  "
            f"missing={len(desc.missing_refs)}  "
            f"warnings={warnings}"
        )

    lines.append("")
    lines.append("By track:")
    track_map: dict[int, list[CatalogDescriptor]] = defaultdict(list)
    for desc in descriptors:
        track_map[desc.location["track"]].append(desc)
    for track in sorted(track_map):
        entries = track_map[track]
        parts = [
            f"S{entry.location['sector']:02d}+0x{entry.location['offset']:X} ({entry.name})"
            for entry in sorted(entries, key=lambda e: e.location["sector"])
        ]
        lines.append(f"  Track {track:02d}: " + ", ".join(parts))

    return "\n".join(lines) + "\n"


def generate_catalog_report(
    recon_dir: Path,
    *,
    input_path: Path,
    input_type: str,
    min_refs: int,
    max_refs: int,
    hypotheses: Iterable[str],
    enable_pppp_descriptors: bool,
    pppp_span_sectors: int,
    require_name_in_pppp_list: bool,
    only_prefix: str | None,
    only_prefix_norm: str | None,
    holes_per_rev: int | None,
) -> dict:
    manifest = read_manifest(recon_dir)
    sector_db = load_sector_db(recon_dir)
    geometry = _derive_geometry(manifest)
    sector_size = _sector_size_hint(manifest)

    descriptors, names_seen, pppp_entries = _gather_descriptors(
        manifest,
        sector_db,
        min_refs=min_refs,
        max_refs=max_refs,
        hypotheses=hypotheses,
        enable_pppp_descriptors=enable_pppp_descriptors,
        pppp_span_sectors=pppp_span_sectors,
        require_name_in_pppp_list=require_name_in_pppp_list,
        only_prefix=only_prefix,
        only_prefix_norm=only_prefix_norm,
    )

    descriptor_names = {d.name.upper() for d in descriptors}
    evidence_summary = {
        "names_only": len(names_seen - descriptor_names),
        "descriptor_only": len(descriptor_names - names_seen),
        "both": len(descriptor_names & names_seen),
    }

    hypo_counts: dict[str, int] = defaultdict(int)
    for desc in descriptors:
        hypo_counts[desc.hypothesis] += 1

    json_report = {
        "tool": "hardsector_tool",
        "version": __version__,
        "input": {"path": str(input_path), "type": input_type},
        "reconstruction_path": str(recon_dir),
        "geometry": {
            "tracks": len(geometry.tracks),
            "logical_sectors": geometry.sectors_per_track,
            "sector_size": sector_size,
            "holes_per_rev": holes_per_rev,
        },
        "names_seen": sorted(names_seen),
        "descriptors": [
            {
                "name": desc.name,
                "safe_name": _safe_name(desc.name),
                "source_type": desc.source_type,
                "descriptor_location": desc.location,
                "hypothesis": desc.hypothesis,
                "refs": desc.refs_ts,
                "refs_linear": desc.refs_linear,
                "ref_count": len(desc.refs_linear),
                "missing_refs": desc.missing_refs,
                "inferred_size_bytes": desc.inferred_size_bytes,
                "warnings": desc.warnings,
                "corroboration": {"in_pppp_list": desc.corroborated},
            }
            for desc in descriptors
        ],
        "summary": {
            "totals": {
                "descriptors": len(descriptors),
                "names_seen": len(names_seen),
                "pppp_entries": len(pppp_entries),
            },
            "evidence": evidence_summary,
            "hypotheses": dict(sorted(hypo_counts.items())),
        },
    }

    text_report = render_text_report(
        input_path=input_path,
        input_type=input_type,
        reconstruction_path=recon_dir,
        geometry=geometry,
        holes_per_rev=holes_per_rev,
        sector_size=sector_size,
        descriptors=descriptors,
        names_seen=names_seen,
    )

    return {
        "json": json_report,
        "text": text_report,
        "descriptors": descriptors,
        "names_seen": names_seen,
    }


def _ensure_reconstruction(
    input_path: Path,
    *,
    tracks: Sequence[int] | None,
    side: int,
    track_step: str | int,
    logical_sectors: int,
    sectors_per_rotation: int,
    sector_sizes: Sequence[int] | None,
    keep_best: int,
    similarity_threshold: float,
    clock_factor: float,
    dump_raw_windows: bool,
    cache_dir: Path,
    force_reconstruct: bool,
    reconstruct_out: Path | None,
    force: bool,
) -> tuple[Path, bool]:
    cache_root = cache_dir if cache_dir is not None else Path(".qc_cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    recon_tracks = list(tracks) if tracks is not None else list(range(0, 77))
    recon_sector_sizes = sector_sizes
    params = _normalize_reconstruct_params(
        tracks=recon_tracks,
        side=side,
        track_step=track_step,
        logical_sectors=logical_sectors,
        sectors_per_rotation=sectors_per_rotation,
        sector_sizes=recon_sector_sizes,
        keep_best=keep_best,
        similarity_threshold=similarity_threshold,
        clock_factor=clock_factor,
        dump_raw_windows=dump_raw_windows,
    )

    output_dir, _ = _cache_dir_for_run(input_path, cache_root, params)
    used_cache = False

    if reconstruct_out is not None:
        output_dir = reconstruct_out
        _ensure_empty_dir(output_dir, allow_nonempty=force or force_reconstruct)
    elif not force_reconstruct and output_dir.exists():
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            used_cache = True

    if not used_cache:
        reconstructor = DiskReconstructor(
            image_path=input_path,
            output_dir=output_dir,
            tracks=recon_tracks,
            side=side,
            logical_sectors=logical_sectors,
            sectors_per_rotation=sectors_per_rotation,
            sector_sizes=recon_sector_sizes,
            keep_best=keep_best,
            similarity_threshold=similarity_threshold,
            clock_factor=clock_factor,
            dump_raw_windows=dump_raw_windows,
            write_manifest=True,
            write_report=False,
            force=force or force_reconstruct or output_dir.exists(),
            track_step=str(track_step),
        )
        reconstructor.run()

    return output_dir, used_cache


def catalog_report(
    input_path: Path,
    *,
    out_dir: Path | None = None,
    min_refs: int = 3,
    max_refs: int = 2000,
    hypotheses: Iterable[str] = ("H1", "H2", "H3", "H4"),
    enable_pppp_descriptors: bool = False,
    pppp_span_sectors: int = 1,
    require_name_in_pppp_list: bool = False,
    only_prefix: str | None = None,
    only_prefix_norm: str | None = None,
    tracks: Sequence[int] | None = None,
    side: int = 0,
    track_step: str | int = "auto",
    logical_sectors: int = 16,
    sectors_per_rotation: int = 32,
    sector_sizes: Sequence[int] | None = None,
    keep_best: int = 3,
    similarity_threshold: float = 0.80,
    clock_factor: float = 1.0,
    dump_raw_windows: bool = False,
    cache_dir: Path | None = None,
    force_reconstruct: bool = False,
    reconstruct_out: Path | None = None,
    force: bool = False,
) -> dict:
    input_path = Path(input_path)
    output_dir = Path(out_dir) if out_dir is not None else None
    cache_root = cache_dir if cache_dir is not None else Path(".qc_cache")

    if input_path.suffix.lower() == ".scp":
        recon_dir, used_cache = _ensure_reconstruction(
            input_path,
            tracks=tracks,
            side=side,
            track_step=track_step,
            logical_sectors=logical_sectors,
            sectors_per_rotation=sectors_per_rotation,
            sector_sizes=sector_sizes,
            keep_best=keep_best,
            similarity_threshold=similarity_threshold,
            clock_factor=clock_factor,
            dump_raw_windows=dump_raw_windows,
            cache_dir=cache_root,
            force_reconstruct=force_reconstruct,
            reconstruct_out=reconstruct_out,
            force=force,
        )
        input_type = "scp"
        holes_per_rev = sectors_per_rotation
        cache_note = {"cache_dir": str(cache_root), "used_cache": used_cache}
    else:
        recon_dir = input_path
        input_type = "reconstruction"
        holes_per_rev = sectors_per_rotation
        cache_note = {"cache_dir": None, "used_cache": False}

    report_out_dir = (
        output_dir if output_dir is not None else recon_dir / "catalog_report"
    )
    _safe_mkdir(report_out_dir)

    report = generate_catalog_report(
        recon_dir,
        input_path=input_path,
        input_type=input_type,
        min_refs=min_refs,
        max_refs=max_refs,
        hypotheses=hypotheses,
        enable_pppp_descriptors=enable_pppp_descriptors,
        pppp_span_sectors=pppp_span_sectors,
        require_name_in_pppp_list=require_name_in_pppp_list,
        only_prefix=only_prefix,
        only_prefix_norm=only_prefix_norm,
        holes_per_rev=holes_per_rev,
    )

    json_path = report_out_dir / "catalog_report.json"
    txt_path = report_out_dir / "catalog_report.txt"

    json_payload = report["json"]
    json_payload["reconstruction_cache"] = cache_note
    json_path.write_text(json.dumps(json_payload, indent=2))
    txt_path.write_text(report["text"])

    return {
        "json_path": json_path,
        "txt_path": txt_path,
        "report": report["json"],
    }


def build_arg_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "catalog-report",
        help="Generate an evidence-backed catalog listing from reconstructed sectors",
    )
    parser.add_argument("input", type=Path, help="SCP image or reconstruction output")
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Output directory. Default: reconstruction path under catalog_report/ "
            "(or cache reconstruction for SCP inputs)."
        ),
    )
    parser.add_argument(
        "--min-refs", type=int, default=3, help="Minimum pointer references to accept"
    )
    parser.add_argument(
        "--max-refs", type=int, default=2000, help="Safety cap on references parsed"
    )
    parser.add_argument(
        "--hypotheses",
        default="H1,H2,H3,H4",
        help="Comma-separated hypotheses to try (H1,H2,H3,H4)",
    )
    parser.add_argument(
        "--enable-pppp-descriptors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable parsing of pppp= entries with embedded pointers",
    )
    parser.add_argument(
        "--pppp-span-sectors",
        type=int,
        default=1,
        help="Allow including bytes from the next sector when decoding pppp pointers",
    )
    parser.add_argument(
        "--require-name-in-pppp-list",
        action="store_true",
        help="Only decode descriptors whose names also appear in the pppp name list",
    )
    parser.add_argument(
        "--only-prefix", help="Only include descriptors with this name prefix"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".qc_cache"),
        help="Cache directory for reconstruction outputs (shared with qc-capture)",
    )
    parser.add_argument(
        "--force-reconstruct",
        action="store_true",
        help="Ignore cache and rerun reconstruction for SCP inputs",
    )
    parser.add_argument(
        "--reconstruct-out",
        type=Path,
        help="Write reconstruction output to this directory instead of cache",
    )
    parser.add_argument(
        "--tracks", default="0-76", help="Track range for reconstruction from SCP"
    )
    parser.add_argument(
        "--side", type=int, default=0, help="Head/side index to decode from SCP"
    )
    parser.add_argument(
        "--track-step",
        choices=["auto", "1", "2"],
        default="auto",
        help="SCP track spacing: auto-detect, dense (1), or even-only (2)",
    )
    parser.add_argument(
        "--logical-sectors",
        type=int,
        default=16,
        help="Logical sectors per track for reconstruction",
    )
    parser.add_argument(
        "--sectors-per-rotation",
        type=int,
        default=32,
        help="Hard-sector holes per rotation",
    )
    parser.add_argument(
        "--sector-sizes",
        default="auto",
        help="Comma-separated sector sizes to try (auto to guess)",
    )
    parser.add_argument(
        "--keep-best",
        type=int,
        default=3,
        help="Number of rotations to keep when reconstructing",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.80,
        help="Similarity threshold for rotation consensus",
    )
    parser.add_argument(
        "--clock-factor",
        type=float,
        default=1.0,
        help="Clock factor passed to Wang reconstruction",
    )
    parser.add_argument(
        "--dump-raw-windows",
        action="store_true",
        help="Dump decoded window bytes for debugging reconstruction",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting a reconstruction output directory",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hardsector_tool.catalog-report")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_arg_parser(subparsers)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "catalog-report":
        hypotheses = [h.strip() for h in args.hypotheses.split(",") if h.strip()]
        only_prefix = args.only_prefix.strip() if args.only_prefix else None
        prefix_norm = only_prefix.lstrip("=") if only_prefix else None
        sector_sizes = _parse_sector_sizes(args.sector_sizes)
        tracks = _parse_track_range(args.tracks) if args.tracks else None

        catalog_report(
            args.input,
            out_dir=args.out,
            min_refs=args.min_refs,
            max_refs=args.max_refs,
            hypotheses=hypotheses,
            enable_pppp_descriptors=args.enable_pppp_descriptors,
            pppp_span_sectors=max(0, args.pppp_span_sectors),
            require_name_in_pppp_list=args.require_name_in_pppp_list,
            only_prefix=only_prefix,
            only_prefix_norm=prefix_norm,
            tracks=tracks,
            side=args.side,
            track_step=args.track_step,
            logical_sectors=args.logical_sectors,
            sectors_per_rotation=args.sectors_per_rotation,
            sector_sizes=sector_sizes,
            keep_best=args.keep_best,
            similarity_threshold=args.similarity_threshold,
            clock_factor=args.clock_factor,
            dump_raw_windows=args.dump_raw_windows,
            cache_dir=args.cache_dir,
            force_reconstruct=args.force_reconstruct,
            reconstruct_out=args.reconstruct_out,
            force=args.force,
        )
        return 0

    parser.error(f"Unknown command {args.command}")
    return 1


__all__ = [
    "CatalogDescriptor",
    "catalog_report",
    "build_arg_parser",
    "generate_catalog_report",
    "render_text_report",
]
