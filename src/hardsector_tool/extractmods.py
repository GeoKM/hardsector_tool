"""Experimental module carving from reconstructed logical sectors."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DescriptorHit = tuple[str, int, int, int]


SYSGEN_RE = re.compile(rb"=([A-Z0-9]{2,12}.[A-Z0-9]{2,20}(?:.[A-Z0-9]{2,20})*)/{4,}")
PPPP_RE = re.compile(rb"pppp=([A-Z0-9]{2,12}\.[A-Z0-9]{2,12})")


@dataclass
class Geometry:
    tracks: Sequence[int]
    sectors_per_track: int
    total_sectors: int


@dataclass
class HypothesisResult:
    name: str
    refs_linear: list[int]
    score: float
    parsed_items: int


def read_manifest(out_dir: Path) -> dict:
    """Load manifest.json from a reconstruction output directory."""

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {out_dir}")
    return json.loads(manifest_path.read_text())


def lin_to_ts(lin: int, sectors_per_track: int = 16) -> tuple[int, int]:
    """Convert linear sector number to (track, sector)."""

    return lin // sectors_per_track, lin % sectors_per_track


def load_sector_db(out_dir: Path) -> dict[tuple[int, int], bytes]:
    """Load all Txx_Syy.bin payloads into memory."""

    manifest = read_manifest(out_dir)
    tracks = [entry.get("track_number") for entry in manifest.get("tracks", [])]
    tracks = [t for t in tracks if t is not None]
    sectors_per_track = 16
    expected = manifest.get("totals", {}).get("expected_sectors")
    if expected and tracks:
        derived = expected // len(tracks)
        if derived:
            sectors_per_track = derived

    sector_db: dict[tuple[int, int], bytes] = {}
    for track in tracks:
        for sector in range(sectors_per_track):
            path = out_dir / "sectors" / f"T{track:02d}_S{sector:02d}.bin"
            if not path.exists():
                continue
            sector_db[(track, sector)] = path.read_bytes()
    return sector_db


def _safe_mkdir(path: Path, force: bool = False) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Output directory {path} already exists; use --force to overwrite"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _derive_geometry(manifest: dict) -> Geometry:
    tracks = [entry.get("track_number") for entry in manifest.get("tracks", [])]
    tracks = [t for t in tracks if t is not None]
    if not tracks:
        tracks = list(range(77))

    expected_total = manifest.get("totals", {}).get("expected_sectors") or 0
    sectors_per_track = 16
    if expected_total and len(tracks) and expected_total % len(tracks) == 0:
        sectors_per_track = expected_total // len(tracks)

    total_sectors = sectors_per_track * len(tracks)
    return Geometry(
        tracks=tracks, sectors_per_track=sectors_per_track, total_sectors=total_sectors
    )


def _scan_descriptors(payload: bytes, track: int, sector: int) -> list[DescriptorHit]:
    hits: list[DescriptorHit] = []
    for match in SYSGEN_RE.finditer(payload):
        name = match.group(1).decode("ascii", errors="ignore")
        after = match.end()
        hits.append((name, track, sector, after))
    return hits


def _decode_le16(
    data: bytes, offset: int, *, total_sectors: int, min_refs: int, max_refs: int
) -> HypothesisResult:
    refs: list[int] = []
    parsed = 0
    invalid_run = 0
    sentinel_run = 0
    for idx in range(offset, min(len(data), offset + max_refs * 2), 2):
        if idx + 2 > len(data):
            break
        word = int.from_bytes(data[idx : idx + 2], "little")
        parsed += 1
        if word >= total_sectors:
            invalid_run += 1
            if invalid_run >= 2:
                break
            continue
        invalid_run = 0
        if word in (0, 0xFFFF) and len(refs) >= min_refs:
            break
        refs.append(word)
        if word in (0, 0xFFFF):
            sentinel_run += 1
            if sentinel_run >= 3:
                break
        else:
            sentinel_run = 0
    score = _score_refs(refs, parsed, min_refs)
    return HypothesisResult("H1_le16", refs, score, parsed)


def _decode_be16(
    data: bytes, offset: int, *, total_sectors: int, min_refs: int, max_refs: int
) -> HypothesisResult:
    refs: list[int] = []
    parsed = 0
    invalid_run = 0
    sentinel_run = 0
    for idx in range(offset, min(len(data), offset + max_refs * 2), 2):
        if idx + 2 > len(data):
            break
        word = int.from_bytes(data[idx : idx + 2], "big")
        parsed += 1
        if word >= total_sectors:
            invalid_run += 1
            if invalid_run >= 2:
                break
            continue
        invalid_run = 0
        if word in (0, 0xFFFF) and len(refs) >= min_refs:
            break
        refs.append(word)
        if word in (0, 0xFFFF):
            sentinel_run += 1
            if sentinel_run >= 3:
                break
        else:
            sentinel_run = 0
    score = _score_refs(refs, parsed, min_refs)
    return HypothesisResult("H2_be16", refs, score, parsed)


def _decode_le32_hi0(
    data: bytes, offset: int, *, total_sectors: int, min_refs: int, max_refs: int
) -> HypothesisResult:
    refs: list[int] = []
    parsed = 0
    invalid_run = 0
    sentinel_run = 0
    for idx in range(offset, min(len(data), offset + max_refs * 4), 4):
        if idx + 4 > len(data):
            break
        lo = int.from_bytes(data[idx : idx + 2], "little")
        hi = int.from_bytes(data[idx + 2 : idx + 4], "little")
        parsed += 1
        if hi != 0 or lo >= total_sectors:
            invalid_run += 1
            if invalid_run >= 2:
                break
            continue
        invalid_run = 0
        if lo == 0 and len(refs) >= min_refs:
            break
        refs.append(lo)
        if lo == 0:
            sentinel_run += 1
            if sentinel_run >= 3:
                break
        else:
            sentinel_run = 0
    score = _score_refs(refs, parsed, min_refs)
    return HypothesisResult("H3_le32_hi0", refs, score, parsed)


def _decode_ts_pairs(
    data: bytes,
    offset: int,
    *,
    geometry: Geometry,
    min_refs: int,
    max_refs: int,
) -> HypothesisResult:
    refs: list[int] = []
    parsed = 0
    invalid_run = 0
    max_track = max(geometry.tracks) if geometry.tracks else 76
    for idx in range(offset, min(len(data), offset + max_refs * 2), 2):
        if idx + 2 > len(data):
            break
        track = data[idx]
        sector = data[idx + 1]
        parsed += 1
        if track == 0 and sector == 0:
            invalid_run += 1
            if invalid_run >= 2:
                break
            continue
        if track > max_track or sector >= geometry.sectors_per_track:
            invalid_run += 1
            if invalid_run >= 2:
                break
            continue
        refs.append(track * geometry.sectors_per_track + sector)
        invalid_run = 0
    score = _score_refs(refs, parsed, min_refs)
    return HypothesisResult("H4_ts_pairs", refs, score, parsed)


def _score_refs(refs: Sequence[int], parsed: int, min_refs: int) -> float:
    if len(refs) < min_refs:
        return 0.0
    if not parsed:
        return 0.0
    unique_ratio = len(set(refs)) / len(refs)
    monotonic = 1.0 if refs == sorted(refs) else 0.0
    fraction = len(refs) / parsed
    return len(refs) + fraction * 2 + unique_ratio * 2 + monotonic


def _choose_hypothesis(
    data: bytes,
    after_offset: int,
    *,
    geometry: Geometry,
    min_refs: int,
    max_refs: int,
    enabled: set[str],
) -> HypothesisResult | None:
    candidates: list[HypothesisResult] = []
    if "H1" in enabled:
        candidates.append(
            _decode_le16(
                data,
                after_offset,
                total_sectors=geometry.total_sectors,
                min_refs=min_refs,
                max_refs=max_refs,
            )
        )
    if "H2" in enabled:
        candidates.append(
            _decode_be16(
                data,
                after_offset,
                total_sectors=geometry.total_sectors,
                min_refs=min_refs,
                max_refs=max_refs,
            )
        )
    if "H3" in enabled:
        candidates.append(
            _decode_le32_hi0(
                data,
                after_offset,
                total_sectors=geometry.total_sectors,
                min_refs=min_refs,
                max_refs=max_refs,
            )
        )
    if "H4" in enabled:
        candidates.append(
            _decode_ts_pairs(
                data,
                after_offset,
                geometry=geometry,
                min_refs=min_refs,
                max_refs=max_refs,
            )
        )

    viable = [c for c in candidates if c.score > 0]
    if not viable:
        return None
    viable.sort(key=lambda c: (c.score, len(c.refs_linear)), reverse=True)
    return viable[0]


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return safe or "unnamed"


def extract_modules(
    out_dir: Path,
    derived_dir: Path,
    *,
    min_refs: int = 3,
    max_refs: int = 2000,
    hypotheses: Iterable[str] = ("H1", "H2", "H3", "H4"),
    only_prefix: str | None = None,
    only_prefix_norm: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    manifest = read_manifest(out_dir)
    geometry = _derive_geometry(manifest)
    sector_db = load_sector_db(out_dir)
    enabled = {h.strip() for h in hypotheses if h.strip()}

    prefix_clean = only_prefix.strip() if only_prefix else None
    prefix_norm = (only_prefix_norm or prefix_clean or "").strip().lstrip("=")
    prefix_norm = prefix_norm or None
    prefix_norm_upper = prefix_norm.upper() if prefix_norm else None
    prefix_raw_upper = prefix_clean.upper() if prefix_clean else None

    _safe_mkdir(derived_dir, force=force)

    descriptors: list[DescriptorHit] = []
    pppp_names: set[str] = set()
    for (track, sector), payload in sector_db.items():
        descriptors.extend(_scan_descriptors(payload, track, sector))
        for match in PPPP_RE.finditer(payload):
            pppp_names.add(match.group(1).decode("ascii", errors="ignore"))

    extracted: list[dict] = []
    per_hypothesis: dict[str, int] = {
        "H1_le16": 0,
        "H2_be16": 0,
        "H3_le32_hi0": 0,
        "H4_ts_pairs": 0,
        "unparsed": 0,
    }

    descriptors_found = len(descriptors)
    parsed_descriptors = 0
    extracted_modules = 0
    missing_ref_modules = 0
    bytes_written_total = 0

    for name, track, sector, after_offset in descriptors:
        name_upper = name.upper()
        if prefix_norm_upper and not (
            name_upper.startswith(prefix_norm_upper)
            or (prefix_raw_upper and name_upper.startswith(prefix_raw_upper))
        ):
            continue
        payload = sector_db.get((track, sector), b"")
        hypothesis = _choose_hypothesis(
            payload,
            after_offset,
            geometry=geometry,
            min_refs=min_refs,
            max_refs=max_refs,
            enabled=enabled,
        )
        if hypothesis is None:
            per_hypothesis["unparsed"] += 1
            safe_name = _safe_name(name)
            module_entry = {
                "module_name": name,
                "safe_name": safe_name,
                "found_at": {
                    "track": track,
                    "sector": sector,
                    "offset": after_offset,
                },
                "hypothesis_used": None,
                "refs_linear": [],
                "refs_ts": [],
                "refs_count": 0,
                "missing_refs": [],
                "bytes_written": 0,
                "warnings": [],
                "warnings_count": 0,
            }
            extracted.append(module_entry)
            continue

        refs_linear = hypothesis.refs_linear
        refs_ts = [lin_to_ts(ref, geometry.sectors_per_track) for ref in refs_linear]
        missing: list[int] = []
        buffer = bytearray()
        for ref_lin, (t, s) in zip(refs_linear, refs_ts):
            payload_bytes = sector_db.get((t, s))
            if payload_bytes is None:
                missing.append(ref_lin)
                continue
            buffer.extend(payload_bytes)

        warnings: list[str] = []
        if missing:
            warnings.append("some referenced sectors were missing; confidence low")
        if pppp_names and name not in pppp_names:
            warnings.append("name not seen in pppp= list (if present)")

        safe_name = _safe_name(name)
        sidecar = {
            "module_name": name,
            "safe_name": safe_name,
            "found_at": {"track": track, "sector": sector, "offset": after_offset},
            "hypothesis_used": hypothesis.name,
            "refs_linear": refs_linear,
            "refs_ts": [{"track": t, "sector": s} for t, s in refs_ts],
            "refs_count": len(refs_linear),
            "missing_refs": missing,
            "bytes_written": len(buffer),
            "warnings": warnings,
            "warnings_count": len(warnings),
        }

        if not dry_run:
            (derived_dir / f"{safe_name}.bin").write_bytes(buffer)
            (derived_dir / f"{safe_name}.json").write_text(
                json.dumps(sidecar, indent=2)
            )

        per_hypothesis[hypothesis.name] += 1
        parsed_descriptors += 1
        extracted_modules += 1
        if missing:
            missing_ref_modules += 1
        bytes_written_total += 0 if dry_run else len(buffer)
        extracted.append(sidecar)

    skipped_descriptors = descriptors_found - parsed_descriptors

    summary = {
        "modules_extracted": extracted_modules,
        "totals": {
            "descriptors_found": descriptors_found,
            "parsed_descriptors": parsed_descriptors,
            "extracted_modules": extracted_modules,
            "skipped_descriptors": skipped_descriptors,
            "missing_ref_modules": missing_ref_modules,
            "bytes_written_total": bytes_written_total,
        },
        "by_hypothesis": per_hypothesis,
        "geometry": {
            "tracks": len(geometry.tracks),
            "sectors_per_track": geometry.sectors_per_track,
            "total_sectors": geometry.total_sectors,
        },
        "pppp_names": sorted(pppp_names),
        "dry_run": dry_run,
        "modules": extracted,
    }

    (derived_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_arg_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "extract-modules",
        help="Experimental carving of module payloads from reconstructed sectors",
    )
    parser.add_argument("out_dir", type=Path, help="Reconstruction output directory")
    parser.add_argument(
        "--out", required=True, type=Path, help="Derived output directory"
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
        "--only-prefix", help="Only extract descriptors with this name prefix"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Analyze only; do not write .bin files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting derived output directory",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hardsector_tool.extract-modules")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_arg_parser(subparsers)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "extract-modules":
        hypotheses = [h.strip() for h in args.hypotheses.split(",") if h.strip()]
        only_prefix = args.only_prefix.strip() if args.only_prefix else None
        prefix_norm = only_prefix.lstrip("=") if only_prefix else None
        extract_modules(
            args.out_dir,
            args.out,
            min_refs=args.min_refs,
            max_refs=args.max_refs,
            hypotheses=hypotheses,
            only_prefix=only_prefix,
            only_prefix_norm=prefix_norm,
            dry_run=args.dry_run,
            force=args.force,
        )
        return 0

    parser.error(f"Unknown command {args.command}")
    return 1


__all__ = [
    "DescriptorHit",
    "Geometry",
    "HypothesisResult",
    "build_arg_parser",
    "extract_modules",
    "lin_to_ts",
    "load_sector_db",
    "main",
    "read_manifest",
]
