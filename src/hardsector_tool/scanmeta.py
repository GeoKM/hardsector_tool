"""Metadata scanning over reconstructed logical sectors."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

NAME_EXT_RE = re.compile(rb"[A-Z0-9]{2,12}\.[A-Z0-9]{2,12}")
UPPER_TOKEN_RE = re.compile(rb"[A-Z0-9][A-Z0-9._]{5,24}")
PRINTABLE_RUN_RE = re.compile(rb"[\x20-\x7e]{3,}")

SIGNATURE_STRINGS = [
    b"Office Information System",
    b"Volume Recovery",
    b"Version",
    b"INSTALL",
    b"CONTROL.",
    b"DOS.",
    b"SYSGEN",
    b"SECURITY",
    b"Wang",
    b"SF",
]


@dataclass
class SectorData:
    track: int
    sector: int
    payload: bytes
    sector_size: int


def _load_manifest(out_dir: Path) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {out_dir}")
    return json.loads(manifest_path.read_text())


def _load_sector(
    track: int, sector: int, out_dir: Path, sector_size: int
) -> SectorData | None:
    path = out_dir / "sectors" / f"T{track:02d}_S{sector:02d}.bin"
    if not path.exists():
        return None
    payload = path.read_bytes()
    return SectorData(
        track=track, sector=sector, payload=payload, sector_size=sector_size
    )


def _sector_size_for_track(
    out_dir: Path, track: int, default_sector_size: int = 256
) -> tuple[int, str]:
    """Infer sector size for a track using track JSON or sector payloads."""

    track_json_candidates = [
        out_dir / "tracks" / f"track_{track:02d}.json",
        out_dir / "tracks" / f"T{track:02d}.json",
    ]

    for track_path in track_json_candidates:
        if not track_path.exists():
            continue
        try:
            data = json.loads(track_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        for key in ("selected_sector_size", "sector_size", "best_sector_size"):
            if key not in data:
                continue
            try:
                return int(data[key]), "track_json"
            except (TypeError, ValueError):
                continue

    sector_files = list((out_dir / "sectors").glob(f"T{track:02d}_S*.bin"))
    if sector_files:
        size_counts = Counter(path.stat().st_size for path in sector_files)
        max_count = max(size_counts.values())
        candidates = [size for size, count in size_counts.items() if count == max_count]
        chosen = max(candidates)
        return chosen, "sector_files"

    return default_sector_size, "default"


def _slice_context(payload: bytes, offset: int, radius: int = 32) -> tuple[str, str]:
    start = max(0, offset - radius)
    end = min(len(payload), offset + radius)
    window = payload[start:end]
    ascii_ctx = window.decode("ascii", errors="replace")
    hex_ctx = window.hex()
    return ascii_ctx, hex_ctx


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent


def _pointer_counts(payload: bytes) -> dict[str, int]:
    ts_pairs = 0
    ts_pairs_nonzero = 0
    packed_nibble = 0
    linear_le = 0
    linear_be = 0

    for i in range(len(payload) - 1):
        track = payload[i]
        sector = payload[i + 1]
        if track <= 76 and sector <= 15:
            ts_pairs += 1
            if track or sector:
                ts_pairs_nonzero += 1

        val_le = int.from_bytes(payload[i : i + 2], "little")
        val_be = int.from_bytes(payload[i : i + 2], "big")
        if val_le <= 1231:
            linear_le += 1
        if val_be <= 1231:
            linear_be += 1

    for byte in payload:
        track_nib = (byte >> 4) & 0x0F
        sector_nib = byte & 0x0F
        if track_nib <= 4 and sector_nib <= 15:
            packed_nibble += 1

    return {
        "ts_pairs_count": ts_pairs,
        "ts_pairs_nonzero_count": ts_pairs_nonzero,
        "packed_nibble_count": packed_nibble,
        "linear16_le_count": linear_le,
        "linear16_be_count": linear_be,
    }


def _name_tokens(
    payload: bytes,
) -> tuple[list[tuple[int, bytes]], list[tuple[int, bytes]]]:
    names = [(m.start(), m.group(0)) for m in NAME_EXT_RE.finditer(payload)]
    uppercase = [(m.start(), m.group(0)) for m in UPPER_TOKEN_RE.finditer(payload)]
    return names, uppercase


def _signature_hits(
    payload: bytes, track: int, sector: int, limit: int = 200
) -> list[dict]:
    hits: list[dict] = []
    for token in SIGNATURE_STRINGS:
        start = 0
        while True:
            idx = payload.find(token, start)
            if idx == -1:
                break
            ascii_ctx, hex_ctx = _slice_context(payload, idx)
            hits.append(
                {
                    "type": "text",
                    "token": token.decode("ascii", errors="replace"),
                    "track": track,
                    "sector": sector,
                    "offset": idx,
                    "context_ascii": ascii_ctx,
                    "context_hex": hex_ctx,
                }
            )
            if len(hits) >= limit:
                return hits
            start = idx + 1

    for regex, label in ((NAME_EXT_RE, "name.ext"), (re.compile(rb"pppp="), "pppp=")):
        for match in regex.finditer(payload):
            ascii_ctx, hex_ctx = _slice_context(payload, match.start())
            hits.append(
                {
                    "type": label,
                    "token": match.group(0).decode("ascii", errors="replace"),
                    "track": track,
                    "sector": sector,
                    "offset": match.start(),
                    "context_ascii": ascii_ctx,
                    "context_hex": hex_ctx,
                }
            )
            if len(hits) >= limit:
                return hits
    return hits


def _record_scores(payload: bytes) -> list[dict]:
    record_sizes = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    results: list[dict] = []
    name_positions: dict[int, Counter[int]] = defaultdict(Counter)

    for size in record_sizes:
        if len(payload) % size != 0:
            continue
        records = [payload[i : i + size] for i in range(0, len(payload), size)]
        table_like = 0
        example_fields: list[str] = []
        for ridx, record in enumerate(records):
            printable = PRINTABLE_RUN_RE.search(record)
            ptr_counts = _pointer_counts(record)
            has_pointer = any(ptr_counts.values())
            if printable or has_pointer:
                table_like += 1
            names, upper = _name_tokens(record)
            if names:
                name_positions[size][names[0][0]] += 1
            if printable and len(example_fields) < 3:
                example_fields.append(
                    f"rec{ridx}@{printable.start()}:"
                    f" {printable.group(0).decode('ascii', errors='replace')}"
                )
        score = table_like / (len(payload) // size) if size else 0
        bonus = 0.0
        if name_positions[size]:
            top_off = name_positions[size].most_common(1)[0][1]
            bonus = top_off / (len(payload) // size)
        results.append(
            {
                "record_size": size,
                "score": score + bonus,
                "table_like_ratio": score,
                "name_offset_bonus": bonus,
                "example_fields": example_fields,
            }
        )

    results.sort(key=lambda r: (-r["score"], r["record_size"]))
    return results[:3]


def _directory_score(
    name_density: int, pointer_score: int, record_score: float
) -> float:
    return record_score * 3 + name_density * 0.5 + pointer_score * 0.25


def scan_metadata(out_dir: Path) -> dict:
    manifest = _load_manifest(out_dir)
    track_entries = manifest.get("tracks", [])
    expected_tracks = [
        entry.get("track_number", entry.get("logical_track")) for entry in track_entries
    ]
    expected_tracks = [t for t in expected_tracks if t is not None]
    if not expected_tracks:
        expected_tracks = list(range(77))
    logical_sectors = 16
    default_sector_size = 256

    sectors: list[SectorData] = []
    missing: list[str] = []
    inferred_sizes: list[tuple[int, str]] = []
    for track in expected_tracks:
        sector_size, size_source = _sector_size_for_track(
            out_dir, track, default_sector_size
        )
        inferred_sizes.append((sector_size, size_source))
        for sector in range(logical_sectors):
            entry = _load_sector(track, sector, out_dir, sector_size)
            if entry is None:
                missing.append(f"T{track:02d}_S{sector:02d}")
                continue
            sectors.append(entry)

    sector_size_inferred = default_sector_size
    sector_size_source = "default"
    if inferred_sizes:
        size_counts = Counter(size for size, _ in inferred_sizes)
        max_count = max(size_counts.values())
        candidates = [size for size, count in size_counts.items() if count == max_count]
        sector_size_inferred = max(candidates)
        source_priority = {"track_json": 2, "sector_files": 1, "default": 0}
        for size, source in inferred_sizes:
            if size == sector_size_inferred and source_priority.get(
                source, -1
            ) >= source_priority.get(sector_size_source, -1):
                sector_size_source = source

    summary = {
        "tracks": len(expected_tracks),
        "sectors_per_track": logical_sectors,
        "sector_size": sector_size_inferred,
        "sector_size_inferred": sector_size_inferred,
        "sector_size_source": sector_size_source,
        "manifest_totals": manifest.get("totals", {}),
        "missing": missing,
    }

    signature_hits: list[dict] = []
    name_density: dict[str, dict] = {}
    record_candidates: dict[str, list[dict]] = {}
    pointer_details: dict[str, dict] = {}

    for sector in sectors:
        signature_hits.extend(
            _signature_hits(sector.payload, sector.track, sector.sector)
        )

        names, upper = _name_tokens(sector.payload)
        name_count = len({n for _, n in names})
        upper_count = len({u for _, u in upper})
        density_score = name_count + upper_count
        if name_count >= 4 or upper_count >= 10:
            name_density[f"T{sector.track:02d}_S{sector.sector:02d}"] = {
                "track": sector.track,
                "sector": sector.sector,
                "name_ext_count": name_count,
                "uppercase_token_count": upper_count,
            }

        record_scores = _record_scores(sector.payload)
        if record_scores:
            record_candidates[f"T{sector.track:02d}_S{sector.sector:02d}"] = (
                record_scores
            )

        ptr_counts = _pointer_counts(sector.payload)
        entropy = _entropy(sector.payload)
        fill_ratio = sum(1 for b in sector.payload if b in (0x00, 0xFF)) / max(
            1, len(sector.payload)
        )
        ptr_counts.update(
            {
                "entropy": entropy,
                "fill00ff_ratio": fill_ratio,
            }
        )
        pointer_details[f"T{sector.track:02d}_S{sector.sector:02d}"] = ptr_counts

    name_sorted = sorted(
        name_density.values(),
        key=lambda x: -(x["name_ext_count"] + x["uppercase_token_count"]),
    )[:50]

    pointer_stats = {
        "per_sector": pointer_details,
        "top": {
            "ts_pairs": sorted(
                pointer_details.items(),
                key=lambda kv: -kv[1]["ts_pairs_nonzero_count"],
            )[:20],
            "packed_nibble": sorted(
                pointer_details.items(),
                key=lambda kv: -kv[1]["packed_nibble_count"],
            )[:20],
            "linear16": sorted(
                pointer_details.items(),
                key=lambda kv: -(
                    kv[1]["linear16_le_count"] + kv[1]["linear16_be_count"]
                ),
            )[:20],
        },
    }

    candidate_tables: list[dict] = []
    for sector in sectors:
        key = f"T{sector.track:02d}_S{sector.sector:02d}"
        name_info = name_density.get(key)
        record_info = record_candidates.get(key, [])
        pointer_info = pointer_details.get(key, {})
        name_score = 0
        if name_info:
            name_score = name_info.get("name_ext_count", 0) + name_info.get(
                "uppercase_token_count", 0
            )
        record_score = record_info[0]["score"] if record_info else 0.0
        pointer_score = (
            pointer_info.get("ts_pairs_nonzero_count", 0)
            + pointer_info.get("packed_nibble_count", 0)
            + pointer_info.get("linear16_le_count", 0)
            + pointer_info.get("linear16_be_count", 0)
        )
        score = _directory_score(name_score, pointer_score, record_score)
        if score <= 0:
            continue
        hex_preview = sector.payload[:64].hex()
        candidate_tables.append(
            {
                "track": sector.track,
                "sector": sector.sector,
                "score": score,
                "record_sizes": record_info,
                "name_density": name_info,
                "pointer_evidence": pointer_info,
                "hex_preview": hex_preview,
            }
        )

    candidate_tables.sort(key=lambda c: -c["score"])
    candidate_tables = candidate_tables[:30]

    clusters: list[dict] = []
    if candidate_tables:
        current_cluster: list[dict] = [candidate_tables[0]]
        for entry in candidate_tables[1:]:
            if abs(entry["track"] - current_cluster[-1]["track"]) <= 2:
                current_cluster.append(entry)
            else:
                clusters.append(
                    {
                        "tracks": [c["track"] for c in current_cluster],
                        "members": current_cluster,
                    }
                )
                current_cluster = [entry]
        clusters.append(
            {
                "tracks": [c["track"] for c in current_cluster],
                "members": current_cluster,
            }
        )

    return {
        "summary": summary,
        "signature_hits": signature_hits,
        "candidate_tables": {
            "dense_names": name_sorted,
            "tables": candidate_tables,
            "clusters": clusters,
        },
        "candidate_records": record_candidates,
        "pointer_stats": pointer_stats,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan reconstructed sectors for metadata"
    )
    parser.add_argument("out_dir", type=Path, help="Reconstruction output directory")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = scan_metadata(args.out_dir)
    args.out.write_text(json.dumps(result, indent=2))
    return 0


__all__ = ["scan_metadata", "main", "build_arg_parser"]
