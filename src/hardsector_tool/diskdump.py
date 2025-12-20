"""Disk reconstruction and reverse-engineering helpers."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .scp import SCPImage
from .wang import SectorReconstruction, WangSector, dominant_prefix, reconstruct_track

PRINTABLE_ASCII = re.compile(rb"[\x20-\x7e]{6,}")


@dataclass
class ReconstructionResult:
    output_dir: Path
    manifest: dict


def _parse_track_range(spec: str) -> list[int]:
    if "-" in spec:
        start_s, end_s = spec.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise ValueError("track range end must be >= start")
        return list(range(start, end + 1))
    return [int(spec)]


def _parse_sector_sizes(spec: str | None) -> Sequence[int] | None:
    if spec is None or spec.lower() == "auto":
        return None
    sizes: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        sizes.append(int(part))
    return sizes or None


def _safe_mkdir(path: Path, force: bool = False) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Output directory {path} already exists; use --force to overwrite"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _sector_prefix_counts(sectors: dict[int, WangSector]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sector in sectors.values():
        for alg in sector.checksum_algorithms:
            prefix = alg.split(":", 1)[0]
            counts[prefix] += 1
    return counts


def _track_selection_stats(
    recon: dict[int, SectorReconstruction],
) -> tuple[int | None, int | None]:
    if not recon:
        return None, None
    sizes = Counter(r.sector_size for r in recon.values())
    hole_shifts = Counter(r.hole_shift for r in recon.values())
    size = sizes.most_common(1)[0][0] if sizes else None
    shift = hole_shifts.most_common(1)[0][0] if hole_shifts else None
    return size, shift


def _ascii_strings(payload: bytes) -> list[str]:
    return [
        match.decode("ascii", errors="ignore")
        for match in PRINTABLE_ASCII.findall(payload)
    ]


def _periodicity_scores(
    payload: bytes, offsets: Iterable[int] = (8, 16, 32, 64)
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for offset in offsets:
        if len(payload) <= offset:
            continue
        matches = sum(
            1 for i in range(len(payload) - offset) if payload[i] == payload[i + offset]
        )
        scores[offset] = matches / (len(payload) - offset)
    return scores


def _pointer_mappings(
    value: int, logical_sectors: int, track_numbers: Sequence[int]
) -> list[tuple[str, int, int]]:
    mappings: list[tuple[str, int, int]] = []
    if not track_numbers:
        return mappings
    track_set = set(track_numbers)
    total_sectors = logical_sectors * len(track_numbers)

    if 0 <= value < total_sectors:
        track = value // logical_sectors
        sector = value % logical_sectors
        if track in track_set:
            mappings.append(("linear", track, sector))

    track_val = value >> 4
    sector_val = value & 0x0F
    if sector_val < logical_sectors and track_val in track_set:
        mappings.append(("packed", track_val, sector_val))
    return mappings


def _write_track_metadata(
    path: Path,
    track_number: int,
    physical_track: int,
    track_score: float,
    sector_size: int | None,
    hole_shift: int | None,
    prefix_counts: Counter[str],
    sectors: dict[int, WangSector],
    recon: dict[int, SectorReconstruction],
) -> None:
    dom_prefix = dominant_prefix(sectors, min_count=1)
    dom_count = prefix_counts.get(dom_prefix or "", 0)
    data = {
        "track_number": track_number,
        "physical_track": physical_track,
        "track_score": track_score,
        "sector_size": sector_size,
        "hole_shift": hole_shift,
        "dominant_prefix": dom_prefix,
        "dominant_prefix_match_count": dom_count,
        "sectors": [],
    }
    for sid, sector in sorted(sectors.items()):
        entry: dict = {
            "sector_id": sid,
            "payload_len": len(sector.payload),
            "checksum_algorithms": list(sector.checksum_algorithms),
        }
        rec = recon.get(sid)
        if rec:
            entry.update(
                {
                    "window_score": rec.window_score,
                    "window_similarity": rec.window_similarity,
                    "payload_gap": rec.payload_gap,
                    "payload_entropy": rec.payload_entropy,
                    "payload_fill_ratio": rec.payload_fill_ratio,
                    "payload_offset": rec.payload_offset,
                    "prefix_rescue_applied": rec.prefix_rescue_applied,
                    "rescue_from_prefixes": rec.rescue_from_prefixes,
                    "rescue_to_prefixes": rec.rescue_to_prefixes,
                    "rescue_reason": rec.rescue_reason,
                }
            )
        data["sectors"].append(entry)
    path.write_text(json.dumps(data, indent=2))


class DiskReconstructor:
    """High-level coordinator for disk reconstruction runs."""

    def __init__(
        self,
        image_path: Path,
        output_dir: Path,
        *,
        tracks: Sequence[int],
        side: int = 0,
        logical_sectors: int = 16,
        sectors_per_rotation: int = 32,
        sector_sizes: Sequence[int] | None = None,
        keep_best: int = 3,
        similarity_threshold: float = 0.80,
        clock_factor: float = 1.0,
        dump_raw_windows: bool = False,
        write_manifest: bool = True,
        write_report: bool = True,
        force: bool = False,
    ) -> None:
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.tracks = list(tracks)
        self.side = side
        self.logical_sectors = logical_sectors
        self.sectors_per_rotation = sectors_per_rotation
        self.sector_sizes = sector_sizes
        self.keep_best = keep_best
        self.similarity_threshold = similarity_threshold
        self.clock_factor = clock_factor
        self.dump_raw_windows = dump_raw_windows
        self.write_manifest = write_manifest
        self.write_report = write_report
        self.force = force

    def run(self) -> ReconstructionResult:
        _safe_mkdir(self.output_dir, force=self.force)
        sectors_dir = self.output_dir / "sectors"
        tracks_dir = self.output_dir / "tracks"
        sectors_dir.mkdir(parents=True, exist_ok=True)
        tracks_dir.mkdir(parents=True, exist_ok=True)

        image = SCPImage.from_file(self.image_path)

        manifest: dict = {
            "image": {
                "path": str(self.image_path),
                "version": image.header.version,
                "sample_freq_hz": getattr(image, "sample_freq_hz", None),
                "disk_type": image.header.disk_type,
                "revolutions": image.header.revolutions,
                "capture_resolution": image.header.capture_resolution,
                "heads": image.header.heads,
            },
            "tracks": [],
            "totals": {
                "expected_sectors": len(self.tracks) * self.logical_sectors,
                "written_sectors": 0,
                "missing_sectors": 0,
            },
        }

        ascii_hits: Counter[str] = Counter()
        ascii_locations: dict[str, set[str]] = defaultdict(set)
        periodicity: dict[int, dict[int, float]] = defaultdict(dict)
        pointer_hits: Counter[tuple[str, int, int]] = Counter()

        dump_base = self.output_dir / "raw_windows" if self.dump_raw_windows else None

        for track_number in self.tracks:
            physical_track = track_number * image.header.sides + self.side
            sectors, recon, track_score = reconstruct_track(
                image,
                track_number=physical_track,
                logical_sectors=self.logical_sectors,
                sectors_per_rotation=self.sectors_per_rotation,
                sector_sizes=self.sector_sizes,
                keep_best=self.keep_best,
                similarity_threshold=self.similarity_threshold,
                clock_factor=self.clock_factor,
                dump_raw=dump_base,
            )
            prefix_counts = _sector_prefix_counts(sectors)
            sector_size, hole_shift = _track_selection_stats(recon)

            for sid, sector in sorted(sectors.items()):
                sector_path = sectors_dir / f"T{track_number:02d}_S{sid:02d}.bin"
                sector_path.write_bytes(sector.payload)
                for string in _ascii_strings(sector.payload):
                    ascii_hits[string] += 1
                    ascii_locations[string].add(f"T{track_number:02d}S{sid:02d}")
                per_scores = _periodicity_scores(sector.payload)
                if per_scores:
                    periodicity[track_number] = {
                        k: max(
                            per_scores.get(k, 0.0),
                            periodicity[track_number].get(k, 0.0),
                        )
                        for k in set(per_scores) | set(periodicity[track_number])
                    }
                for i in range(len(sector.payload) - 1):
                    window = sector.payload[i : i + 2]
                    val_le = int.from_bytes(window, "little")
                    val_be = int.from_bytes(window, "big")
                    for mapping in _pointer_mappings(
                        val_le, self.logical_sectors, self.tracks
                    ):
                        pointer_hits[(mapping[0], mapping[1], mapping[2])] += 1
                    for mapping in _pointer_mappings(
                        val_be, self.logical_sectors, self.tracks
                    ):
                        pointer_hits[(mapping[0], mapping[1], mapping[2])] += 1

            _write_track_metadata(
                tracks_dir / f"T{track_number:02d}.json",
                track_number,
                physical_track,
                track_score,
                sector_size,
                hole_shift,
                prefix_counts,
                sectors,
                recon,
            )

            dom = dominant_prefix(sectors, min_count=1)
            manifest["tracks"].append(
                {
                    "track_number": track_number,
                    "physical_track": physical_track,
                    "recovered_sectors": len(sectors),
                    "sector_size": sector_size,
                    "track_score": track_score,
                    "dominant_prefix": dom,
                    "dominant_prefix_match_count": _sector_prefix_counts(sectors).get(
                        dom or "", 0
                    ),
                }
            )
            manifest["totals"]["written_sectors"] += len(sectors)

        manifest["totals"]["missing_sectors"] = (
            manifest["totals"]["expected_sectors"]
            - manifest["totals"]["written_sectors"]
        )

        if self.write_manifest:
            (self.output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )

        if self.write_report:
            report_path = self.output_dir / "report.txt"
            report_path.write_text(
                self._render_report(
                    ascii_hits,
                    ascii_locations,
                    periodicity,
                    pointer_hits,
                )
            )

        return ReconstructionResult(output_dir=self.output_dir, manifest=manifest)

    def _render_report(
        self,
        ascii_hits: Counter[str],
        ascii_locations: dict[str, set[str]],
        periodicity: dict[int, dict[int, float]],
        pointer_hits: Counter[tuple[str, int, int]],
    ) -> str:
        lines: list[str] = []
        lines.append("== Reverse-engineering hints ==")
        lines.append("")

        lines.append("ASCII strings (runs >=6 chars, top 200):")
        for string, count in ascii_hits.most_common(200):
            locs = sorted(ascii_locations[string])
            lines.append(f"  [{count:4d}] {string} :: {', '.join(locs)}")
        if not ascii_hits:
            lines.append("  (no printable runs found)")
        lines.append("")

        lines.append("Record cadence hints (byte periodicity matches):")
        if periodicity:
            for track in sorted(periodicity):
                per = periodicity[track]
                strong = [(k, v) for k, v in sorted(per.items()) if v >= 0.25]
                if not strong:
                    continue
                peaks = ", ".join(f"{k} bytes -> {v:.2f}" for k, v in strong)
                lines.append(f"  Track {track:02d}: {peaks}")
        else:
            lines.append("  (no sectors decoded)")
        lines.append("")

        lines.append("Pointer plausibility (repeated 2-byte targets):")
        strong_hits = [(k, v) for k, v in pointer_hits.items() if v >= 5]
        if strong_hits:
            for (scheme, track, sector), count in sorted(
                strong_hits, key=lambda kv: (-kv[1], kv[0])
            ):
                lines.append(
                    f"  {scheme} -> track {track:02d} sector {sector:02d} seen {count} times"
                )
        else:
            lines.append("  (no repeated in-range pointers found)")

        return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hardsector_tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recon = subparsers.add_parser(
        "reconstruct-disk",
        help="Reconstruct logical sectors from a hard-sectored SCP image",
    )
    recon.add_argument("image", type=Path, help="Input SCP image")
    recon.add_argument("--out", required=True, type=Path, help="Output directory")
    recon.add_argument("--tracks", default="0-76", help="Track range, e.g. 0-76")
    recon.add_argument("--side", type=int, default=0, help="Head/side index to decode")
    recon.add_argument(
        "--logical-sectors",
        type=int,
        default=16,
        help="Logical sectors per track",
    )
    recon.add_argument(
        "--sectors-per-rotation",
        type=int,
        default=32,
        help="Hard-sector holes per rotation",
    )
    recon.add_argument(
        "--sector-sizes",
        default="auto",
        help="Comma-separated sector sizes to try (auto to guess)",
    )
    recon.add_argument(
        "--keep-best",
        type=int,
        default=3,
        help="Number of rotations to keep when reconstructing",
    )
    recon.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.80,
        help="Similarity threshold for rotation consensus",
    )
    recon.add_argument(
        "--clock-factor",
        type=float,
        default=1.0,
        help="Clock factor passed to Wang reconstruction",
    )
    recon.add_argument(
        "--dump-raw-windows",
        action="store_true",
        help="Dump decoded window bytes for debugging",
    )
    recon.add_argument(
        "--no-json",
        dest="write_json",
        action="store_false",
        help="Skip manifest.json",
    )
    recon.add_argument(
        "--no-report",
        dest="write_report",
        action="store_false",
        help="Skip report.txt",
    )
    recon.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing output directory",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "reconstruct-disk":
        tracks = _parse_track_range(args.tracks)
        sector_sizes = _parse_sector_sizes(args.sector_sizes)
        recon = DiskReconstructor(
            image_path=args.image,
            output_dir=args.out,
            tracks=tracks,
            side=args.side,
            logical_sectors=args.logical_sectors,
            sectors_per_rotation=args.sectors_per_rotation,
            sector_sizes=sector_sizes,
            keep_best=args.keep_best,
            similarity_threshold=args.similarity_threshold,
            clock_factor=args.clock_factor,
            dump_raw_windows=args.dump_raw_windows,
            write_manifest=args.write_json,
            write_report=args.write_report,
            force=args.force,
        )
        recon.run()
        return 0

    parser.error(f"Unknown command {args.command}")
    return 1


__all__ = ["DiskReconstructor", "ReconstructionResult", "main", "build_arg_parser"]
