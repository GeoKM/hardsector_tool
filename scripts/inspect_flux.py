#!/usr/bin/env python
"""
Inspect an SCP flux image and emit per-track summaries.

Example:
    python scripts/inspect_flux.py tests/ACMS80217/ACMS80217-HS32.scp --track 0 --revs 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Iterable

from hardsector_tool.fm import (
    decode_fm_bytes,
    decode_mfm_bytes,
    pll_decode_fm_bytes,
    scan_fm_sectors,
)
from hardsector_tool.hardsector import (
    FORMAT_PRESETS,
    assemble_rotation,
    best_sector_map,
    decode_hole,
    group_hard_sectors,
    decode_track_best_map,
    build_raw_image,
)
from hardsector_tool.scp import SCPImage


def ticks_to_us(ticks: int, sample_freq_hz: int) -> float:
    return ticks / sample_freq_hz * 1_000_000


def fmt_tick_span(ticks: int, sample_freq_hz: int) -> str:
    return f"{ticks_to_us(ticks, sample_freq_hz):.2f} us"


def describe_flux(flux: list[int], sample_freq_hz: int) -> str:
    if not flux:
        return "0 samples"
    return (
        f"{len(flux)} samples, "
        f"min {fmt_tick_span(min(flux), sample_freq_hz)}, "
        f"mean {fmt_tick_span(int(mean(flux)), sample_freq_hz)}, "
        f"max {fmt_tick_span(max(flux), sample_freq_hz)}"
    )


def summarize_tracks(image: SCPImage, tracks: Iterable[int], rev_limit: int) -> None:
    for track in tracks:
        data = image.read_track(track)
        if data is None:
            print(f"- Track {track}: no data (offset not present)")
            continue

        print(
            f"- Track {track}: {data.revolution_count} rev entries, "
            f"flux bytes {len(data.flux_data)}"
        )
        limit = data.revolution_count if rev_limit <= 0 else min(
            rev_limit, data.revolution_count
        )
        for rev_idx in range(limit):
            rev = data.revolutions[rev_idx]
            flux = data.decode_flux(rev_idx)
            print(
                f"    rev {rev_idx}: index {fmt_tick_span(rev.index_ticks, image.sample_freq_hz)}, "
                f"flux count {rev.flux_count}, {describe_flux(flux, image.sample_freq_hz)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an SCP flux image.")
    parser.add_argument("scp_path", type=Path, help="Path to SCP image")
    parser.add_argument(
        "--track",
        action="append",
        type=int,
        dest="tracks",
        help="Track number(s) to inspect (default: first non-empty)",
    )
    parser.add_argument(
        "--revs",
        type=int,
        default=1,
        help="Revolutions per track to decode (0 = all, default 1)",
    )
    parser.add_argument(
        "--decode-fm",
        action="store_true",
        help="Attempt FM byte decode for the first selected track/rev",
    )
    parser.add_argument(
        "--use-pll",
        action="store_true",
        help="Use PLL-based FM decode (default is heuristic).",
    )
    parser.add_argument(
        "--scan-sectors",
        action="store_true",
        help="Scan decoded bytes for FM sector address marks.",
    )
    parser.add_argument(
        "--hard-sector-summary",
        action="store_true",
        help="Summarize hard-sector groupings (assumes 32 holes + index).",
    )
    parser.add_argument(
        "--hole",
        type=int,
        default=None,
        help="Decode a specific hole (revolution index) instead of the first track default.",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=0,
        help="Rotation index to decode in hard-sector mode (default 0).",
    )
    parser.add_argument(
        "--sector-map",
        action="store_true",
        help="Assemble a best-effort sector map across all rotations (assumes 16x256).",
    )
    parser.add_argument(
        "--encoding",
        choices=["fm", "mfm"],
        default="fm",
        help="Encoding to decode when scanning sectors.",
    )
    parser.add_argument(
        "--expected-sectors",
        type=int,
        default=16,
        help="Expected sectors per track when building sector map.",
    )
    parser.add_argument(
        "--sector-size",
        type=int,
        default=256,
        help="Expected sector size in bytes.",
    )
    parser.add_argument(
        "--preset",
        choices=["auto"] + sorted(FORMAT_PRESETS.keys()),
        default="auto",
        help="Use a predefined format preset (overrides encoding/sector counts).",
    )
    parser.add_argument(
        "--calibrate-rotation",
        action="store_true",
        help="Calibrate PLL clock from the first hole in each rotation.",
    )
    parser.add_argument(
        "--write-sectors",
        type=Path,
        default=None,
        help="Directory to write decoded sectors from the best map (implies --sector-map).",
    )
    parser.add_argument(
        "--image-out",
        type=Path,
        default=None,
        help="Write a flat image assembled from best sector maps across tracks.",
    )
    parser.add_argument(
        "--require-sync",
        action="store_true",
        help="Require an 0xA1 sync byte before IDAM when scanning sectors.",
    )
    args = parser.parse_args()

    if args.preset != "auto":
        preset = FORMAT_PRESETS[args.preset]
        args.expected_sectors = preset["expected_sectors"]
        args.sector_size = preset["sector_size"]
        args.encoding = preset["encoding"]

    image = SCPImage.from_file(args.scp_path)
    hdr = image.header
    present_tracks = hdr.non_empty_tracks
    default_tracks = [present_tracks[0]] if present_tracks else []
    tracks = args.tracks or default_tracks

    print(
        f"File: {args.scp_path}\n"
        f" Tracks: {hdr.start_track}-{hdr.end_track} "
        f"({len(present_tracks)} with data, sides={hdr.sides})\n"
        f" Revolutions recorded: {hdr.revolutions}\n"
        f" Cell width: {hdr.cell_width} (flags=0x{hdr.flags:02x})\n"
        f" Sample freq: {image.sample_freq_hz} Hz\n"
        f" Non-empty tracks: {', '.join(str(t) for t in present_tracks[:10])}"
        f"{'...' if len(present_tracks) > 10 else ''}"
    )

    if not tracks:
        print("No tracks found.")
        return

    print("\nTrack details:")
    summarize_tracks(image, tracks, args.revs)

    if args.hard_sector_summary and tracks:
        track = image.read_track(tracks[0])
        if track is None:
            print("\nHard-sector summary skipped: no data on track")
        else:
            grouping = group_hard_sectors(track, sectors_per_rotation=32)
            print(
                f"\nHard-sector summary (track {tracks[0]}): "
                f"{grouping.rotations} rotations, "
                f"{grouping.sectors_per_rotation}+{grouping.index_holes_per_rotation} holes each"
            )
            avg_tick = sum(r.index_ticks for r in track.revolutions) / track.revolution_count
            print(f" Avg index ticks per hole: {avg_tick:.1f}")
            print(" First rotation hole counts:")
            for hole in grouping.groups[0][:5]:
                print(
                    f"  hole {hole.hole_index:02d}: rev {hole.revolution_index} "
                    f"ticks={hole.index_ticks} flux_count={hole.flux_count}"
                )

    if args.decode_fm and tracks:
        target_track = tracks[0]
        track = image.read_track(target_track)
        if track is None:
            print("\nFM decode skipped: no data on track")
            return
        rev_index = args.hole if args.hole is not None else 0
        if rev_index >= track.revolution_count:
            print(f"\nFM decode skipped: hole {rev_index} beyond available revolutions")
            return
        flux = track.decode_flux(rev_index)
        if args.encoding == "mfm":
            result = decode_mfm_bytes(
                flux, sample_freq_hz=image.sample_freq_hz, index_ticks=track.revolutions[0].index_ticks
            )
            meta = f"MFM via PLL (bit shift {result.bit_shift})"
        elif args.use_pll:
            result = pll_decode_fm_bytes(
                flux, sample_freq_hz=image.sample_freq_hz, index_ticks=track.revolutions[0].index_ticks
            )
            meta = f"pll clock ~{result.initial_clock_ticks:.1f} ticks"
        else:
            result = decode_fm_bytes(flux)
            meta = (
                f"half-cell ~{result.half_cell_ticks:.1f} ticks, "
                f"full-cell ~{result.full_cell_ticks:.1f} ticks, "
                f"threshold {result.threshold_ticks:.1f}"
            )

        preview = result.bytes_out[:64]
        hex_preview = " ".join(f"{b:02x}" for b in preview)
        print(
            "\nFM decode (track {t}, rev 0, {method}):\n"
            " {meta}, bit shift {shift}\n"
            " first 64 bytes: {preview}".format(
                t=target_track,
                method=getattr(result, "method", ""),  # type: ignore[arg-type]
                meta=meta,
                shift=result.bit_shift,
                preview=hex_preview,
            )
        )
        if args.scan_sectors:
            guesses = scan_fm_sectors(result.bytes_out, require_sync=args.require_sync)
            if not guesses:
                print(" No FM sectors detected")
            else:
                print(" Detected FM sectors (IDAM guesses):")
                for g in guesses[:16]:
                    print(
                        f"  off {g.offset:06d}: C/H/S={g.track}/{g.head}/{g.sector_id} "
                        f"size={g.length} crc_ok={g.crc_ok} id_crc={g.id_crc_ok} data_crc={g.data_crc_ok}"
                    )

    if args.hard_sector_summary and args.scan_sectors and tracks:
        track = image.read_track(tracks[0])
        if track:
            grouping = group_hard_sectors(track, sectors_per_rotation=32)
            rotation = min(args.rotation, grouping.rotations - 1)
            guesses = assemble_rotation(
                image,
                track,
                grouping,
                rotation_index=rotation,
                use_pll=args.use_pll,
                require_sync=args.require_sync,
                encoding=args.encoding,
                calibrate_rotation=args.calibrate_rotation,
            )
            if guesses:
                print(f"\nRotation {rotation} sector guesses (first 16):")
                for g in guesses[:16]:
                    print(
                        f"  hole? off={g.offset:06d} C/H/S={g.track}/{g.head}/{g.sector_id} "
                        f"size={g.length} crc_ok={g.crc_ok}"
                    )
            if args.sector_map or args.write_sectors:
                all_rotations = [
                    assemble_rotation(
                        image,
                        track,
                        grouping,
                        r,
                        use_pll=args.use_pll,
                        require_sync=args.require_sync,
                        encoding=args.encoding,
                        calibrate_rotation=args.calibrate_rotation,
                    )
                    for r in range(grouping.rotations)
                ]
                best = best_sector_map(
                    all_rotations,
                    expected_track=tracks[0],
                    expected_head=0,
                    expected_sector_count=args.expected_sectors,
                    expected_size=args.sector_size,
                )
                print("\nBest-effort sector map (first 16 IDs):")
                for sid in sorted(best.keys()):
                    g = best[sid]
                    status = "ok" if g.crc_ok else "crc?"
                    print(
                        f"  sector {sid:02d}: C/H/S={g.track}/{g.head}/{g.sector_id} "
                        f"len={g.length} status={status}"
                    )
                if args.write_sectors:
                    outdir = args.write_sectors
                    outdir.mkdir(parents=True, exist_ok=True)
                    print(f"\nWriting sectors to {outdir}")
                    for sid, g in best.items():
                        if g.data is None:
                            continue
                        fname = outdir / f"track{tracks[0]:02d}_head0_sector{sid:02d}.bin"
                        fname.write_bytes(g.data)
                        print(f"  wrote {fname} ({len(g.data)} bytes)")

    # Whole-disk pass: assemble best sector maps across all present tracks.
    if args.image_out:
        track_maps = {}
        for t in present_tracks:
            best_map = decode_track_best_map(
                image,
                t,
                sectors_per_rotation=32,
                expected_sectors=args.expected_sectors,
                expected_size=args.sector_size,
                encoding=args.encoding,
                use_pll=args.use_pll,
                require_sync=args.require_sync,
                calibrate_rotation=args.calibrate_rotation,
            )
            track_maps[t] = best_map
        raw = build_raw_image(
            track_maps,
            track_order=list(present_tracks),
            expected_sectors=args.expected_sectors,
            expected_size=args.sector_size,
            fill_byte=0x00,
        )
        args.image_out.parent.mkdir(parents=True, exist_ok=True)
        args.image_out.write_bytes(raw)
        print(f"\nWrote assembled image to {args.image_out} ({len(raw)} bytes)")


if __name__ == "__main__":
    main()
