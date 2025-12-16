#!/usr/bin/env python
"""
Inspect an SCP flux image and emit per-track summaries.

Example:
    python scripts/inspect_flux.py tests/ACMS80217/ACMS80217-HS32.scp --track 0 --revs 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from hardsector_tool.fm import (
    bits_to_bytes,
    brute_force_mark_payloads,
    decode_fm_bytes,
    decode_mfm_bytes,
    fm_bytes_from_bitcells,
    mfm_bytes_from_bitcells,
    pll_decode_bits,
    pll_decode_fm_bytes,
    scan_bit_patterns,
    scan_data_marks,
    scan_fm_sectors,
)
from hardsector_tool.hardsector import (
    FORMAT_PRESETS,
    HoleCapture,
    assemble_rotation,
    best_payload_windows,
    best_sector_map,
    build_raw_image,
    compute_flux_index_diagnostics,
    decode_hole_bytes,
    decode_track_best_map,
    group_hard_sectors,
    payload_metrics,
    stitch_rotation_flux,
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
        limit = (
            data.revolution_count
            if rev_limit <= 0
            else min(rev_limit, data.revolution_count)
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
        "--flux-deltas",
        action="store_true",
        help="Report index_ticks minus summed flux per hole for the first selected track.",
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
        "--physical-sectors",
        type=int,
        default=32,
        help="Physical hard-sector holes per rotation (post index-merge).",
    )
    parser.add_argument(
        "--logical-sectors",
        type=int,
        default=16,
        help="Logical sectors when pairing holes (e.g., 16 for 256-byte logical on HS32).",
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
        "--clock-adjust",
        type=float,
        default=0.10,
        help="PLL clock adjustment range (fraction, default 0.10).",
    )
    parser.add_argument(
        "--clock-scale",
        type=float,
        default=1.0,
        help="Scale estimated clock ticks (e.g., 0.5 or 2.0) before PLL.",
    )
    parser.add_argument(
        "--synthetic-from-holes",
        action="store_true",
        help="If no IDAMs are found, synthesize sectors from hole payloads (mod expected sector count).",
    )
    parser.add_argument(
        "--dump-holes",
        type=Path,
        default=None,
        help="Directory to dump raw decoded bytes per hole (first rotation of each selected track).",
    )
    parser.add_argument(
        "--dump-bitcells",
        type=Path,
        default=None,
        help="Directory to dump PLL bitcells per hole (first rotation of each selected track).",
    )
    parser.add_argument(
        "--scan-marks",
        action="store_true",
        help="Scan decoded bytes for data marks (0xFB/0xFA) without CRC.",
    )
    parser.add_argument(
        "--scan-bit-patterns",
        action="store_true",
        help="Scan bitcell streams for mark bytes across bit shifts (0xFB/0xFA/0xA1/0xFE).",
    )
    parser.add_argument(
        "--bruteforce-marks",
        action="store_true",
        help="Extract payload windows following mark hits in bitcell streams.",
    )
    parser.add_argument(
        "--mark-payload-bytes",
        type=int,
        default=256,
        help="Bytes to capture after each mark when bruteforcing (default 256).",
    )
    parser.add_argument(
        "--mark-payload-dir",
        type=Path,
        default=None,
        help="Optional directory to dump bruteforced mark payloads (implies --bruteforce-marks).",
    )
    parser.add_argument(
        "--invert-bytes",
        action="store_true",
        help="Invert decoded bytes (bitwise NOT) when dumping holes and scanning marks.",
    )
    parser.add_argument(
        "--invert-bitcells",
        action="store_true",
        help="Invert PLL bitcells before byte conversion (useful if polarity is reversed).",
    )
    parser.add_argument(
        "--invert-flux",
        action="store_true",
        help="Invert flux intervals prior to PLL (experimental polarity check).",
    )
    parser.add_argument(
        "--merge-hole-pairs",
        action="store_true",
        help="Concatenate consecutive hole bitstreams (0+1, 2+3, ...) before scanning/dumping.",
    )
    parser.add_argument(
        "--pair-holes",
        action="store_true",
        help="Alias for --merge-hole-pairs (pairs holes after index-split merge).",
    )
    parser.add_argument(
        "--fixed-spacing-scan",
        action="store_true",
        help="When merging hole pairs, also slice fixed-size windows (mark-free) for inspection.",
    )
    parser.add_argument(
        "--fixed-spacing-bytes",
        type=int,
        default=258,
        help="Window size for fixed-spacing-scan (default 258 for 256+CRC).",
    )
    parser.add_argument(
        "--fixed-spacing-step",
        type=int,
        default=2048,
        help="Bit step between fixed windows when scanning merged hole pairs (approx FM sector length in bits).",
    )
    parser.add_argument(
        "--report-entropy",
        action="store_true",
        help="Report top payload windows by lowest fill-ratio/highest entropy (mark and fixed-spacing payloads).",
    )
    parser.add_argument(
        "--payload-windows",
        type=int,
        default=1,
        help="Top-N payload windows to keep per hole when extracting raw payloads (default 1).",
    )
    parser.add_argument(
        "--strict-marks",
        action="store_true",
        help="Limit mark scans to FE/FB only (reduces false positives).",
    )
    parser.add_argument(
        "--bruteforce-step-bits",
        type=int,
        default=2048,
        help="When bruteforcing mark payloads without hits, try windows every N bits (default 2048).",
    )
    parser.add_argument(
        "--stitch-rotation",
        action="store_true",
        help="Also decode a stitched full-rotation bitstream before scanning/dumping.",
    )
    parser.add_argument(
        "--stitch-gap-comp",
        action="store_true",
        help="When stitching, insert a no-transition gap equal to index-flux delta per hole.",
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
        "--track-range",
        type=str,
        default=None,
        help="Track range to process for image-out (e.g., 0-20,40,42). Defaults to all present tracks.",
    )
    parser.add_argument(
        "--require-sync",
        action="store_true",
        help="Require an 0xA1 sync byte before IDAM when scanning sectors.",
    )
    parser.add_argument(
        "--score-grid",
        action="store_true",
        help="Score FM/MFM candidates across clock scales/polarity and report top marks/entropy.",
    )
    parser.add_argument(
        "--score-window",
        type=int,
        default=4096,
        help="Bytes to consider when scoring grid candidates (default 4096).",
    )
    parser.add_argument(
        "--show-hole-timing",
        action="store_true",
        help="Print normalized hole timings and highlight short index-split pair.",
    )
    parser.add_argument(
        "--auto-invert",
        action="store_true",
        help="Try both bitcell polarities and pick the higher-scoring candidate.",
    )
    parser.add_argument(
        "--auto-clock-scale",
        action="store_true",
        help="Sweep common clock scales and pick the best-scoring candidate.",
    )
    args = parser.parse_args()
    if args.mark_payload_dir:
        args.bruteforce_marks = True

    if args.preset != "auto":
        preset = FORMAT_PRESETS[args.preset]
        args.expected_sectors = preset["expected_sectors"]
        args.sector_size = preset["sector_size"]
        args.encoding = preset["encoding"]

    args.pair_holes = args.pair_holes or args.merge_hole_pairs
    if args.pair_holes:
        args.expected_sectors = args.logical_sectors

    image = SCPImage.from_file(args.scp_path)
    hdr = image.header
    present_tracks = hdr.non_empty_tracks

    def expand_track_range(range_arg: str | None) -> list[int]:
        selected: list[int] = []
        if not range_arg:
            return selected
        for part in range_arg.split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token:
                start, end = token.split("-", 1)
                selected.extend(range(int(start), int(end) + 1))
            else:
                selected.append(int(token))
        return selected

    def tracks_for_bulk() -> list[int]:
        if args.track_range:
            chosen = expand_track_range(args.track_range)
            return [t for t in chosen if t in present_tracks]
        if args.tracks:
            return [t for t in args.tracks if t in present_tracks]
        return list(present_tracks)

    top_payloads: list[tuple[float, float, int, str, bytes]] = []

    def record_payload(label: str, data: bytes) -> None:
        if not args.report_entropy:
            return
        fill_ratio, entropy = payload_metrics(data)
        top_payloads.append((fill_ratio, -entropy, len(data), label, data[:32]))
        top_payloads.sort()
        if len(top_payloads) > 12:
            del top_payloads[12:]

    default_tracks = [present_tracks[0]] if present_tracks else []
    tracks = args.tracks or default_tracks

    index_aligned_flag = bool(hdr.flags & 0x01)
    alignment_note = "index-aligned" if index_aligned_flag else "random-start"
    print(
        f"File: {args.scp_path}\n"
        f" Tracks: {hdr.start_track}-{hdr.end_track} "
        f"({len(present_tracks)} with data, sides={hdr.sides})\n"
        f" Revolutions recorded: {hdr.revolutions}\n"
        f" Cell width code: {hdr.cell_width_code} res={hdr.capture_resolution} "
        f"(flags=0x{hdr.flags:02x}, {alignment_note})\n"
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
            grouping = group_hard_sectors(
                track,
                sectors_per_rotation=args.physical_sectors,
                index_aligned=bool(hdr.flags & 0x01),
            )
            print(
                f"\nHard-sector summary (track {tracks[0]}): "
                f"{grouping.rotations} rotations, "
                f"{len(grouping.groups[0])} merged intervals (raw {grouping.sectors_per_rotation + 1}) "
                f"(rotated_by={grouping.rotated_by}, index_conf={grouping.index_confidence:.2f})"
            )
            if grouping.short_pair_positions:
                short = grouping.short_pair_positions[0]
                if short is not None:
                    print(
                        f" Index-split short pair starts at capture {short} "
                        f"(holes {short} & {(short + 1) % (args.physical_sectors + 1)})"
                    )
            if args.show_hole_timing and grouping.groups:
                durations = [h.index_ticks for h in grouping.groups[0]]
                med = median(durations) if durations else 0
                labels = []
                short_pair = grouping.short_pair_positions[0]
                for idx, ticks in enumerate(durations):
                    marker = ""
                    if short_pair is not None and idx in {
                        short_pair,
                        (short_pair + 1) % len(durations),
                    }:
                        marker = "*"
                    labels.append(f"{idx:02d}:{ticks}{marker}")
                print(
                    " Normalized hole durations (rot0): med="
                    f"{med:.1f} ticks -> {' '.join(labels)}"
                )
            avg_tick = (
                sum(r.index_ticks for r in track.revolutions) / track.revolution_count
            )
            print(f" Avg index ticks per hole: {avg_tick:.1f}")
            print(" First rotation hole counts:")
            for hole in grouping.groups[0][:5]:
                print(
                    f"  hole {hole.hole_index:02d}: rev {hole.revolution_index} "
                    f"ticks={hole.index_ticks} flux_count={hole.flux_count}"
                )
            if args.flux_deltas:
                diagnostics = compute_flux_index_diagnostics(track, grouping)
                if diagnostics:
                    deltas = [d["delta"] for d in diagnostics]
                    ratios = [d["ratio"] for d in diagnostics if d["index_ticks"]]
                    ratio_str = (
                        f" ratio range {min(ratios):.3f}-{max(ratios):.3f}"
                        if ratios
                        else ""
                    )
                    print(
                        f" Flux-index deltas (index_ticks - sum(flux)) rotation 0: "
                        f"min {min(deltas)} max {max(deltas)} worst_abs {max(deltas, key=abs)}{ratio_str}"
                    )
                    for diag in diagnostics:
                        revs = ",".join(str(r) for r in diag["revolution_indices"])
                        print(
                            f"  hole {diag['hole_index']:02d} revs=[{revs}] "
                            f"flux={diag['flux_total']} idx={diag['index_ticks']} "
                            f"delta={diag['delta']} ratio={diag['ratio']:.3f}"
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
                flux,
                sample_freq_hz=image.sample_freq_hz,
                index_ticks=track.revolutions[rev_index].index_ticks,
            )
            meta = f"MFM via PLL (bit shift {result.bit_shift})"
        elif args.use_pll:
            result = pll_decode_fm_bytes(
                flux,
                sample_freq_hz=image.sample_freq_hz,
                index_ticks=track.revolutions[rev_index].index_ticks,
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
        if args.scan_marks:
            payload = (
                bytes(~b & 0xFF for b in result.bytes_out)
                if args.invert_bytes
                else result.bytes_out
            )
            marks = scan_data_marks(payload)
            if marks:
                print(
                    f" Data marks found at offsets: {', '.join(str(m[0]) for m in marks[:20])}"
                )

    if args.score_grid and tracks:
        track = image.read_track(tracks[0])
        if track is None:
            print("\nGrid scoring skipped: no data on track")
        else:
            grouping = group_hard_sectors(
                track,
                sectors_per_rotation=args.physical_sectors,
                index_aligned=bool(hdr.flags & 0x01),
            )
            stitched_flux, stitched_ticks = stitch_rotation_flux(
                track, grouping, rotation_index=0, compensate_gaps=args.stitch_gap_comp
            )
            base_flux = stitched_flux if stitched_flux else track.decode_flux(0)
            base_ticks = (
                stitched_ticks if stitched_ticks else track.revolutions[0].index_ticks
            )
            candidates = []
            mark_set = {0xFE, 0xFB, 0xFA, 0xA1}
            for encoding in ("fm", "mfm"):
                for invert_bits in (False, True):
                    for scale in (0.5, 0.75, 1.0, 1.25, 2.0):
                        flux = list(base_flux)
                        if args.invert_flux:
                            flux = list(reversed(flux))
                        if scale != 1.0:
                            flux = [max(1, int(x * scale)) for x in flux]
                        bitcells = pll_decode_bits(
                            flux,
                            sample_freq_hz=image.sample_freq_hz,
                            index_ticks=base_ticks,
                            clock_adjust=args.clock_adjust,
                            initial_clock_ticks=None,
                            invert=invert_bits,
                        )
                        if encoding == "fm":
                            _, candidate_bytes = fm_bytes_from_bitcells(bitcells)
                        else:
                            _, candidate_bytes = mfm_bytes_from_bitcells(bitcells)
                        window = candidate_bytes[: args.score_window]
                        mark_count = sum(1 for b in window if b in mark_set)
                        fill_ratio, entropy = payload_metrics(window)
                        crc_hits = 0
                        if encoding == "fm":
                            crc_hits = sum(
                                1
                                for g in scan_fm_sectors(
                                    candidate_bytes, require_sync=False
                                )
                                if g.crc_ok
                            )
                        candidates.append(
                            (
                                -mark_count,
                                fill_ratio,
                                -entropy,
                                -crc_hits,
                                encoding,
                                invert_bits,
                                scale,
                                mark_count,
                                entropy,
                                crc_hits,
                            )
                        )
            candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
            print("\nGrid score candidates (top 6):")
            for entry in candidates[:6]:
                (
                    _,
                    fill_ratio,
                    neg_entropy,
                    _,
                    enc,
                    inv_bits,
                    scale,
                    mark_count,
                    entropy,
                    crc_hits,
                ) = entry
                print(
                    f"  {enc} scale={scale} invert_bits={inv_bits}: "
                    f"marks={mark_count} crc_hits={crc_hits} fill={fill_ratio:.3f} entropy={entropy:.2f}"
                )

    if args.hard_sector_summary and args.scan_sectors and tracks:
        track = image.read_track(tracks[0])
        if track:
            grouping = group_hard_sectors(
                track,
                sectors_per_rotation=args.physical_sectors,
                index_aligned=bool(hdr.flags & 0x01),
            )
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
                synthetic_from_hole=args.synthetic_from_holes,
                expected_sectors=args.expected_sectors,
                expected_size=args.sector_size,
                clock_adjust=args.clock_adjust,
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
                        synthetic_from_hole=args.synthetic_from_holes,
                        expected_sectors=args.expected_sectors,
                        expected_size=args.sector_size,
                        clock_adjust=args.clock_adjust,
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
                        fname = (
                            outdir / f"track{tracks[0]:02d}_head0_sector{sid:02d}.bin"
                        )
                        fname.write_bytes(g.data)
                        print(f"  wrote {fname} ({len(g.data)} bytes)")

    bulk_tracks = tracks_for_bulk()

    # Whole-disk pass: assemble best sector maps across selected tracks.
    if args.image_out:
        if not bulk_tracks:
            print("\nNo tracks selected for image output.")
        else:
            track_maps = {}
            for t in bulk_tracks:
                best_map = decode_track_best_map(
                    image,
                    t,
                    sectors_per_rotation=args.physical_sectors,
                    expected_sectors=args.expected_sectors,
                    expected_size=args.sector_size,
                    encoding=args.encoding,
                    use_pll=args.use_pll,
                    require_sync=args.require_sync,
                    calibrate_rotation=args.calibrate_rotation,
                    synthetic_from_holes=args.synthetic_from_holes,
                    clock_adjust=args.clock_adjust,
                )
                track_maps[t] = best_map
            raw = build_raw_image(
                track_maps,
                track_order=bulk_tracks,
                expected_sectors=args.expected_sectors,
                expected_size=args.sector_size,
                fill_byte=0x00,
            )
            args.image_out.parent.mkdir(parents=True, exist_ok=True)
            args.image_out.write_bytes(raw)
            print(f"\nWrote assembled image to {args.image_out} ({len(raw)} bytes)")

    if args.dump_holes:
        if not bulk_tracks:
            print("\nNo tracks selected for hole dumps.")
        else:
            outdir = args.dump_holes
            outdir.mkdir(parents=True, exist_ok=True)
            for t in bulk_tracks:
                track = image.read_track(t)
                if not track:
                    continue
                grouping = group_hard_sectors(
                    track,
                    sectors_per_rotation=args.physical_sectors,
                    index_aligned=bool(hdr.flags & 0x01),
                )
                if not grouping.groups:
                    continue
                first_rot = grouping.groups[0]

                def iter_hole_groups():
                    if args.pair_holes:
                        limit = len(first_rot) - len(first_rot) % 2
                        for i in range(0, limit, 2):
                            h0, h1 = first_rot[i], first_rot[i + 1]
                            combined = HoleCapture(
                                hole_index=i // 2,
                                revolution_indices=h0.revolution_indices
                                + h1.revolution_indices,
                                index_ticks=h0.index_ticks + h1.index_ticks,
                                flux_count=h0.flux_count + h1.flux_count,
                            )
                            yield combined
                    else:
                        yield from first_rot

                for hole in iter_hole_groups():
                    data = decode_hole_bytes(
                        image,
                        track,
                        hole,
                        use_pll=args.use_pll,
                        encoding=args.encoding,
                        initial_clock_ticks=None,
                        clock_adjust=args.clock_adjust,
                    )
                    if args.invert_bytes:
                        data = bytes(~b & 0xFF for b in data)

                    raw_name = outdir / f"track{t:03d}_hole{hole.hole_index:02d}.bin"
                    raw_name.write_bytes(data)

                    windows = best_payload_windows(
                        data, args.sector_size, top_n=max(1, args.payload_windows)
                    )
                    if not windows:
                        fill_ratio, entropy = payload_metrics(data[: args.sector_size])
                        windows = [(0, data[: args.sector_size], fill_ratio, entropy)]
                    for win_idx, (offset, payload, fill_ratio, entropy) in enumerate(
                        windows
                    ):
                        payload_label = (
                            f"track{t:03d}_hole{hole.hole_index:02d}_"
                            f"win{win_idx}_off{offset:04d}"
                        )
                        record_payload(payload_label, payload)
                        payload_path = outdir / f"{payload_label}.bin"
                        payload_path.write_bytes(payload)
                        print(
                            f"  hole {hole.hole_index:02d} window {win_idx} "
                            f"off={offset} fill={fill_ratio:.3f} entropy={entropy:.2f}"
                        )
            print(f"\nWrote hole dumps to {outdir}")

    needs_bitcells = bool(
        args.dump_bitcells or args.scan_bit_patterns or args.bruteforce_marks
    )
    if args.fixed_spacing_scan:
        needs_bitcells = True
    if needs_bitcells:
        if not bulk_tracks:
            print("\nNo tracks selected for bitcell scans.")
        else:
            outdir = args.dump_bitcells
            payload_dir = args.mark_payload_dir
            payload_cap = 128
            if outdir:
                outdir.mkdir(parents=True, exist_ok=True)
            if payload_dir:
                payload_dir.mkdir(parents=True, exist_ok=True)
            for t in bulk_tracks:
                track = image.read_track(t)
                if not track:
                    continue
                grouping = group_hard_sectors(
                    track,
                    sectors_per_rotation=args.physical_sectors,
                    index_aligned=bool(hdr.flags & 0x01),
                )
                if not grouping.groups:
                    continue
                first_rot = grouping.groups[0]

                clock_scale = args.clock_scale
                invert_bits = args.invert_bitcells

                def score_candidate(bitcells: list[int]) -> tuple:
                    if args.encoding == "mfm":
                        _, candidate_bytes = mfm_bytes_from_bitcells(bitcells)
                    else:
                        _, candidate_bytes = fm_bytes_from_bitcells(bitcells)
                    window = candidate_bytes[: args.score_window]
                    mark_set = {0xFE, 0xFB, 0xFA, 0xA1}
                    mark_count = sum(1 for b in window if b in mark_set)
                    fill_ratio, entropy = payload_metrics(window)
                    return (-mark_count, fill_ratio, -entropy, mark_count, entropy)

                if args.auto_invert or args.auto_clock_scale:
                    stitched_flux, stitched_ticks = stitch_rotation_flux(
                        track,
                        grouping,
                        rotation_index=0,
                        compensate_gaps=args.stitch_gap_comp,
                    )
                    base_flux = stitched_flux if stitched_flux else track.decode_flux(0)
                    base_ticks = (
                        stitched_ticks
                        if stitched_ticks
                        else track.revolutions[0].index_ticks
                    )
                    best = None
                    for inv in (False, True):
                        if args.auto_invert is False and inv != args.invert_bitcells:
                            continue
                        for scale in (0.5, 0.75, 1.0, 1.25, 1.5):
                            if not args.auto_clock_scale and scale != args.clock_scale:
                                continue
                            flux_scaled = [max(1, int(x * scale)) for x in base_flux]
                            bitcells = pll_decode_bits(
                                flux_scaled,
                                sample_freq_hz=image.sample_freq_hz,
                                index_ticks=base_ticks,
                                clock_adjust=args.clock_adjust,
                                initial_clock_ticks=None,
                                invert=inv,
                            )
                            score = score_candidate(bitcells)
                            if best is None or score < best[0]:
                                best = (score, scale, inv)
                    if best:
                        _, clock_scale, invert_bits = best
                        print(
                            f" Auto-selected clock_scale={clock_scale} invert_bits={invert_bits}"
                        )

                def decode_bits_for_hole(hole: HoleCapture) -> list[int]:
                    flux: list[int] = []
                    for rev_idx in hole.revolution_indices:
                        part = track.decode_flux(rev_idx)
                        if args.invert_flux:
                            part = list(reversed(part))
                        if clock_scale != 1.0:
                            part = [max(1, int(x * clock_scale)) for x in part]
                        flux.extend(part)
                    return pll_decode_bits(
                        flux,
                        sample_freq_hz=image.sample_freq_hz,
                        index_ticks=hole.index_ticks,
                        clock_adjust=args.clock_adjust,
                        initial_clock_ticks=None,
                        invert=invert_bits,
                    )

                def iter_hole_groups():
                    if args.pair_holes:
                        limit = len(first_rot) - len(first_rot) % 2
                        for i in range(0, limit, 2):
                            h0, h1 = first_rot[i], first_rot[i + 1]
                            combined = HoleCapture(
                                hole_index=i // 2,
                                revolution_indices=h0.revolution_indices
                                + h1.revolution_indices,
                                index_ticks=h0.index_ticks + h1.index_ticks,
                                flux_count=h0.flux_count + h1.flux_count,
                            )
                            yield (
                                combined,
                                f"hole{h0.hole_index:02d}-{h1.hole_index:02d}",
                            )
                    else:
                        for hole in first_rot:
                            yield hole, f"hole{hole.hole_index:02d}"

                for hole, label in iter_hole_groups():
                    bits = decode_bits_for_hole(hole)

                    if outdir:
                        fname = outdir / f"track{t:03d}_{label}.bits"
                        fname.write_bytes(bytes(bits))
                    if args.scan_bit_patterns:
                        hits = scan_bit_patterns(bits)
                        if hits:
                            print(
                                f"Track {t} {label}: bit-pattern hits "
                                f"(shift,byte,val): {hits[:10]}"
                            )
                    if args.bruteforce_marks:
                        patterns = (
                            (0xFE, 0xFB)
                            if args.strict_marks
                            else (0xFB, 0xFA, 0xA1, 0xFE)
                        )
                        payloads = brute_force_mark_payloads(
                            bits,
                            payload_bytes=args.mark_payload_bytes,
                            patterns=patterns,
                        )
                        if not payloads and payload_dir and args.bruteforce_step_bits:
                            # Fallback: brute-force fixed windows even without mark hits.
                            step_bits = args.bruteforce_step_bits
                            window_bits = args.mark_payload_bytes * 8
                            bit_offset = 0
                            idx = 0
                            while bit_offset + window_bits <= len(bits):
                                payload = bits_to_bytes(
                                    bits[bit_offset : bit_offset + window_bits]
                                )
                                payloads.append((0, bit_offset // 8, 0x00, payload))
                                bit_offset += step_bits
                                idx += 1
                        if payloads:
                            first = payloads[0]
                            preview = " ".join(f"{b:02x}" for b in first[3][:16])
                            print(
                                f"Track {t} {label}: {len(payloads)} mark payloads; "
                                f"first (shift {first[0]}, off {first[1]}, val {first[2]:02x}) {preview}"
                            )
                            if payload_dir:
                                for _idx, (shift, off, val, payload) in enumerate(
                                    payloads[:payload_cap]
                                ):
                                    fname = payload_dir / (
                                        f"track{t:03d}_{label}_"
                                        f"shift{shift}_off{off:05d}_val{val:02x}.bin"
                                    )
                                    fname.write_bytes(payload)
                                    record_payload(
                                        f"{t}:{label}:mark_shift{shift}_off{off}",
                                        payload,
                                    )
                                if len(payloads) > payload_cap:
                                    print(
                                        f"  ...truncated payload dumps at {payload_cap} entries for {payload_dir}"
                                    )
                            else:
                                for shift, off, _val, payload in payloads[:payload_cap]:
                                    record_payload(
                                        f"{t}:{label}:mark_shift{shift}_off{off}",
                                        payload,
                                    )
                    if args.fixed_spacing_scan and len(entry) == 2 and payload_dir:
                        step_bits = args.fixed_spacing_step
                        window_bits = args.fixed_spacing_bytes * 8
                        h_a, h_b = entry
                        bit_offset = 0
                        idx = 0
                        while bit_offset + window_bits <= len(bits):
                            window = bits_to_bytes(
                                bits[bit_offset : bit_offset + window_bits]
                            )
                            fname = payload_dir / (
                                f"track{t:03d}_{label}_fixed{idx:03d}_"
                                f"off{bit_offset:05d}_len{len(window):03d}.bin"
                            )
                            fname.write_bytes(window)
                            record_payload(
                                f"{t}:{label}:fixed{idx}_off{bit_offset}", window
                            )
                            bit_offset += step_bits
                            idx += 1
                if (
                    args.stitch_rotation
                    or args.scan_bit_patterns
                    or args.bruteforce_marks
                ):
                    stitched_flux, stitched_ticks = stitch_rotation_flux(
                        track,
                        grouping,
                        rotation_index=0,
                        compensate_gaps=args.stitch_gap_comp,
                    )
                    if stitched_flux:
                        if args.invert_flux:
                            stitched_flux = list(reversed(stitched_flux))
                        flux_scaled = (
                            [max(1, int(x * clock_scale)) for x in stitched_flux]
                            if clock_scale != 1.0
                            else stitched_flux
                        )
                        stitched_bits = pll_decode_bits(
                            flux_scaled,
                            sample_freq_hz=image.sample_freq_hz,
                            index_ticks=stitched_ticks,
                            clock_adjust=args.clock_adjust,
                            initial_clock_ticks=None,
                            invert=invert_bits,
                        )
                        label = "rotation0_stitched"
                        if outdir:
                            fname = outdir / f"track{t:03d}_{label}.bits"
                            fname.write_bytes(bytes(stitched_bits))
                        if args.scan_bit_patterns:
                            hits = scan_bit_patterns(stitched_bits)
                            if hits:
                                print(
                                    f"Track {t} {label}: bit-pattern hits (shift,byte,val): {hits[:10]}"
                                )
                        if args.bruteforce_marks:
                            patterns = (
                                (0xFE, 0xFB)
                                if args.strict_marks
                                else (0xFB, 0xFA, 0xA1, 0xFE)
                            )
                            payloads = brute_force_mark_payloads(
                                stitched_bits,
                                payload_bytes=args.mark_payload_bytes,
                                patterns=patterns,
                            )
                            if payloads and payload_dir:
                                for _idx, (shift, off, val, payload) in enumerate(
                                    payloads[:payload_cap]
                                ):
                                    fname = payload_dir / (
                                        f"track{t:03d}_{label}_shift{shift}_off{off:05d}_val{val:02x}.bin"
                                    )
                                    fname.write_bytes(payload)
                                    record_payload(
                                        f"{t}:{label}:mark_shift{shift}_off{off}",
                                        payload,
                                    )
                            elif payloads:
                                for shift, off, _val, payload in payloads[:payload_cap]:
                                    record_payload(
                                        f"{t}:{label}:mark_shift{shift}_off{off}",
                                        payload,
                                    )
            if outdir:
                print(f"\nWrote bitcell dumps to {outdir}")
            if payload_dir:
                print(f"Wrote mark payload windows to {payload_dir}")

    if args.report_entropy and top_payloads:
        print("\nTop payload windows (lowest fill ratio, highest entropy):")
        for fill_ratio, neg_entropy, size, label, preview in top_payloads:
            entropy = -neg_entropy
            print(
                f"  {label}: size={size} fill={fill_ratio:.3f} entropy={entropy:.2f} "
                f"first16={preview[:16].hex()}"
            )


if __name__ == "__main__":
    main()
