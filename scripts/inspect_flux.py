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

from hardsector_tool.fm import decode_fm_bytes
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
    args = parser.parse_args()

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

    if args.decode_fm and tracks:
        track = image.read_track(tracks[0])
        if track is None:
            print("\nFM decode skipped: no data on track")
            return
        flux = track.decode_flux(0)
        result = decode_fm_bytes(flux)
        preview = result.bytes_out[:64]
        hex_preview = " ".join(f"{b:02x}" for b in preview)
        print(
            "\nFM decode (track {t}, rev 0):\n"
            " half-cell ~{hc:.1f} ticks, full-cell ~{fc:.1f} ticks, "
            "threshold {thr:.1f}, bit shift {shift}\n"
            " first 64 bytes: {preview}".format(
                t=tracks[0],
                hc=result.half_cell_ticks,
                fc=result.full_cell_ticks,
                thr=result.threshold_ticks,
                shift=result.bit_shift,
                preview=hex_preview,
            )
        )


if __name__ == "__main__":
    main()
