"""
SCP flux image parser and helpers.

Inspiration: Greaseweazle's public-domain SCP handling
(`scripts/greaseweazle/image/scp.py`). This module keeps a small,
documented subset tailored for inspection and decoding tasks in this
repository.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

SAMPLE_FREQ_HZ = 40_000_000  # SCP captures at 40 MHz (25 ns ticks)
TRACK_OFFSET_COUNT = 168


@dataclass(frozen=True)
class RevolutionEntry:
    index_ticks: int
    flux_count: int
    data_offset: int


@dataclass
class TrackData:
    track_number: int
    revolutions: List[RevolutionEntry]
    flux_data: bytes
    flux_data_offset: int

    @property
    def revolution_count(self) -> int:
        return len(self.revolutions)

    def decode_flux(self, rev_index: int) -> List[int]:
        rev = self.revolutions[rev_index]
        rel_offset = rev.data_offset - self.flux_data_offset
        end = rel_offset + rev.flux_count * 2
        segment = self.flux_data[rel_offset:end]
        return decode_flux_segment(segment)


@dataclass(frozen=True)
class SCPHeader:
    version: int
    disk_type: int
    revolutions: int
    start_track: int
    end_track: int
    flags: int
    cell_width_code: int
    capture_resolution: int
    heads: int
    checksum: int
    track_offsets: Tuple[int, ...]

    @property
    def non_empty_tracks(self) -> Tuple[int, ...]:
        return tuple(i for i, off in enumerate(self.track_offsets) if off)

    @property
    def track_count(self) -> int:
        return self.end_track - self.start_track + 1

    @property
    def sides(self) -> int:
        if self.heads <= 1:
            return 1
        return 2

    @property
    def cell_width(self) -> int:
        """
        Compatibility alias for the legacy cell_width field.

        SCP stores a bitcell width selector byte (typically 0 = 16-bit
        entries) plus a capture resolution byte. We expose the selector
        here to preserve existing prints.
        """
        return self.cell_width_code


class SCPImage:
    sample_freq_hz = SAMPLE_FREQ_HZ

    def __init__(self, header: SCPHeader, raw: bytes):
        self.header = header
        self._raw = raw

    @classmethod
    def from_file(cls, path: Path | str) -> "SCPImage":
        data = Path(path).read_bytes()
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SCPImage":
        (
            signature,
            version,
            disk_type,
            revolutions,
            start_track,
            end_track,
            flags,
            cell0,
            heads_raw,
            resolution,
            checksum,
        ) = struct.unpack("<3s9BI", data[0:16])
        if signature != b"SCP":
            raise ValueError("Not an SCP image (missing SCP signature)")

        track_offsets = struct.unpack("<168I", data[16:0x2B0])

        hdr = SCPHeader(
            version=version,
            disk_type=disk_type,
            revolutions=revolutions,
            start_track=start_track,
            end_track=end_track,
            flags=flags,
            cell_width_code=cell0,
            capture_resolution=resolution,
            heads=heads_raw,
            checksum=checksum,
            track_offsets=tuple(track_offsets),
        )
        return cls(header=hdr, raw=data)

    def track_offset(self, track_number: int) -> Optional[int]:
        if track_number < 0 or track_number >= len(self.header.track_offsets):
            return None
        off = self.header.track_offsets[track_number]
        return off or None

    def read_track(self, track_number: int) -> Optional[TrackData]:
        track_offset = self.track_offset(track_number)
        if track_offset is None:
            return None

        nr_revs = self.header.revolutions
        header_len = 4 + 12 * nr_revs
        track_header = self._raw[track_offset : track_offset + header_len]
        sig, trk_num = struct.unpack("<3sB", track_header[:4])
        if sig != b"TRK":
            raise ValueError(f"Track {track_number} missing TRK signature")
        if trk_num != track_number:
            raise ValueError(
                f"Track header number {trk_num} does not match expected {track_number}"
            )

        revs = []
        for idx in range(nr_revs):
            start = 4 + idx * 12
            end = start + 12
            index_ticks, flux_count, data_offset = struct.unpack(
                "<III", track_header[start:end]
            )
            revs.append(
                RevolutionEntry(
                    index_ticks=index_ticks,
                    flux_count=flux_count,
                    data_offset=data_offset,
                )
            )

        first_offset = revs[0].data_offset
        last_rev = revs[-1]
        flux_start = track_offset + first_offset
        flux_end = track_offset + last_rev.data_offset + last_rev.flux_count * 2
        flux_data = self._raw[flux_start:flux_end]

        return TrackData(
            track_number=track_number,
            revolutions=revs,
            flux_data=flux_data,
            flux_data_offset=first_offset,
        )


def decode_flux_segment(segment: bytes) -> List[int]:
    """
    Decode an SCP flux segment into tick durations.

    Each flux time is stored as a big-end 16-bit value; a word of 0 adds
    65536 ticks to the next non-zero value (carry).
    """
    flux = []
    carry = 0
    for idx in range(0, len(segment), 2):
        val = (segment[idx] << 8) | segment[idx + 1]
        if val == 0:
            carry += 65536
            continue
        flux.append(carry + val)
        carry = 0
    return flux
