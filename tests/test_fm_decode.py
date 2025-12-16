from pathlib import Path

from hardsector_tool.fm import (
    brute_force_mark_payloads,
    decode_fm_bytes,
    decode_mfm_bytes,
    fm_bytes_from_bitcells,
    estimate_cell_ticks,
    pll_decode_fm_bytes,
    pll_decode_bits,
    scan_fm_sectors,
    mfm_bytes_from_bitcells,
)
from hardsector_tool.scp import SCPImage


FIXTURE = Path("tests/ACMS80217/ACMS80217-HS32.scp")


def test_fm_decode_produces_bytes() -> None:
    scp = SCPImage.from_file(FIXTURE)
    track = scp.read_track(0)
    assert track is not None
    flux = track.decode_flux(0)

    half, full, threshold = estimate_cell_ticks(flux)
    # Expect a clear separation between half and full cells.
    assert half < full
    assert threshold > 0

    result = decode_fm_bytes(flux)
    # Heuristic decode should produce a non-trivial byte stream.
    assert len(result.bytes_out) > 100
    assert result.bytes_out.count(0xFF) >= 1


def test_pll_decode_and_sector_scan() -> None:
    scp = SCPImage.from_file(FIXTURE)
    track = scp.read_track(0)
    assert track is not None
    flux = track.decode_flux(0)

    pll_result = pll_decode_fm_bytes(flux, sample_freq_hz=scp.sample_freq_hz, index_ticks=track.revolutions[0].index_ticks)
    assert len(pll_result.bytes_out) > 50

    guesses = scan_fm_sectors(pll_result.bytes_out)
    # Sector guesses may be zero on noisy data but should not raise.
    assert isinstance(guesses, list)


def test_fm_bytes_from_bitcells_chooses_phase() -> None:
    # FM pattern with alternating clock=1, data=0 bits (phase 0 clocks)
    bitcells = [1, 0] * 8
    phase, data = fm_bytes_from_bitcells(bitcells)
    assert phase in (0, 1)
    assert data  # should produce at least one byte


def test_scan_require_sync_filters_results() -> None:
    # Create a minimal stream with FE but no sync, then with sync.
    stream_no_sync = bytes([0x00, 0xFE, 0x00, 0x00, 0x00, 0x00, 0x00])
    assert scan_fm_sectors(stream_no_sync, require_sync=False) == []
    assert scan_fm_sectors(stream_no_sync, require_sync=True) == []

    stream_with_sync = bytes([0xA1, 0xFE, 0x00, 0x00, 0x00, 0x00, 0x00])
    assert scan_fm_sectors(stream_with_sync, require_sync=False) == []
    # With sync required, it still has too few bytes to form a sector, but call should work.
    assert scan_fm_sectors(stream_with_sync, require_sync=True) == []


def test_mfm_bitcell_roundtrip() -> None:
    # Build a simple MFM-encoded bitcell stream for two bytes: 0xA1, 0x4E
    # Generate clock/data bits according to MFM rules.
    def encode_mfm_byte(byte: int, prev_data_bit: int = 0) -> list[int]:
        bits = []
        for i in range(8):
            data_bit = (byte >> (7 - i)) & 1
            clock_bit = 0 if (data_bit or prev_data_bit) else 1
            bits.extend([clock_bit, data_bit])
            prev_data_bit = data_bit
        return bits, prev_data_bit

    bitcells: list[int] = []
    prev = 0
    for b in (0xA1, 0x4E):
        encoded, prev = encode_mfm_byte(b, prev)
        bitcells.extend(encoded)

    shift, decoded = mfm_bytes_from_bitcells(bitcells)
    # Expect to recover the original bytes regardless of offset (shift may be 0)
    assert decoded[:2] == bytes([0xA1, 0x4E])


def test_pll_decode_bit_inversion() -> None:
    # A tiny flux stream should produce a bit, and inversion should flip it.
    bits = pll_decode_bits([10], sample_freq_hz=10)
    inverted = pll_decode_bits([10], sample_freq_hz=10, invert=True)
    assert bits and inverted
    assert inverted[0] != bits[0]


def test_bruteforce_mark_payloads_extracts_window() -> None:
    def bytes_to_bits(data: bytes) -> list[int]:
        bits: list[int] = []
        for b in data:
            for i in range(8):
                bits.append((b >> (7 - i)) & 1)
        return bits

    bits = [0, 0, 0] + bytes_to_bits(bytes([0xFE, 0x11, 0x22, 0x33]))
    payloads = brute_force_mark_payloads(bits, payload_bytes=2, patterns=(0xFE,))
    assert payloads
    shift, offset, val, payload = payloads[0]
    assert (shift, offset, val) == (3, 0, 0xFE)
    assert payload.startswith(bytes([0xFE, 0x11]))
