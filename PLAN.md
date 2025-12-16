# Work Plan

Goal: build a usable hard-sector SCP flux decoder and inspection CLI tailored to the ACMS80217 sample (8" SS/DD, hard-sector 32, recorded with Greaseweazle `--hardsector --raw --revs=5`).

Short-term tasks
- SCP ingestion: parse the image header (tracks 0–152, single-sided) and 168-entry track offset table; extract per-track revolution metadata and flux blobs.
- Flux decoding utilities: convert SCP bitcell samples (40 MHz base) into tick durations per revolution; verify sums against the recorded index intervals.
- CLI: add `scripts/inspect_flux.py` to summarize headers and track stats (track range, revolution count, index timing, flux counts) with an option to decode a limited number of revolutions for speed.
- Tests: assert header fields match the fixture, track offsets are populated on even indices (77 tracks), and decoded flux for track 0 revolution 0 matches stored counts and index timing within tolerance.

Notes and assumptions
- Track numbers are absolute (`cylinder * sides + side`); in this image only even-numbered tracks are present, consistent with single-sided capture.
- SCP sample frequency is 40 MHz; flux deltas are stored as big-end 16-bit words with 0 indicating an extra 65,536 ticks (carry logic mirrors Greaseweazle’s SCP code).
- Keep fixture reads to the minimum needed for tests (one track, first revolution by default) to avoid long runtimes.
- Future steps: layer FM decoding (expected 16 × 256-byte sectors/track) and format detection on top of the flux parser, then extend the CLI to emit decoded sector maps.
