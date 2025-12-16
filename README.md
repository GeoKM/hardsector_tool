# hardsector_tool

Tools to decode and reverse engineer hard-sectored floppy flux images (SCP). The current focus is a Wang OIS 100 8" SS/DD hard-sector capture (32 holes + index, 5 revolutions per track) taken with Greaseweazle; CP/M-style presets are included for future comparisons.

## Repository layout
- `tests/ACMS80217`: Sample SCP plus two reference PNGs (captured with `--hardsector --raw --revs=5`, 77 tracks expected). Large binaries are ignored; see `tests/ACMS80217/README.md` for checksums.
- `src/hardsector_tool`: Parsers and decoders (`scp.py`, FM/MFM + PLL helpers in `fm.py`, hard-sector framing in `hardsector.py`).
- `scripts/inspect_flux.py`: CLI for inspecting tracks, dumping holes/bitcells, scanning sectors/marks, and assembling raw images.
- `decoded/`: Scratch output (ignored by git) for dumps, mark payloads, and assembled images.

## Environment
```
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```
Use `.venv/bin/python` for commands in CI-like runs: `.venv/bin/python -m pytest`.

## CLI: inspect_flux
Quick summaries:
```
python scripts/inspect_flux.py tests/ACMS80217/ACMS80217-HS32.scp \
  --track 0 --revs 2 --hard-sector-summary --scan-sectors
```
Multi-track mark/bitcell sweep (with inverted polarity and slower clock):
```
python scripts/inspect_flux.py tests/ACMS80217/ACMS80217-HS32.scp \
  --preset cpm-16x256 --use-pll --track-range 0-60 \
  --dump-bitcells decoded/bitcells --scan-bit-patterns --bruteforce-marks \
  --mark-payload-dir decoded/mark_payloads --invert-bitcells --clock-scale 0.5
```
Assemble a flat image and per-sector dumps:
```
python scripts/inspect_flux.py tests/ACMS80217/ACMS80217-HS32.scp \
  --preset cpm-16x256 --use-pll --sector-map --write-sectors decoded/track00 \
  --image-out decoded/wang_full.img
```
Key options to know:
- `--preset` (`cpm-16x256`, `cpm-26x128`, `ibm-9x512-mfm`, `auto`) sets encoding and expected sizes.
- `--encoding {fm,mfm}`, `--require-sync`, `--calibrate-rotation`, `--clock-scale`, `--use-pll` tune decoding strictness.
- `--dump-holes`, `--dump-bitcells`, `--invert-bytes`, `--invert-bitcells` control raw dumps/polarity.
- `--scan-bit-patterns`, `--bruteforce-marks`, `--mark-payload-dir`, `--mark-payload-bytes` hunt for mark-like bytes and capture the following payload window.
- `--sector-map`, `--write-sectors`, `--image-out` assemble best-effort maps (missing sectors are filled with 0x00).

## Current Wang OIS 100 findings
- SCP header shows 198 revolutions per track; 77 even-numbered tracks have offsets; 32 sector holes plus one index per rotation.
- Hole-level decodes are short and mostly uniform (0xFF or, when inverted, 0x00). FM/MFM scans with sync/no-sync and PLL tuning have not produced valid CRCs or structured sector maps; assembled images are filler.
- Use the new mark sweep to look for structure: dump bitcells across many tracks, invert polarity, and brute-force mark payloads to spot repeated patterns or headers.

## Natural next steps
1. Sweep additional tracks (e.g., `--track-range 0-60`) with `--scan-bit-patterns`, `--bruteforce-marks`, `--invert-bitcells`, and multiple `--clock-scale` values to catch any polarity/timing quirks; review payload dumps for non-uniform bytes.
2. Cluster mark payloads by length/content to infer likely preambles and candidate sector sizes; feed promising offsets back into tighter PLL parameters.
3. Once a plausible frame emerges, implement a Wang-specific address/data parser; later, validate CP/M presets against known-good media when available.
