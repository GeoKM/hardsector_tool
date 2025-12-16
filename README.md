# hardsector_tool

Tools to decode and reverse engineer hard-sectored floppy flux images (SCP). Current focus: Wang OIS 100 8" SS/DD hard-sector (32 holes + index) captured with Greaseweazle; CP/M-style presets included for later testing.

## Repository layout
- `tests/ACMS80217`: Sample SCP and reference images (Greaseweazle `--hardsector --raw --revs=5`, 77 tracks expected). Large files are not tracked by git; see `README.md` inside for checksums.
- `src/hardsector_tool`: Parsers and decoders (`scp.py`, FM/MFM decoding in `fm.py`, hard-sector helpers in `hardsector.py`).
- `scripts/inspect_flux.py`: CLI for inspecting tracks, decoding holes, scanning sectors, and building raw images.

## Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## CLI usage
Inspect and decode (FM preset, hard-sector):
```
python scripts/inspect_flux.py tests/ACMS80217/ACMS80217-HS32.scp \
  --preset cpm-16x256 --use-pll --hard-sector-summary --scan-sectors \
  --sector-map --require-sync --calibrate-rotation \
  --write-sectors decoded/wang_track0 --image-out decoded/wang_full.img
```
Key options:
- `--preset`: `cpm-16x256`, `cpm-26x128`, `ibm-9x512-mfm`, or `auto` (set encoding, sector counts/sizes).
- `--encoding {fm,mfm}`: override preset encoding.
- `--hard-sector-summary`: show 33-hole rotations (32 holes + index).
- `--scan-sectors`: attempt to locate FM/MFM IDAMs and data with CRCs.
- `--sector-map`: build best-effort sector map across rotations; `--write-sectors DIR` dumps per-sector binaries.
- `--image-out PATH`: assemble a flat image (fills missing sectors with 0x00).
- `--use-pll`, `--calibrate-rotation`, `--require-sync`: adjust decoding strictness.

## Current Wang OIS sample status
- SCP header: 198 revolutions per track; 77 even-numbered tracks populated; 32 sector holes + 1 index per rotation.
- Attempts so far (FM/MFM, strict/loose, synthetic-from-holes, PLL tuning, mark scans) do not yield valid IDAMs/CRCs; assembled images are all fill bytes. Hole-level decodes are short and mostly uniform (0xFF or 0x00 when inverted).
- Investigation continues: use `--dump-holes`, `--dump-bitcells`, `--scan-marks`, and `--track-range` to collect evidence across tracks; a Wang-specific framing/directory parser will follow once sectors show structure.

## Notes for other formats
- CP/M presets are available (`--preset cpm-16x256` or `--preset cpm-26x128`), but sample media for validation is still needed.
- MFM decoding is supported via `--encoding mfm` and the IBM 9×512 preset; sync filtering can be enabled with `--require-sync`.
