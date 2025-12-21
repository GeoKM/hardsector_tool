# hardsector_tool

Tools to decode and reverse engineer hard-sectored floppy flux images (SCP). The current focus is two Wang OIS 100 8" SS/DD captures (32 holes + index, 5 revolutions per track) taken with Greaseweazle; CP/M-style presets are included for later comparisons.

## Repository layout
- `tests/ACMS80217`: Sample SCP plus two reference PNGs (captured with `--hardsector --raw --revs=5`, 77 tracks expected). Large binaries are ignored; see `tests/ACMS80217/README.md` for checksums.
- `tests/ACMS80221`: Second Wang OIS capture (same geometry and 5 revolutions) plus reference PNGs.
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
Hard-sector mark/bitcell sweep with polarity/clock experiments, hole merging, and stitched rotation:
```
python scripts/inspect_flux.py tests/ACMS80217/ACMS80217-HS32.scp \
  --preset cpm-16x256 --use-pll --track-range 0-60 \
  --dump-bitcells decoded/bitcells --scan-bit-patterns --bruteforce-marks \
  --mark-payload-dir decoded/mark_payloads --invert-bitcells --clock-scale 0.5 \
  --merge-hole-pairs --fixed-spacing-scan --stitch-rotation --stitch-gap-comp
```
Grid score FM/MFM hypotheses on a stitched rotation (mark count + entropy):
```
python scripts/inspect_flux.py tests/ACMS80217/ACMS80217-HS32.scp \
  --preset cpm-16x256 --use-pll --score-grid --stitch-rotation
```
Assemble a flat image and per-sector dumps:
```
python scripts/inspect_flux.py tests/ACMS80217/ACMS80217-HS32.scp \
  --preset cpm-16x256 --use-pll --sector-map --write-sectors decoded/track00 \
  --image-out decoded/wang_full.img
```
Key options:
- `--preset` (`cpm-16x256`, `cpm-26x128`, `ibm-9x512-mfm`, `auto`) sets encoding and expected sizes; `--encoding`, `--require-sync`, `--calibrate-rotation`, `--clock-scale`, `--use-pll` tune decoding strictness.
- `--hard-sector-summary`, `--flux-deltas` print grouping, index alignment, and index-vs-flux deltas per hole.
- Polarity/timing sweeps: `--invert-flux`, `--invert-bitcells`, `--clock-scale`, `--merge-hole-pairs`, `--stitch-rotation` (optional `--stitch-gap-comp`).
- Mark hunting: `--scan-bit-patterns`, `--bruteforce-marks`, `--mark-payload-dir`, `--fixed-spacing-scan`, `--report-entropy`, `--mark-payload-bytes`, `--bruteforce-step-bits`.
- Heuristics: `--score-grid` ranks FM/MFM candidates across clock scales and polarity; `--sector-map`, `--write-sectors`, `--image-out` assemble best-effort sector maps (missing sectors fill with 0x00).

## CLI: scan-metadata
Scan reconstructed logical sectors (output of `reconstruct-disk`) for catalog-like structures, name tables, and pointer evidence. This does not assume a filesystem and writes a JSON summary for reverse-engineering.

```
python -m hardsector_tool scan-metadata decoded/wang_recon --out decoded/metadata.json
```

The scanner works over `manifest.json` plus the `sectors/` and `tracks/` directories, flags dense name tables and record cadences, tallies likely pointer encodings, and records signature strings in context.

## Current Wang OIS 100 findings
- SCP headers show 198 revolutions per track; 77 even-numbered tracks have offsets; 32 sector holes plus one index per rotation. Index alignment is flagged in the header and respected when grouping.
- Hole-level decodes remain mostly uniform (0xFF or, when inverted, 0x00) across both disks. FM/MFM scans (sync/no-sync, PLL tuning, merged holes, fixed-spacing slices, stitched rotation) have not yielded valid CRCs or obvious structure; assembled images stay filler.
- Flux diagnostics (`--flux-deltas`) and stitched-rotation scans are available to validate capture coherence before deeper decoding attempts.

## Natural next steps
1. Use stitched-rotation scans plus `--score-grid` on both disks (tracks 0–20) to chase the likely Wang geometry (16 × 256 FM derived from 32 hard-sector holes); look for repeated mark payloads with `--bruteforce-marks` and entropy reporting.
2. Confirm capture consistency with `--flux-deltas` and index realignment; if deltas look stable, re-run mark sweeps with gap compensation to catch marks that straddle hole boundaries.
3. When any repeated payloads appear, pin down CHRN/DAM spacing and implement a Wang-specific frame parser; then extend to CP/M once reference media/examples arrive.
