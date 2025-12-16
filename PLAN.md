# Work Plan

Goal: decode the Wang OIS 8" hard-sector captures (ACMS80217/ACMS80221, 32 holes + index, 5 revs) and build inspection tooling that can surface sector framing or payload structure.

Progress
- SCP parser now exposes cell width code and capture resolution; hard-sector grouping honors the FLAGS index-aligned bit and can auto-rotate to the detected index hole.
- FM/MFM decoding includes PLL-based FM framing, DAM-after-IDAM scanning, merged-hole and fixed-spacing sweeps, stitched-rotation scans, entropy reporting, and flux delta diagnostics.
- CLI can brute-force mark payloads, score FM/MFM hypotheses across clock scales and polarity, and assemble best-effort sector maps.

Next steps (guided by GPT-5.2 and Wang geometry prior: 16×256 FM from 32 hard-sector holes)
1. Run stitched-rotation sweeps with `--score-grid --stitch-gap-comp` across tracks 0–20 on both disks; capture top mark payloads/entropy outliers for clustering.
2. Use `--flux-deltas` to verify capture coherence; if stable, rerun mark sweeps with merged holes and gap compensation to catch IDAM/DAM that straddle hole boundaries.
3. When repeated payloads appear, pin down CHRN/DAM spacing for a Wang-specific parser (16 logical sectors per track) and then extend presets/decoders for CP/M once reference images arrive.
