# hardsector_tool

Forensic-oriented tooling to **reconstruct and analyze hard-sectored floppy disks** from **flux-level .scp (SuperCard Pro) images**, with an emphasis on **repeatable reverse engineering** rather than assumptions about known filesystems.

This project is designed for workflows where you have:

* flux captures (e.g. from **Greaseweazle** in raw flux + hardsector mode), and
* disks whose **logical layout / catalog format is unknown or only partially understood**.

## Preservation philosophy

This toolset separates stages:

1. **Flux capture** (external; keep original .scp unchanged)
2. **Sector reconstruction** (hardsector_tool): recover a complete logical sector database
3. **Metadata scanning** (hardsector_tool): locate candidate tables/labels/manifests
4. **Experimental carving** (hardsector_tool): extract derived payloads *only when pointer evidence is strong*, with full provenance

**Never discard flux data.** Derived artifacts (sector DB, carved modules) are secondary outputs.

---

## Installation

Recommended: editable install for development.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Inputs

### Flux images

* `.scp` flux images (SuperCard Pro format)
* Typically captured with multiple revolutions (e.g. 5 revs) and hardsector awareness enabled on the capture side.

### Expected media geometry (current focus)

* 77 tracks (0–76)
* 16 hard sectors per track
* 256 bytes per sector payload (as recovered by this tool)

> If your disk differs, treat it as a new artifact: confirm geometry from evidence (index/sector timing) rather than assuming.

---

## Suggested workflows

### Workflow A — Triage a new disk (recommended default)

1. **QC capture (flux heuristics + reconstruction health)**

   ```bash
   python -m hardsector_tool qc-capture <image.scp> \
     --mode detail \
     --track-step auto \
     --tracks 0-76
   ```

2. **Reconstruction (derived artefact)**

   ```bash
   python -m hardsector_tool reconstruct-disk <image.scp> \
     --out <recon_dir> \
     --track-step auto \
     --keep-best 2 \
     --force
   ```

3. **Scan metadata + message extraction (evidence-only)**

   ```bash
   python -m hardsector_tool scan-metadata <recon_dir> \
     --out <recon_dir>/scanmeta.json \
     --messages-out <recon_dir>/messages.tsv
   ```

4. **Write a short evidence note**

   * inferred geometry / assumptions used
   * QC summary PASS/WARN
   * any revision markers and message keyword clusters

### Workflow B — Reverse engineering / hypothesis-driven work

1. Use scanmeta to identify candidate regions.
2. Run catalog-report:

   ```bash
   python -m hardsector_tool catalog-report <recon_dir> --out <reports_dir>
   ```

3. If and only if evidence exists, run extract-modules:

   ```bash
   python -m hardsector_tool extract-modules <recon_dir> \
     --out <derived_dir> \
     --min-refs 3
   ```

> Carved artifacts are derived. Do not claim filesystem semantics unless proven.

### Example (ACMS80227)

A short, ACMS-specific run after capture:

```bash
python -m hardsector_tool qc-capture tests/ACMS80227/ACMS80227-HS32.scp --mode detail --track-step auto --tracks 0-76
python -m hardsector_tool reconstruct-disk tests/ACMS80227/ACMS80227-HS32.scp --out out_acms80227 --track-step auto --keep-best 2 --force
python -m hardsector_tool scan-metadata out_acms80227 --out out_acms80227/scanmeta.json --messages-out out_acms80227/messages.tsv
```

---

## Command reference

### QC a capture or reconstruction

Generate a preservation-focused report. The command prints a human-readable summary and writes JSON alongside the inputs unless `--out` is provided.

```bash
python -m hardsector_tool qc-capture <image.scp> --mode detail --track-step auto --revs 5
python -m hardsector_tool qc-capture <recon_dir> --mode brief
```

Tips:

* The QC JSON is saved as `qc_<image>.json` for SCP inputs or `<recon_dir>/qc.json` unless `--out` overrides the path.
* `--track-step` controls track spacing assumptions during QC reconstruction (use `--track-step 2` for even-only heads).
* `--reconstruct-verbose` mirrors `reconstruct-disk` debugging; combine with `--show-all-tracks` in `--mode detail` to audit clean tracks as well as failures.

### Reconstruct logical sectors from a flux image

Produce a reconstruction directory containing sector payloads, per-track reports, `manifest.json`, and summary stats.

```bash
python -m hardsector_tool reconstruct-disk <image.scp> --out <recon_dir> --track-step auto --keep-best 2 --force
python -m hardsector_tool reconstruct-disk <image.scp> --out <recon_dir> --track-step 2 --tracks 0-10 --side 0
```

Useful for validating decoding/mapping quickly before processing the full disk.

### Scan for metadata / candidate structures (format-agnostic)

Search for text signatures, dense filename-like tables, repeating record cadences, and pointer evidence.

```bash
python -m hardsector_tool scan-metadata <recon_dir> --out <recon_dir>/scanmeta.json
python -m hardsector_tool scan-metadata <recon_dir> --out <recon_dir>/scanmeta.json --messages-out <recon_dir>/messages.tsv
```

Use the JSON to guide reverse engineering (what sectors look like labels/catalogs/manifests). `scan-metadata` accepts track diagnostics named either `tracks/track_XX.json` or `tracks/TXX.json`.

### Catalog reporting (evidence-only)

Inventory descriptor-backed module candidates and name lists without writing payloads. If `--out` is omitted, reports land in `<recon_dir>/catalog_report/` (or the cache reconstruction directory when running directly from an SCP capture).

```bash
python -m hardsector_tool catalog-report <recon_dir> --out <reports_dir>
python -m hardsector_tool catalog-report <image.scp> --out <reports_dir> --cache-dir .qc_cache
```

### Experimental module extraction (carving with provenance)

`extract-modules` is **NOT** a general filesystem extractor. It attempts to carve derived payloads only when descriptor-style records or `pppp=` pointer evidence is strong.

```bash
python -m hardsector_tool extract-modules <recon_dir> --out <derived_dir> --min-refs 3 --force
python -m hardsector_tool extract-modules <recon_dir> --out <derived_dir> --only-prefix SYSGEN. --require-name-in-pppp-list --min-refs 3
```

Use `--hypotheses H1,H3` to try specific pointer decoding hypotheses or `--enable-pppp-descriptors` and `--pppp-span-sectors` when your hypothesis suggests padded `pppp=` pointers. Every extracted `.bin` gets a `.json` sidecar summarizing provenance and warnings.

---

## How to interpret outputs

* `qc-capture`: **PASS** means per-track decode stability; **WARN** flags marginal sectors or noisy timing. Recapture when you see recurrent WARNs across multiple revolutions.
* `manifest.json`: reconstruction decisions and stats per track; use it to explain how sector payloads were chosen.
* `scanmeta.json`: candidate structures and clusters to guide hypotheses. It is not a filesystem decode.
* `messages.tsv`: evidence extraction of packed UI/message tables. Offsets are byte offsets within each sector. By default only candidate sectors are included; add `--messages-include-all` to dump every sector’s strings.
* `catalog-report`: a `0/0` result means no recognized descriptor/name-table scheme was found; it does not imply a blank disk.
* `extract-modules`: produces carved artifacts with provenance sidecars; treat them as derived evidence, not confirmed files.

---

## Help / discoverability

```bash
python -m hardsector_tool --help
python -m hardsector_tool scan-metadata --help
python -m hardsector_tool reconstruct-disk --help
python -m hardsector_tool extract-modules --help
python -m hardsector_tool catalog-report --help
```

## License

Licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.

**Top-level commands:** `hardsector_tool {reconstruct-disk,scan-metadata,extract-modules,catalog-report,qc-capture}`

**reconstruct-disk options:** `--out`, `--tracks`, `--side`, `--logical-sectors`, `--track-step {auto,1,2}`, `--sectors-per-rotation`, `--sector-sizes`, `--keep-best`, `--similarity-threshold`, `--clock-factor`, `--dump-raw-windows`, `--no-json`, `--no-report`, `--force`

**scan-metadata options:** `--out OUT out_dir` plus message-catalog toggles (`--messages-out`, `--messages-min-len`, `--messages-include-all`, `--no-messages`)

**extract-modules options:** `--out`, `--min-refs`, `--max-refs`, `--hypotheses (H1,H2,H3,H4)`, `--enable-pppp-descriptors / --no-enable-pppp-descriptors`, `--pppp-span-sectors`, `--require-name-in-pppp-list`, `--only-prefix`, `--dry-run`, `--force`

**catalog-report options:** `--out`, `--min-refs`, `--max-refs`, `--hypotheses (H1,H2,H3,H4)`, `--enable-pppp-descriptors / --no-enable-pppp-descriptors`, `--pppp-span-sectors`, `--require-name-in-pppp-list`, `--only-prefix`, reconstruction toggles for SCP inputs (`--cache-dir`, `--force-reconstruct`, `--reconstruct-out`, `--tracks`, `--side`, `--track-step`, `--logical-sectors`, `--sectors-per-rotation`, `--sector-sizes`, `--keep-best`, `--similarity-threshold`, `--clock-factor`, `--dump-raw-windows`, `--force`)

**qc-capture options:** `--mode {brief,detail}`, `--tracks`, `--side`, `--track-step {auto,1,2}`, `--sectors-per-rotation`, `--revs`, `--cache-dir`, `--reconstruct-out`, `--force`, `--reconstruct/--no-reconstruct`, `--reconstruct-verbose`, `--logical-sectors`, `--reconstruct-sectors-per-rotation`, `--sector-sizes`, `--keep-best`, `--similarity-threshold`, `--clock-factor`, `--dump-raw-windows`, `--show-all-tracks`

---

## `extract-modules` options (most used)

* `--out DIR`
  Output directory for derived artifacts

* `--min-refs N`
  Minimum number of referenced sectors required to accept a pointer list (helps avoid noise)

* `--max-refs N`
  Safety cap on pointer list length

* `--hypotheses H1,H2,...`
  Comma-separated pointer decoding hypotheses to try (matches CLI `--hypotheses`)

* `--enable-pppp-descriptors` / `--no-enable-pppp-descriptors`
  Enable/disable treating `pppp=` entries with embedded pointers as extractable descriptors (see `extract-modules --help` for default)

* `--pppp-span-sectors N`
  Allow including bytes from the next sector when decoding `pppp=` pointer lists

* `--only-prefix PREFIX`
  Restrict extraction to names starting with PREFIX (case/leading `=` normalized)

* `--dry-run`
  Do discovery + parsing + summary, but do not write `.bin` payloads

* `--force`
  Allow overwriting an existing output directory

* `--require-name-in-pppp-list`
  Safety filter for true descriptors: only extract if the module name appears in `pppp=` lists on disk

### Option glossary (pppp-related)

* `--enable-pppp-descriptors` / `--no-enable-pppp-descriptors`: enable or disable treating `pppp=` entries with embedded pointers as extractable descriptors. (See `--help` for default.)
* `--pppp-span-sectors N`: allow reading pointer bytes from the next sector while decoding `pppp=` entries.

### Record types in output

`extraction_summary.json` separates evidence sources:

* **descriptor records**: true `=NAME//// ... pointer list ...`
* **name list hits**: `pppp=NAME.EXT` occurrences (not necessarily extractable)
* (when pppp descriptor extraction is enabled) **pppp descriptors**: `pppp=` entries upgraded to extractable when a pointer list is found after padding

This is intentional: it prevents name lists from inflating descriptor counts and keeps summaries interpretable.

---

## Output directory structure

### Reconstruct output (`<recon_dir>`)

* `manifest.json` — totals, mapping, track stats
* `sectors/Txx_Syy.bin` — recovered logical sector payloads (primary analysis artifact)
* `tracks/track_XX.json` or `tracks/TXX.json` — per-track decode diagnostics

### Extract output (`<derived_dir>`)

* `extraction_summary.json` — overall counts, hypotheses, extracted module list
* `<MODULE>.bin` — carved payload (derived)
* `<MODULE>.json` — provenance + pointer decoding used + sector refs + warnings

---

## Practical tips

* **Capture multiple revolutions** and retain raw flux images. If you suspect marginal media, recapture and compare.
* Prefer using `scan-metadata` before extraction. It helps identify where the real catalog/allocation structures might live.
* Treat extracted `.bin` files as **carved artifacts**, not “files” in the OS sense until the directory/allocation format is fully confirmed.

---

## Development / Tests

* Run checks locally before pushing: `pytest`, `ruff check .`, and `black .`.
* See `tests/fixtures/README.md` for optional large SCP fixtures used by slow tests.

---

## Status / limitations

* This tool currently focuses on **hard-sectored** layouts and evidence-driven reverse engineering.
* It does not yet implement a confirmed Wang/OIS filesystem directory decoder.
* Extraction is conservative by default and will skip records that do not meet plausibility thresholds.

If you discover a repeatable catalog/allocation structure, please preserve:

* the original `.scp`,
* the full reconstructed sector DB,
* and the exact tool version/commit used.
