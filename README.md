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

## Quickstart workflow

### Step 0.5 — QC the capture or reconstruction

Use `qc-capture` to generate a preservation-focused report (brief by default, detail for deeper evidence). The command prints a
human-readable summary and writes JSON alongside the inputs unless `--out` is provided.

```bash
# SCP capture integrity (full QC + reconstruction cached in .qc_cache)
python -m hardsector_tool qc-capture ACMS80221-HS32.scp \
  --mode detail --sectors-per-rotation 16 --revs 5 --cache-dir .qc_cache

# Capture-only QC (skip reconstruction entirely)
python -m hardsector_tool qc-capture ACMS80221-HS32.scp --mode brief --no-reconstruct

# Explicit reconstruction target (write to a different directory than the cache)
python -m hardsector_tool qc-capture ACMS80221-HS32.scp \
  --mode detail --reconstruct-out out_80221_qc --force

# Reconstruction output health (already decoded directory)
python -m hardsector_tool qc-capture out_80221_v3 --mode brief
python -m hardsector_tool qc-capture out_80221_v3 --mode detail --out qc_out.json
```

Tips:

* The QC JSON is saved as `qc_<image>.json` for SCP inputs or `<out_dir>/qc.json` for reconstruction directories unless `--out`
  overrides the path.
* `--track-step` controls track spacing assumptions during QC reconstruction (use `--track-step 2` for even-only heads).
* `--reconstruct-verbose` mirrors `reconstruct-disk` debugging; combine with `--show-all-tracks` in `--mode detail` to audit
  clean tracks as well as failures.

### Step 1 — Reconstruct logical sectors from a flux image

This produces a directory containing:

* per-sector payloads (`sectors/Txx_Syy.bin`)
* per-track reports (`tracks/*.json`)
* `manifest.json` and summary stats

Examples:

* Default run (auto track step):

  ```bash
  python -m hardsector_tool reconstruct-disk ACMS80221-HS32.scp --out out_80221_v3
  ```

* Force even-only track step:

  ```bash
  python -m hardsector_tool reconstruct-disk ACMS80221-HS32.scp --out out_80221_step2 --track-step 2
  ```

* Explicit holes-per-rev with pruning:

  ```bash
  python -m hardsector_tool reconstruct-disk ACMS80217-HS32.scp --out out_80217_keep5 --sectors-per-rotation 16 --keep-best 5
  ```

* (Optional) cautious debug run:

  ```bash
  python -m hardsector_tool reconstruct-disk ACMS80217-HS32.scp --out out_80217_debug --dump-raw-windows
  ```

* Force even-only track stepping (common for single-sided captures):

  ```bash
  python -m hardsector_tool reconstruct-disk ACMS80221-HS32.scp --out out_80221_even --track-step 2
  ```

* Fast sanity run (decode just a few tracks while iterating):

  ```bash
  python -m hardsector_tool reconstruct-disk ACMS80221-HS32.scp --out out_80221_t0_5 --tracks 0-5 --side 0
  ```

Useful during development to validate decoding/mapping quickly before processing the full disk.

**What you get:** a *logical sector database* suitable for analysis and repeatability. Preserve it alongside the original `.scp`.

---

### Step 2 — Scan for metadata / candidate structures (format-agnostic)

This does **not** assume a filesystem. It searches for:

* text signatures (“Wang”, “SF…”, “Office Information System”, etc.)
* dense filename-like tables (NAME.EXT)
* repeating record cadences
* pointer plausibility evidence (TS pairs, linear sector numbers, etc.)

```bash
python -m hardsector_tool scan-metadata out_80221_v3 --out scan_80221.json
python -m hardsector_tool scan-metadata out_80217_v3 --out scan_80217.json
```

Use the JSON to guide reverse engineering (what sectors look like labels/catalogs/manifests).

---

### Step 3 — Experimental “module extraction” (carving with provenance)

`extract-modules` is **NOT** a general filesystem extractor.
It attempts to carve derived payloads only when it finds:

* descriptor-style records with pointer lists, and/or
* (optionally) pointer-bearing `pppp=` manifest entries

Every extracted `.bin` gets a `.json` sidecar with:

* where the descriptor was found (track/sector/offset),
* which pointer decoding hypothesis was used,
* the exact referenced sectors (provenance),
* warnings / missing refs.

#### Basic usage

```bash
python -m hardsector_tool extract-modules out_80221_v3 --out derived_80221 --min-refs 3 --force
python -m hardsector_tool extract-modules out_80217_v3 --out derived_80217 --min-refs 3 --force
```

* Conservative run:

  ```bash
  python -m hardsector_tool extract-modules out_80217_v3 --out derived_80217_cons --min-refs 3
  ```

* SYSGEN focus:

  ```bash
  python -m hardsector_tool extract-modules out_80217_v3 --out derived_sysgen --only-prefix SYSGEN. --min-refs 3
  ```

* Choose hypotheses:

  ```bash
  python -m hardsector_tool extract-modules out_80221_v3 --out derived_h13 --hypotheses H1,H3 --min-refs 3
  ```

* Toggle pppp descriptor carving:

  ```bash
  python -m hardsector_tool extract-modules out_80221_v3 --out derived_pppp_on --enable-pppp-descriptors --min-refs 3
  python -m hardsector_tool extract-modules out_80221_v3 --out derived_pppp_off --no-enable-pppp-descriptors --min-refs 3
  ```

  Note: if pppp descriptors are already enabled by default in your build, you can omit `--enable-pppp-descriptors`.

* Span pppp pointers across sectors:

  ```bash
  python -m hardsector_tool extract-modules out_80221_v3 --out derived_span1 --pppp-span-sectors 1 --min-refs 3
  ```

* Reduce false positives:

  ```bash
  python -m hardsector_tool extract-modules out_80217_v3 --out derived_sysgen_safe --only-prefix SYSGEN. --require-name-in-pppp-list --min-refs 3
  ```

#### Focus on a module family (prefix filter)

Prefix matching is normalized, so these behave the same:

```bash
python -m hardsector_tool extract-modules out_80217_v3 --out derived_sysgen --only-prefix SYSGEN. --min-refs 3 --force
python -m hardsector_tool extract-modules out_80217_v3 --out derived_sysgen --only-prefix =SYSGEN. --min-refs 3 --force
```

#### Reduce false positives (recommended while reverse engineering)

Only attempt extraction for true descriptor records **whose name is also present** in the disk’s `pppp=` name lists:

```bash
python -m hardsector_tool extract-modules out_80217_v3 --out derived_sysgen \
  --only-prefix SYSGEN. \
  --require-name-in-pppp-list \
  --min-refs 3 --force
```

Optional pppp-based descriptor carving (if your hypothesis suggests names-with-pointers sitting in `pppp=` records) can be toggled with `--enable-pppp-descriptors`, with `--pppp-span-sectors` controlling whether to include bytes from the next sector when parsing those entries.

### Catalog reporting (evidence-only)

`catalog-report` inventories descriptor-backed module candidates and name lists without writing payloads. It reuses the reconstruction cache used by `qc-capture` so running against an SCP image will transparently reconstruct into `.qc_cache/` (unless `--reconstruct-out` or `--force-reconstruct` is set). Outputs are written to `catalog_report.json` and `catalog_report.txt` inside the `--out` directory.

Examples:

```bash
# From an SCP image (reconstruction cached under .qc_cache/ by default)
python -m hardsector_tool catalog-report tests/ACMS80217/ACMS80217-HS32.scp --out reports/acms80217 --cache-dir .qc_cache

# From an existing reconstruction directory
python -m hardsector_tool catalog-report out_80217_v3 --out reports/out_80217_v3

# Filtered view requiring names to appear in pppp lists
python -m hardsector_tool catalog-report out_80217_v3 --out reports/out_sysgen --only-prefix SYSGEN. --require-name-in-pppp-list --enable-pppp-descriptors
```

---

## Help / discoverability

```bash
python -m hardsector_tool --help
python -m hardsector_tool scan-metadata --help
python -m hardsector_tool reconstruct-disk --help
python -m hardsector_tool extract-modules --help
python -m hardsector_tool catalog-report --help
```

**Top-level commands:** `hardsector_tool {reconstruct-disk,scan-metadata,extract-modules,catalog-report,qc-capture}`

**reconstruct-disk options:** `--out`, `--tracks`, `--side`, `--logical-sectors`, `--track-step {auto,1,2}`, `--sectors-per-rotation`, `--sector-sizes`, `--keep-best`, `--similarity-threshold`, `--clock-factor`, `--dump-raw-windows`, `--no-json`, `--no-report`, `--force`

**scan-metadata options:** `--out OUT out_dir`

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

* `--enable-pppp-descriptors` / `--no-enable-pppp-descriptors`: enable or disable treating `pppp=` entries with embedded pointers as extractable descriptors.
  (See `--help` for default.)
* `--pppp-span-sectors N`: allow reading pointer bytes from the next sector while decoding `pppp=` entries.

### Record types in output

`extraction_summary.json` separates evidence sources:

* **descriptor records**: true `=NAME//// ... pointer list ...`
* **name list hits**: `pppp=NAME.EXT` occurrences (not necessarily extractable)
* (when pppp descriptor extraction is enabled) **pppp descriptors**: `pppp=` entries upgraded to extractable when a pointer list is found after padding

This is intentional: it prevents name lists from inflating descriptor counts and keeps summaries interpretable.

---

## Output directory structure

### Reconstruct output (`out_*`)

* `manifest.json` — totals, mapping, track stats
* `sectors/Txx_Syy.bin` — recovered logical sector payloads (primary analysis artifact)
* `tracks/track_XX.json` — per-track decode diagnostics

### Extract output (`derived_*`)

* `extraction_summary.json` — overall counts, hypotheses, extracted module list
* `<MODULE>.bin` — carved payload (derived)
* `<MODULE>.json` — provenance + pointer decoding used + sector refs + warnings

---

## Practical tips

* **Capture multiple revolutions** and retain raw flux images. If you suspect marginal media, recapture and compare.
* Prefer using `scan-metadata` before extraction. It helps identify where the real catalog/allocation structures might live.
* Treat extracted `.bin` files as **carved artifacts**, not “files” in the OS sense until the directory/allocation format is fully confirmed.

---

## Status / limitations

* This tool currently focuses on **hard-sectored** layouts and evidence-driven reverse engineering.
* It does not yet implement a confirmed Wang/OIS filesystem directory decoder.
* Extraction is conservative by default and will skip records that do not meet plausibility thresholds.

If you discover a repeatable catalog/allocation structure, please preserve:

* the original `.scp`,
* the full reconstructed sector DB,
* and the exact tool version/commit used.
