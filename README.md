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

Run commands as:

```bash
python -m hardsector_tool --help
```

---

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

### Step 1 — Reconstruct logical sectors from a flux image

This produces a directory containing:

* per-sector payloads (`sectors/Txx_Syy.bin`)
* per-track reports (`tracks/*.json`)
* `manifest.json` and summary stats

Example:

```bash
python -m hardsector_tool reconstruct-disk ACMS80221-HS32.scp --out out_80221_v3
python -m hardsector_tool reconstruct-disk ACMS80217-HS32.scp --out out_80217_v3
```

**What you get:** a *logical sector database* suitable for analysis and repeatability.

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

---

## Help / discoverability

```bash
python -m hardsector_tool --help
python -m hardsector_tool extract-modules --help
python -m hardsector_tool scan-metadata --help
```

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
  Toggle treating `pppp=` entries with embedded pointers as descriptors (off by default)

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

### Record types in output

`extraction_summary.json` separates evidence sources:

* **descriptor records**: true `=NAME//// ... pointer list ...`
* **name list hits**: `pppp=NAME.EXT` occurrences (not necessarily extractable)
* (when `--enable-pppp-descriptors` is used) **pppp descriptors**: `pppp=` entries upgraded to extractable when a pointer list is found after padding

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
