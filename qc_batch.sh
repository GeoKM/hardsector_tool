#!/usr/bin/env bash
set -euo pipefail

# Root directory to scan (default: 8-00014-OIS_100-Software)
ROOT="${1:-8-00014-OIS_100-Software}"

# Python runner (adjust if needed)
PYTHON="${PYTHON:-.venv/bin/python}"

# Optional: capture full logs as well as brief summaries
WRITE_FULL_LOGS="${WRITE_FULL_LOGS:-1}"

# qc-capture options for HS32 batch triage
QC_MODE="${QC_MODE:-brief}"
TRACK_STEP="${TRACK_STEP:-auto}"
TRACKS="${TRACKS:-0-76}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: PYTHON not found/executable: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python or run from repo root with .venv/" >&2
  exit 2
fi

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: root directory not found: $ROOT" >&2
  exit 2
fi

echo "Scanning for .scp under: $ROOT"
echo "Using python: $PYTHON"
echo

# Use -print0 to safely handle spaces
find "$ROOT" -type f -name "*.scp" -print0 | while IFS= read -r -d '' scp; do
  dir="$(dirname "$scp")"
  base_dir="$(basename "$dir")"   # e.g. ACMS80215
  out_txt="$dir/${base_dir}-TA.txt"

  echo "QC: $scp"
  echo " -> $out_txt"

  # Run qc-capture and capture output
  # We record:
  #   - brief single-line result into *-TA.txt
  #   - optional full log alongside it
  tmp_out="$(mktemp)"
  tmp_err="$(mktemp)"
  set +e
  "$PYTHON" -m hardsector_tool qc-capture "$scp" \
    --mode "$QC_MODE" \
    --track-step "$TRACK_STEP" \
    --tracks "$TRACKS" \
    >"$tmp_out" 2>"$tmp_err"
  rc=$?
  set -e

  # Extract the key “brief” line(s). If mode=brief already produces one line,
  # we just take the first non-empty line from stdout.
  brief="$(awk 'NF{print; exit}' "$tmp_out")"
  if [[ -z "${brief:-}" ]]; then
    # If stdout empty, fall back to first non-empty stderr line
    brief="$(awk 'NF{print; exit}' "$tmp_err")"
  fi

  {
    echo "Image: $(basename "$scp")"
    echo "Path:  $scp"
    echo "Date:  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Cmd:   $PYTHON -m hardsector_tool qc-capture \"$scp\" --mode $QC_MODE --track-step $TRACK_STEP --tracks $TRACKS"
    echo
    echo "Result: ${brief:-<no output>}"
    echo "Exit:   $rc"
  } > "$out_txt"

  if [[ "$WRITE_FULL_LOGS" == "1" ]]; then
    full_log="$dir/${base_dir}-QC-full.txt"
    {
      echo "STDOUT:"
      cat "$tmp_out"
      echo
      echo "STDERR:"
      cat "$tmp_err"
      echo
      echo "Exit: $rc"
    } > "$full_log"
  fi

  rm -f "$tmp_out" "$tmp_err"
  echo
done

echo "Done."

