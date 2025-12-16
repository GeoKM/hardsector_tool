# ACMS80217 Fixture

- Source: 8" SS/DD NASHUA hard-sectored (32 holes) capture via Greaseweazle using `--hardsector --raw --revs=5`; disk is believed to be 77 tracks, FM encoded with 16 sectors/track for Wang OIS media.
- Files (large; keep path stable for tests and scripts):
  - `ACMS80217-HS32.scp` (~46M, sha1 `e25598f3b644bfc09c9ac9e9b264f0bc826f4c33`) — raw flux baseline for decoders.
  - `ACMS80217-HS32-A.png` (~44M, sha1 `99c3cacdcbb2006dc4a6b1c4dcfa5d58265f6973`) — visual reference frame.
  - `ACMS80217-HS32-B.png` (~44M, sha1 `333da34571b897008d6a19267062fe2efc382040`) — alternate visual reference.
- Usage: point parsers/CLIs at `tests/ACMS80217/ACMS80217-HS32.scp` for deterministic runs; avoid copying these fixtures into additional folders to keep repo size controlled.
- If adding new captures, document provenance, sizes, and checksums here and prefer trimmed examples when possible.
