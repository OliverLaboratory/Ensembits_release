#!/usr/bin/env bash
#
# Pack ``data/`` and ``ckpt/`` into a single, symlink-dereferenced zip
# suitable for Zenodo upload. The output zip mirrors the in-repo layout
# so unzipping at the repo root puts everything back in place:
#
#   ensembits-repro/
#   ├── data/
#   │   ├── mdcath_real_bb/   (real files, not symlinks)
#   │   ├── misato_real_bb/
#   │   ├── tokens/
#   │   ├── codebooks/
#   │   ├── labels/
#   │   ├── splits/
#   │   └── pg/
#   └── ckpt/combined_esm3/
#
# Run from the repo root:
#
#   bash scripts/pack_data.sh                    # writes ./ensembits_repro_data.zip
#   bash scripts/pack_data.sh /path/to/out.zip   # custom path
#
# Total size ~9 GB; allow ~10 min on a fast disk.

set -euo pipefail

OUT="${1:-ensembits_repro_data.zip}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [ ! -d data ] || [ ! -d ckpt/combined_esm3 ]; then
    echo "ERROR: run scripts/setup_data_links.py first so data/ and ckpt/ are populated." >&2
    exit 1
fi

if command -v zip >/dev/null 2>&1; then
    # ``zip -y`` would store the symlinks; we want the real bytes, so omit -y.
    # ``-r`` recurses; default behavior follows symlinks to files and
    # directories.
    echo "[pack] writing $OUT (~9 GB; ETA ~10 min)..."
    zip -r "$OUT" data/ ckpt/combined_esm3/ \
        -x 'data/cached_descriptors/*'   # this is unused / empty
    echo "[pack] done: $(du -sh "$OUT" | cut -f1)  $OUT"
else
    # Fallback: tar.gz if zip isn't installed
    echo "[pack] zip not found; falling back to tar.gz"
    OUT="${OUT%.zip}.tar.gz"
    tar --dereference -czf "$OUT" data/ ckpt/combined_esm3/
    echo "[pack] done: $(du -sh "$OUT" | cut -f1)  $OUT"
fi

echo ""
echo "Upload $OUT to Zenodo, paste the public URL into MANIFEST.md,"
echo "and readers can fetch it with one curl + unzip."
