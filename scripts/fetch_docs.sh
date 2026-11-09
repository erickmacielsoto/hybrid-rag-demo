#!/usr/bin/env bash
set -euo pipefail
COMMIT=${FASTAPI_COMMIT:-ba47d6d71b3e806900c8471f2f661781b8c43ec4}
DEST=${1:-docs-src}
rm -rf "$DEST"
git clone --quiet --filter=blob:none --no-checkout https://github.com/fastapi/fastapi.git "$DEST"
git -C "$DEST" sparse-checkout set --no-cone "docs/en/docs/" "LICENSE"
git -C "$DEST" checkout --quiet "$COMMIT"
echo "FastAPI docs at $COMMIT -> $DEST/docs/en/docs"
