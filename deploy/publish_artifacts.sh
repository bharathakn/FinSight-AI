#!/usr/bin/env bash
# Publish the artifact bundle as a GitHub Release asset = versioned "model registry" (D-024).
# Needs: gh CLI logged in (docs/SETUP.md §4) and `make bundle` run first.
# Prints the download URL to use as ARTIFACTS_URL (Render build arg / FINSIGHT_ARTIFACTS_URI).
set -euo pipefail
cd "$(dirname "$0")/.."
TB=artifacts/bundle/finsight-artifacts.tar.gz
MAN=artifacts/bundle/current/manifest.json
[ -f "$TB" ] || { echo "no bundle: run 'make bundle' first" >&2; exit 1; }
VER=$(.venv/bin/python -c "import json; print(json.load(open('$MAN'))['version'])")
TAG="artifacts-${VER}"
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
if gh release view "$TAG" >/dev/null 2>&1; then
  echo "release $TAG already exists (bundle content unchanged)"
else
  gh release create "$TAG" "$TB" "$MAN" --title "Model artifacts ${VER}" \
    --notes "FinSight AI serving bundle ${VER}: production model, anomaly detectors (ONNX), clustering, rules, graph metrics, sanctions entities, transaction store. sha256 per file in manifest.json."
fi
echo "ARTIFACTS_URL=https://github.com/${REPO}/releases/download/${TAG}/finsight-artifacts.tar.gz"
