#!/usr/bin/env bash
# Download MediaPipe's browser runtime and the pose model into app/vendor (not committed).
# Run once before serving the app locally; the GitHub Pages workflow runs it on every deploy.
set -euo pipefail

TASKS_VISION_VERSION=1.1.1
MODEL_URL=https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

app="$(cd "$(dirname "$0")/.." && pwd)"
vendor="$app/vendor"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

mkdir -p "$vendor/tasks-vision" "$vendor/models"
(cd "$work" && npm pack "@mediapipe/tasks-vision@$TASKS_VISION_VERSION" --silent >/dev/null && tar xzf ./*.tgz)
cp "$work/package/vision_bundle.mjs" "$vendor/tasks-vision/"
for f in vision_wasm_internal vision_wasm_nosimd_internal; do
  cp "$work/package/wasm/$f.js" "$work/package/wasm/$f.wasm" "$vendor/tasks-vision/"
done

if [ ! -s "$vendor/models/pose_landmarker_full.task" ]; then
  curl -fsSL "$MODEL_URL" -o "$vendor/models/pose_landmarker_full.task.part"
  mv "$vendor/models/pose_landmarker_full.task.part" "$vendor/models/pose_landmarker_full.task"
fi
echo "vendor files ready in $vendor"
du -sh "$vendor"
