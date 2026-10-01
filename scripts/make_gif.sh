#!/usr/bin/env bash
# Build a small, high-quality GIF from the demo video (two-pass palette for good colours).
# usage: scripts/make_gif.sh [input.mp4] [output.gif] [start_seconds] [duration_seconds] [width] [fps]
set -euo pipefail
IN="${1:-outputs/demo.mp4}"
OUT="${2:-docs/demo.gif}"
START="${3:-1}"
DURATION="${4:-9}"
WIDTH="${5:-720}"
FPS="${6:-12}"

mkdir -p "$(dirname "$OUT")"
FILTERS="fps=${FPS},scale=${WIDTH}:-1:flags=lanczos"
PALETTE="$(mktemp -t palette).png"
ffmpeg -loglevel error -y -ss "$START" -t "$DURATION" -i "$IN" -vf "${FILTERS},palettegen=stats_mode=diff" "$PALETTE"
ffmpeg -loglevel error -y -ss "$START" -t "$DURATION" -i "$IN" -i "$PALETTE" \
  -lavfi "${FILTERS}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" "$OUT"
rm -f "$PALETTE"
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
