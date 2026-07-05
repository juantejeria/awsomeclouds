#!/bin/bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate

MIN_COV=${1:-55}
TAG=${2:-14mayo}

COWS=(
  "100_137.5|85.6"
  "101_156|105.3"
  "102_150|108.2"
  "102_168.5|108.2"
  "102_171.5|90.2"
  "102_177|96.1"
  "103_169|105.4"
  "103.5_173.5|102.1"
  "105_182.5|100.9"
  "110_221|105.1"
  "110_228|112"
  "112_203|102.6"
  "113_214|109.1"
  "114_166|108.9"
)

N=${#COWS[@]}
i=0
for entry in "${COWS[@]}"; do
  i=$((i+1))
  IFS='|' read -r name h <<< "$entry"
  echo ">>> [$i/$N] $name altura_calc=${h}cm  min_cov_x=${MIN_COV}%"
  python procesar_21_frames_filtrado.py "checkpoints/14mayo/$name" "$h" "$name" \
    --min-cov-x "$MIN_COV" --out-tag "$TAG" 2>&1 \
    | grep -E "FILTRADO|consenso|envelope|A \(med|B \(p75|width env|alto max|frames descart|resumen|Done|error" \
    | head -30 || true
  echo "--- $name OK"
done
echo "=== TODOS LISTOS — output en output_modelos3d_live_${TAG}/ ==="
