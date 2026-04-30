#!/bin/bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate

MIN_COV=${1:-55}
TAG=${2:-filtrado}

COWS=(
  "v1|92.5"
  "v2|100.1"
  "v3|92.3"
  "v4|100.1"
  "v5|92.3"
  "v7|96.2"
  "v8|92.3"
  "v9|94.1"
  "v10|95.1"
  "v12|100"
  "v13|97.9"
  "v14|98.6"
  "v15|96.5"
)

N=${#COWS[@]}
i=0
for entry in "${COWS[@]}"; do
  i=$((i+1))
  IFS='|' read -r name h <<< "$entry"
  echo ">>> [$i/$N] $name altura=${h}cm  min_cov_x=${MIN_COV}%"
  python procesar_21_frames_filtrado.py "checkpoints/22abril/$name" "$h" "$name" \
    --min-cov-x "$MIN_COV" --out-tag "$TAG" 2>&1 \
    | grep -E "FILTRADO|consenso|envelope|A \(med|B \(p75|width env|alto max|frames descart|resumen|Done|error" \
    | head -30 || true
  echo "--- $name OK"
done
echo "=== TODOS LISTOS ==="
