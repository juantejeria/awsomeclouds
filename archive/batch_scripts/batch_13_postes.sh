#!/bin/bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate

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
  echo ">>> [$i/$N] $name altura=${h}cm (calib postes)"
  python procesar_21_frames_postes.py "checkpoints/22abril/$name" "$h" "$name" 2>&1 \
    | grep -E "calib|consenso|envelope|A \(med|B \(p75|width env|alto max|Done|error" || true
  echo "--- $name OK"
done
echo "=== TODOS LISTOS ==="
