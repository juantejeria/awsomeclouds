#!/bin/bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate

# Formato: cow_name|altura|carpeta
COWS=(
  "v1|92.5|central47_20260430_101748"
  "v2|100.1|central118_20260430_100425"
  "v3|92.3|central49_20260430_102859"
  "v4|100.1|central46_20260430_102939"
  "v5|92.3|central20_20260430_103145"
  "v7|96.2|central116_20260430_103223"
  "v8|92.3|central74_20260430_103429"
  "v9|94.1|central27_20260430_103504"
  "v10|95.1|central86_20260430_103611"
  "v12|100|central67_20260430_103950"
  "v13|97.9|central36_20260430_104033"
  "v14|98.6|central46_20260430_104124"
  "v15|96.5|central54_20260430_104233"
)

N=${#COWS[@]}
i=0
for entry in "${COWS[@]}"; do
  i=$((i+1))
  IFS='|' read -r name h folder <<< "$entry"
  echo ">>> [$i/$N] $name altura=${h}cm carpeta=$folder"
  python procesar_21_frames.py "checkpoints/22abril/$folder" "$h" "$name" 2>&1 \
    | grep -E "consenso|envelope|ply|resumen|error|Done|A \(med|B \(p75|E \(env|width env|alto max|malla" || true
  python diagnostico_21frames_barril.py "checkpoints/22abril/$folder" 2>&1 \
    | grep -E "ok|error|init\] [0-9]" || true
  echo "--- $name OK"
done
echo "=== TODOS LISTOS ==="
