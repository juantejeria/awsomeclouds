#!/bin/bash
# Volumen + modelo 3D para 20mayo con barril v7 (barril_seg_v7.pt).
# Output: output_modelos3d_live_20mayo_v7/
set -e
cd "$(dirname "$0")"
source venv/bin/activate

MIN_COV=${1:-55}
TAG=${2:-20mayo_v7}
MODEL=${3:-barril_seg_v7.pt}

COWS=(
  "118_462|115.1"
  "118_510|125.5"
  "124_478|119.6"
  "124_498|124.4"
  "126_463|129.5"
  "127_435|120.7"
  "129_472|123.7"
  "129_477|129"
  "129_487|124.4"
  "129_504|126.6"
  "133_544|122.8"
  "134_532|128.9"
  "134_556|134"
  "135_630|135"
)

N=${#COWS[@]}
i=0
for entry in "${COWS[@]}"; do
  i=$((i+1))
  IFS='|' read -r name h <<< "$entry"
  echo ">>> [$i/$N] $name altura_calc=${h}cm  min_cov_x=${MIN_COV}%  model=${MODEL}"
  python procesar_21_frames_filtrado.py "checkpoints/20mayo/$name" "$h" "$name" \
    --min-cov-x "$MIN_COV" --out-tag "$TAG" --barril-model "$MODEL" 2>&1 \
    | grep -E "FILTRADO|consenso|envelope|A \(med|B \(p75|width env|alto max|frames descart|resumen|Done|error|barril=" \
    | head -30 || true
  echo "--- $name OK"
done
echo "=== TODOS LISTOS — output en output_modelos3d_live_${TAG}/ ==="
