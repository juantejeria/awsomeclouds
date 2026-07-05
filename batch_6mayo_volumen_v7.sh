#!/bin/bash
# Recompute de volumen para 6mayo usando barril_seg_v7.pt
# Alturas CALCULADAS (alturas_individuos.json → alturas_calc_6mayo_cm).
# Solo 19 individuos (el _20260513_121331 del JSON no tiene carpeta de frames).
# Output: output_modelos3d_live_6mayo_v7/
set -e
cd "$(dirname "$0")"
source venv/bin/activate

MIN_COV=${1:-55}
TAG=${2:-6mayo_v7}
MODEL=${3:-barril_seg_v7.pt}

COWS=(
  "116_316|117.5"
  "117_307|115.2"
  "117_330|117.6"
  "117_349|115.8"
  "117_363|117.3"
  "118_313_arriba|114.3"
  "119_311|110.6"
  "119_313|117.1"
  "119_389_adelante|117.0"
  "120_305|113.0"
  "120_321|113.7"
  "120_340_arriba|115.1"
  "121_322|124.8"
  "121_330|109.6"
  "121_418_arriba|125.6"
  "122_351|121.0"
  "124_341|113.0"
  "124_364_arriba|118.1"
  "124_374|117.5"
)

N=${#COWS[@]}
i=0
for entry in "${COWS[@]}"; do
  i=$((i+1))
  IFS='|' read -r name h <<< "$entry"
  echo ">>> [$i/$N] $name altura_calc=${h}cm  min_cov_x=${MIN_COV}%  model=${MODEL}"
  python procesar_21_frames_filtrado.py "checkpoints/6mayo/$name" "$h" "$name" \
    --min-cov-x "$MIN_COV" --out-tag "$TAG" --barril-model "$MODEL" 2>&1 \
    | grep -E "FILTRADO|consenso|envelope|A \(med|B \(p75|width env|alto max|frames descart|resumen|Done|error|barril=" \
    | head -30 || true
  echo "--- $name OK"
done
echo "=== TODOS LISTOS — output en output_modelos3d_live_${TAG}/ ==="
