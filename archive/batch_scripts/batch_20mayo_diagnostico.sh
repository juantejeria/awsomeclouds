#!/bin/bash
# Diagnostico barril v6 para los 14 individuos de 20mayo.
# Output: grids en grids_21frames_20mayo/<name>.png
set -e
cd "$(dirname "$0")"
source venv/bin/activate

MODEL=${1:-barril_seg_v6.pt}
TAG=${2:-20mayo}
SUFFIX=${3:-v6}

COWS=(
  "118_462" "118_510" "124_478" "124_498" "126_463" "127_435"
  "129_472" "129_477" "129_487" "129_504" "133_544" "134_532"
  "134_556" "135_630"
)

N=${#COWS[@]}
i=0
for v in "${COWS[@]}"; do
  i=$((i+1))
  echo ">>> [$i/$N] $v  (model=$MODEL)"
  python diagnostico_21frames_barril.py "checkpoints/20mayo/$v" \
    --barril-model "$MODEL" \
    --out "checkpoints/20mayo/$v/diagnostico_barril_grid_${v}_${SUFFIX}.png" 2>&1 | tail -2
done

mkdir -p "grids_21frames_${TAG}"
for v in "${COWS[@]}"; do
  src="checkpoints/20mayo/$v/diagnostico_barril_grid_${v}_${SUFFIX}.png"
  [ -f "$src" ] && cp "$src" "grids_21frames_${TAG}/$v.png"
done
echo "=== TODOS LISTOS — grids en grids_21frames_${TAG}/ ==="
