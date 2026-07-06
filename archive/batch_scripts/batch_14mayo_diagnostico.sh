#!/bin/bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate

COWS=(
  "100_137.5" "101_156" "102_150" "102_168.5" "102_171.5" "102_177"
  "103_169" "103.5_173.5" "105_182.5" "110_221" "110_228" "112_203"
  "113_214" "114_166"
)

N=${#COWS[@]}
i=0
for v in "${COWS[@]}"; do
  i=$((i+1))
  echo ">>> [$i/$N] $v"
  python diagnostico_21frames_barril.py "checkpoints/14mayo/$v" 2>&1 | tail -2
done

mkdir -p grids_21frames_14mayo
for v in "${COWS[@]}"; do
  cp "checkpoints/14mayo/$v/diagnostico_barril_grid.png" "grids_21frames_14mayo/$v.png"
done
echo "=== TODOS LISTOS — grids en grids_21frames_14mayo/ ==="
