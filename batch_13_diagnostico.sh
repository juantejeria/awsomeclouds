#!/bin/bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate

COWS=(v1 v2 v3 v4 v5 v7 v8 v9 v10 v12 v13 v14 v15)
N=${#COWS[@]}
i=0
for v in "${COWS[@]}"; do
  i=$((i+1))
  echo ">>> [$i/$N] $v"
  python diagnostico_21frames_barril.py "checkpoints/22abril/$v" 2>&1 | tail -2
done

# Copiar todos los grids con nombre vN.png a grids_21frames/
mkdir -p grids_21frames
for v in "${COWS[@]}"; do
  cp "checkpoints/22abril/$v/diagnostico_barril_grid.png" "grids_21frames/$v.png"
done
echo "=== TODOS LISTOS — grids en grids_21frames/ ==="
