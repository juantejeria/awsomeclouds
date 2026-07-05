#!/bin/bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate

COWS=(116_316 117_307 117_330 117_349 117_363 118_313_arriba \
      119_311 119_313 119_389_adelante \
      120_305 120_321 120_340_arriba \
      121_322 121_330 121_418_arriba \
      122_351 124_341 124_364_arriba 124_374)
N=${#COWS[@]}
i=0
for v in "${COWS[@]}"; do
  i=$((i+1))
  echo ">>> [$i/$N] $v"
  python diagnostico_21frames_barril.py "checkpoints/6mayo/$v" 2>&1 | tail -2
done

# Copiar todos los grids con nombre <id>.png a grids_21frames_6mayo/
mkdir -p grids_21frames_6mayo
for v in "${COWS[@]}"; do
  cp "checkpoints/6mayo/$v/diagnostico_barril_grid.png" "grids_21frames_6mayo/$v.png"
done
echo "=== TODOS LISTOS — grids en grids_21frames_6mayo/ ==="
