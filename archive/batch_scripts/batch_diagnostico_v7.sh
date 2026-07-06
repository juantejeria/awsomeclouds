#!/bin/bash
# Genera diagnostico_barril_grid PNG con el modelo barril_seg_v7.pt
# para los 14 individuos de 14mayo y los 19 de 6mayo (33 total — el
# 120_305_20260513_121331 no tiene carpeta de frames y se saltea).
#
# Outputs:
#   grids_21frames_14mayo_v7/<cow>.png
#   grids_21frames_6mayo_v7/<cow>.png
#
# Los grids individuales también quedan en cada checkpoints/<ds>/<cow>/
# como diagnostico_barril_grid_v7.png (sin pisar el v6 existente).
set -e
cd "$(dirname "$0")"
source venv/bin/activate

MODEL=${1:-barril_seg_v7.pt}

COWS_14MAYO=(
  100_137.5 101_156 102_150 102_168.5 102_171.5 102_177
  103_169 103.5_173.5 105_182.5 110_221 110_228 112_203
  113_214 114_166
)

COWS_6MAYO=(
  116_316 117_307 117_330 117_349 117_363 118_313_arriba
  119_311 119_313 119_389_adelante 120_305 120_321 120_340_arriba
  121_322 121_330 121_418_arriba 122_351 124_341 124_364_arriba 124_374
)

run_set() {
  local ds=$1; shift
  local out_dir="grids_21frames_${ds}_v7"
  mkdir -p "$out_dir"
  local cows=("$@")
  local N=${#cows[@]}
  local i=0
  for cow in "${cows[@]}"; do
    i=$((i+1))
    local src="checkpoints/${ds}/${cow}"
    local per_cow_out="${src}/diagnostico_barril_grid_v7.png"
    echo ">>> [${ds} ${i}/${N}] ${cow}  model=${MODEL}"
    python diagnostico_21frames_barril.py "$src" \
      --barril-model "$MODEL" \
      --out "$per_cow_out" 2>&1 \
      | grep -E "init|ok|error|barril=" | head -10 || true
    if [ -f "$per_cow_out" ]; then
      cp "$per_cow_out" "${out_dir}/${cow}.png"
    fi
  done
  echo "=== ${ds}: grids en ${out_dir}/ ==="
}

run_set 14mayo "${COWS_14MAYO[@]}"
run_set 6mayo "${COWS_6MAYO[@]}"
echo "=== TODOS LISTOS ==="
