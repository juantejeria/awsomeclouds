#!/bin/bash
# Diagnostico barril v7 para los 14 individuos de 20mayo.
# Output: grids en grids_21frames_20mayo_v7/<name>.png
set -e
cd "$(dirname "$0")"
./batch_20mayo_diagnostico.sh barril_seg_v7.pt 20mayo_v7 v7
