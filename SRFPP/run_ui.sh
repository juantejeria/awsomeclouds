#!/bin/bash
# Script para ejecutar la UI con configuraciones que previenen crashes en macOS

cd "$(dirname "$0")"

# Activar entorno virtual
source .venv/bin/activate

# Configurar variables de entorno para prevenir crashes de PyTorch en macOS
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Ejecutar Streamlit (puedes sobreescribir con PORT=XXXX)
PORT="${PORT:-8502}"
streamlit run streamlit_app.py --server.port "$PORT"

