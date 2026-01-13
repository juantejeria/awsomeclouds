#!/bin/bash
# Script para ejecutar la UI con ngrok para acceso público

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

# Verificar si ngrok está instalado
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok no está instalado."
    echo ""
    echo "Para instalar ngrok:"
    echo "1. Descarga desde: https://ngrok.com/download"
    echo "2. O instala con Homebrew: brew install ngrok/ngrok/ngrok"
    echo "3. O con pip: pip install pyngrok"
    echo ""
    echo "Si usas pyngrok, ejecuta Streamlit primero y luego en otra terminal:"
    echo "  python -c 'from pyngrok import ngrok; print(ngrok.connect(8501))'"
    exit 1
fi

# Puerto por defecto de Streamlit
STREAMLIT_PORT=8501

# Verificar si Streamlit ya está corriendo en el puerto
if lsof -Pi :$STREAMLIT_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Streamlit ya está corriendo en el puerto $STREAMLIT_PORT"
    echo "Usando ese proceso..."
else
    echo "🚀 Iniciando Streamlit en el puerto $STREAMLIT_PORT..."
    # Ejecutar Streamlit en background
    streamlit run streamlit_app.py --server.port=$STREAMLIT_PORT --server.headless=true &
    STREAMLIT_PID=$!
    echo "Streamlit iniciado con PID: $STREAMLIT_PID"
    
    # Esperar a que Streamlit esté listo
    echo "Esperando a que Streamlit esté listo..."
    sleep 5
fi

# Iniciar ngrok
echo ""
echo "🌐 Iniciando ngrok tunnel..."
echo "Presiona Ctrl+C para detener tanto Streamlit como ngrok"
echo ""

# Ejecutar ngrok
ngrok http $STREAMLIT_PORT

# Si llegamos aquí, ngrok se detuvo
echo ""
echo "Ngrok detenido. Deteniendo Streamlit..."
if [ ! -z "$STREAMLIT_PID" ]; then
    kill $STREAMLIT_PID 2>/dev/null
fi

