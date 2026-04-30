#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

# Puerto: argumento posicional, env var PORT, o default 5001.
# Para correr varias instancias en paralelo:
#   ./run_app.sh 5001
#   ./run_app.sh 5002
#   ./run_app.sh 5003
PORT="${1:-${PORT:-5001}}"
export PORT

echo "🚀 Iniciando aplicación Flask..."
echo "📡 Servidor disponible en: http://localhost:${PORT}"
echo "⚠️  Presiona Ctrl+C para detener el servidor"
echo ""
python app.py
