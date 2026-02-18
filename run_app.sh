#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
echo "🚀 Iniciando aplicación Flask..."
echo "📡 Servidor disponible en: http://localhost:5001"
echo "⚠️  Presiona Ctrl+C para detener el servidor"
echo ""
python app.py

