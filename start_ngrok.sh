#!/bin/bash

# Script para iniciar ngrok y exponer el servidor Flask
# Uso:
#   ./start_ngrok.sh        → expone puerto 5001 (default)
#   ./start_ngrok.sh 5002   → expone puerto 5002

PORT="${1:-5001}"

echo "🚀 Iniciando ngrok para exponer puerto $PORT..."
echo "📡 Tu servidor Flask estará disponible públicamente"
echo ""
echo "⚠️  IMPORTANTE:"
echo "   - Mantén esta terminal abierta mientras uses ngrok"
echo "   - La URL pública cambiará cada vez que reinicies ngrok (a menos que tengas cuenta premium)"
echo "   - Presiona Ctrl+C para detener ngrok"
echo ""

# Verificar que el servidor Flask esté corriendo
if ! lsof -ti:$PORT > /dev/null 2>&1; then
    echo "❌ Error: No hay servidor Flask corriendo en el puerto $PORT"
    echo "   Iniciá el servidor primero con: ./run_app.sh $PORT"
    exit 1
fi

echo "✅ Servidor Flask detectado en puerto $PORT"
echo "🌐 Iniciando ngrok..."
echo ""

# Iniciar ngrok
ngrok http $PORT
