#!/bin/bash

# Script para iniciar ngrok y exponer el servidor Flask

PORT=5001

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
    echo "   Por favor, inicia primero el servidor con: python app.py"
    exit 1
fi

echo "✅ Servidor Flask detectado en puerto $PORT"
echo "🌐 Iniciando ngrok..."
echo ""

# Iniciar ngrok
ngrok http $PORT

