#!/usr/bin/env python3
"""
Script para diagnosticar por qué no se está calculando el peso
Analiza los logs y muestra qué puntos clave faltan
"""

import re
import sys

def analyze_logs(log_file='app.log'):
    """Analiza los logs y muestra qué falta para calcular peso"""
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo {log_file}")
        return
    
    # Buscar logs de peso
    weight_logs = []
    for i, line in enumerate(lines):
        if '[WEIGHT]' in line:
            weight_logs.append((i+1, line.strip()))
    
    if not weight_logs:
        print("⚠️  No se encontraron logs de peso en el archivo.")
        print("   Esto puede significar:")
        print("   1. El debug no está activado cuando procesas videos")
        print("   2. No se están detectando vacas (YOLO no encuentra nada)")
        print("   3. No se está llamando a estimate_weight")
        print("\n💡 Solución: Activa el checkbox 'Debug' en la UI cuando proceses un video")
        return
    
    print(f"📊 Encontrados {len(weight_logs)} logs de peso\n")
    print("=" * 80)
    
    # Analizar cada log
    stats = {
        'total': 0,
        'with_eyes': 0,
        'with_keypoints': 0,
        'with_dist_ref': 0,
        'with_dist1': 0,
        'with_dist2': 0,
        'weight_calculated': 0,
        'weight_missing': 0
    }
    
    for line_num, line in weight_logs[-50:]:  # Últimos 50 logs
        stats['total'] += 1
        
        # Analizar ojos
        if 'eyes:' in line:
            if 'masks=0' in line:
                print(f"❌ Línea {line_num}: No se detectaron ojos (masks=0)")
            elif 'masks=' in line and 'masks=0' not in line:
                stats['with_eyes'] += 1
                if 'dist_ref=missing' in line:
                    print(f"⚠️  Línea {line_num}: Ojos detectados pero dist_ref faltante")
                elif 'dist_ref=ok' in line:
                    stats['with_dist_ref'] += 1
        
        # Analizar keypoints
        if 'keypoints:' in line:
            if 'found=False' in line:
                print(f"❌ Línea {line_num}: No se encontraron keypoints")
            elif 'found=True' in line:
                stats['with_keypoints'] += 1
                if 'dist1=missing' in line:
                    print(f"⚠️  Línea {line_num}: Keypoints encontrados pero dist1 faltante")
                elif 'dist1=ok' in line:
                    stats['with_dist1'] += 1
                if 'dist2=missing' in line:
                    print(f"⚠️  Línea {line_num}: Keypoints encontrados pero dist2 faltante")
                elif 'dist2=ok' in line:
                    stats['with_dist2'] += 1
        
        # Analizar peso calculado
        if 'weight=ok' in line:
            stats['weight_calculated'] += 1
            # Extraer valor del peso
            match = re.search(r'value=([\d.]+)kg', line)
            if match:
                weight_val = match.group(1)
                print(f"✅ Línea {line_num}: Peso calculado = {weight_val} kg")
        
        # Analizar peso faltante
        if 'weight=missing' in line:
            stats['weight_missing'] += 1
            reasons = []
            if 'dist_ref=False' in line:
                reasons.append("dist_ref (ojos)")
            if 'dist1=False' in line:
                reasons.append("dist1 (keypoint)")
            if 'dist2=False' in line:
                reasons.append("dist2 (keypoint)")
            
            if reasons:
                print(f"❌ Línea {line_num}: Peso NO calculado - Faltan: {', '.join(reasons)}")
    
    print("\n" + "=" * 80)
    print("📈 RESUMEN ESTADÍSTICO:")
    print("=" * 80)
    print(f"Total de logs analizados: {stats['total']}")
    print(f"✅ Con ojos detectados: {stats['with_eyes']} ({stats['with_eyes']/max(stats['total'],1)*100:.1f}%)")
    print(f"✅ Con dist_ref (ojos): {stats['with_dist_ref']} ({stats['with_dist_ref']/max(stats['total'],1)*100:.1f}%)")
    print(f"✅ Con keypoints encontrados: {stats['with_keypoints']} ({stats['with_keypoints']/max(stats['total'],1)*100:.1f}%)")
    print(f"✅ Con dist1 (keypoint): {stats['with_dist1']} ({stats['with_dist1']/max(stats['total'],1)*100:.1f}%)")
    print(f"✅ Con dist2 (keypoint): {stats['with_dist2']} ({stats['with_dist2']/max(stats['total'],1)*100:.1f}%)")
    print(f"✅ Peso calculado exitosamente: {stats['weight_calculated']} ({stats['weight_calculated']/max(stats['total'],1)*100:.1f}%)")
    print(f"❌ Peso NO calculado: {stats['weight_missing']} ({stats['weight_missing']/max(stats['total'],1)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("🔍 DIAGNÓSTICO:")
    print("=" * 80)
    
    if stats['total'] == 0:
        print("❌ No hay datos para analizar")
        return
    
    # Diagnóstico específico
    if stats['with_eyes'] == 0:
        print("❌ PROBLEMA PRINCIPAL: No se están detectando OJOS")
        print("   Solución:")
        print("   - Verifica que el modelo eye.pt esté funcionando")
        print("   - Las vacas deben estar de lado (vista lateral) para ver los ojos")
        print("   - Mejora la iluminación del video")
        print("   - Reduce YOLO_CONF para eye_model también")
    
    elif stats['with_dist_ref'] == 0:
        print("❌ PROBLEMA: Ojos detectados pero dist_ref no se calcula")
        print("   Esto puede indicar un problema en el código de cálculo de distancia")
    
    if stats['with_keypoints'] == 0:
        print("❌ PROBLEMA PRINCIPAL: No se están detectando KEYPOINTS")
        print("   Solución:")
        print("   - Verifica que el modelo cow.pt tenga keypoints configurados")
        print("   - Las vacas deben estar completas en el cuadro (no cortadas)")
        print("   - Vista lateral es mejor para detectar puntos clave del cuerpo")
        print("   - Reduce YOLO_CONF para cow_model")
    
    elif stats['with_dist1'] == 0 or stats['with_dist2'] == 0:
        print("⚠️  PROBLEMA: Keypoints detectados pero distancias no se calculan")
        print("   - Verifica que el modelo tenga al menos 5 keypoints")
        print("   - Los keypoints deben estar en posiciones válidas (no en 0,0)")
    
    if stats['weight_calculated'] > 0:
        print(f"✅ BUENAS NOTICIAS: Se calculó peso en {stats['weight_calculated']} casos")
        print("   El sistema funciona, solo necesita mejores condiciones de detección")
    
    if stats['weight_missing'] > 0 and stats['weight_calculated'] == 0:
        print("❌ PROBLEMA CRÍTICO: Nunca se calcula peso")
        print("   Revisa los problemas específicos listados arriba")

if __name__ == '__main__':
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'app.log'
    analyze_logs(log_file)

