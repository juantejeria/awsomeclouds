#!/usr/bin/env python
"""Script de diagnóstico para verificar el comportamiento del modelo"""

import os
import json
import numpy as np
from testing import ModelLoad, ImageScore

def test_all_animals(farm="Productor A", version=2):
    """Prueba el modelo con imágenes de todos los animales"""
    
    print("=" * 60)
    print(f"DIAGNÓSTICO DEL MODELO - Granja: {farm}")
    print("=" * 60)
    
    # Cargar modelo
    model_path = os.path.join('./checkpoints', farm, 'chckpt.best.h5')
    print(f"\n📦 Cargando modelo desde: {model_path}")
    model = ModelLoad(filepath=model_path).model_loader()
    
    # Cargar labels
    labels_path = os.path.join('./checkpoints', farm, 'labels.json')
    with open(labels_path) as f:
        labels = json.load(f)
    print(f"📋 Labels encontrados: {list(labels.keys())}")
    print()
    
    # Probar con cada animal
    results = {}
    for animal in ['animal_1', 'animal_2', 'animal_3', 'animal_4', 'animal_5']:
        animal_dir = os.path.join('dataset', animal)
        if not os.path.exists(animal_dir):
            print(f"⚠️  No existe directorio para {animal}")
            continue
        
        # Buscar primera imagen
        images = [f for f in os.listdir(animal_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            print(f"⚠️  No hay imágenes en {animal_dir}")
            continue
        
        test_image = os.path.join(animal_dir, images[0])
        print(f"\n🔍 Probando con {animal}: {test_image}")
        print("-" * 60)
        
        try:
            preds = ImageScore(
                model=model,
                img=test_image,
                farm=farm,
                version=version,
                confidence_threshold=0.5
            ).scores()
            
            predictions = preds['predictions']
            metadata = preds['metadata']
            
            print(f"✅ Predicción principal: {metadata['predicted_class']}")
            print(f"📊 Confianza máxima: {metadata['max_confidence']*100:.2f}%")
            print(f"\n📈 Todas las predicciones:")
            
            # Mostrar todas las predicciones ordenadas
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            for i, (class_name, confidence) in enumerate(sorted_preds):
                marker = "👉" if i == 0 else "  "
                print(f"{marker} {class_name}: {confidence*100:.2f}%")
            
            results[animal] = {
                'expected': animal,
                'predicted': metadata['predicted_class'],
                'confidence': metadata['max_confidence'],
                'all_predictions': predictions
            }
            
        except Exception as e:
            print(f"❌ Error procesando {animal}: {e}")
            results[animal] = {'error': str(e)}
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    
    correct = 0
    total = 0
    
    for animal, result in results.items():
        if 'error' in result:
            print(f"❌ {animal}: ERROR - {result['error']}")
            continue
        
        total += 1
        expected = result['expected']
        predicted = result['predicted']
        confidence = result['confidence']
        
        if expected == predicted:
            correct += 1
            status = "✅ CORRECTO"
        else:
            status = "❌ INCORRECTO"
        
        print(f"{status} {animal}: Esperado={expected}, Predicho={predicted} ({confidence*100:.2f}%)")
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n🎯 Precisión: {correct}/{total} ({accuracy:.1f}%)")
        
        if accuracy < 50:
            print("\n⚠️  ADVERTENCIA: El modelo tiene baja precisión.")
            print("   Posibles causas:")
            print("   1. Modelo sobreajustado a una clase")
            print("   2. Dataset desbalanceado")
            print("   3. Modelo necesita reentrenamiento")
            print("   4. Imágenes de prueba muy diferentes a las de entrenamiento")

if __name__ == '__main__':
    import sys
    
    farm = sys.argv[1] if len(sys.argv) > 1 else "Productor A"
    version = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    test_all_animals(farm=farm, version=version)



