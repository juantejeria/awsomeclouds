# Resultados Completos — Estimación de Peso por Volumen de Barril

## Dataset: GRANDES
**Captura:** Cámara fija, animal pasa a distancia variable
**Ángulo:** Lateral a nivel del animal

| Individuo | Altura (cm) | Peso Real (kg) | Vol Barril (L) | Peso Barril (kg) | K |
|-----------|-------------|---------------|----------------|-----------------|-------|
| vaca_370_36 | 127 | 370 | 260.4 | 268.2 | 1.380 |
| vaca_382_36 | 108 | 382 | 193.4 | 199.2 | 1.918 |
| vaca_398_36 | 114 | 398 | 194.2 | 200.0 | 1.990 |
| vaca_449_36 | 119 | 449 | 184.4 | 189.9 | 2.364 |
| vaca_462_36 | 113 | 462 | 246.7 | 254.1 | 1.818 |
| vaca_487_36 | 120 | 487 | 266.4 | 274.4 | 1.775 |
| vaca_506_36 | 108 | 506 | 228.9 | 235.8 | 2.146 |
| vaca_510_36 | 124 | 510 | 246.0 | 253.4 | 2.013 |

**K: mean=1.925 | std=0.270 | variación=14%**

---

## Dataset: DESFILE 26MARZ
**Captura:** Animal pasa caminando, cámara a nivel del animal
**Ángulo:** Lateral a nivel del animal

| Individuo | Altura (cm) | Peso Real (kg) | Vol Barril (L) | Peso Barril (kg) | K |
|-----------|-------------|---------------|----------------|-----------------|-------|
| vaca1 | 76.0 | 192 | 99.7 | 102.7 | 1.870 |
| vaca2 | 80.1 | 185 | 73.4 | 75.6 | 2.447 |
| vaca3 | 74.7 | 197 | 85.9 | 88.5 | 2.226 |
| vaca4 | 75.0 | 192 | 58.1 | 59.8 | 3.211 |
| vacaC1 | 76.0 | 192 | 77.5 | 79.9 | 2.403 |
| vacaC2 | 80.1 | 185 | 86.0 | 88.6 | 2.088 |
| vacaC3 | 74.7 | 197 | 77.4 | 79.7 | 2.472 |
| vacaC4 | 75.0 | 192 | 100.6 | 103.7 | 1.851 |
| vacaC5 | 77.9 | 238 | 119.0 | 122.6 | 1.941 |
| vacaC6 | 75.7 | 196 | 73.0 | 75.2 | 2.606 |

**K: mean=2.312 | std=0.394 | variación=17%**

---

## Dataset: 26MARZ
**Captura:** Cámara elevada ~30-40° sobre el animal
**Ángulo:** Semi-cenital

| Individuo | Altura (cm) | Peso Real (kg) | Vol Barril (L) | Peso Barril (kg) | K |
|-----------|-------------|---------------|----------------|-----------------|-------|
| 1VisualA | 66.5 | 66.0 | 36.6 | 37.7 | 1.751 |
| 1VisualP | 66.5 | 66.0 | 23.9 | 24.6 | 2.683 |
| 2VisualA | 69.4 | 70.0 | 36.2 | 37.3 | 1.877 |
| 2VisualP | 69.4 | 70.0 | 33.8 | 34.8 | 2.011 |
| 4VisualA | 65.3 | 60.0 | 25.8 | 26.5 | 2.264 |
| 4VisualP | 65.3 | 60.0 | 22.7 | 23.4 | 2.564 |
| 5VisualA | 67.0 | 66.5 | 49.9 | 51.4 | 1.294 |
| 5VisualP | 67.0 | 66.5 | 39.5 | 40.7 | 1.634 |
| 6VisualA | 69.0 | 71.0 | 28.8 | 29.7 | 2.391 |
| 6VisualP | 69.0 | 71.0 | 41.4 | 42.7 | 1.663 |
| 7VisualA | 70.0 | 74.0 | 44.7 | 46.0 | 1.609 |
| 7VisualP | 70.0 | 74.0 | 32.4 | 33.4 | 2.216 |
| 8VisualP | 72.0 | 71.5 | 60.1 | 61.9 | 1.155 |
| 9VisualA250 | 115.0 | 250.0 | 254.7 | 262.4 | 0.953 |
| 9VisualC250 | 115.0 | 250.0 | 267.5 | 275.5 | 0.907 |
| 10VisualA253 | 111.5 | 253.0 | 253.5 | 261.1 | 0.969 |
| 10Visualc253 | 111.5 | 253.0 | 293.1 | 301.9 | 0.838 |

**K: mean=1.693 | std=0.586 | variación=35%**

---

## Resumen Comparativo

| Dataset | Individuos | K medio | K std | Variación K | Ángulo |
|---------|-----------|---------|-------|-------------|--------|
| Grandes | 8 | 1.925 | 0.270 | 14% | Lateral a nivel |
| Desfile 26marz | 10 | 2.312 | 0.394 | 17% | Lateral a nivel |
| 26marz | 17 | 1.693 | 0.586 | 35% | Elevado 30-40° |

---

## Conclusiones

### 1. Constante K por dataset
- Grandes (lateral, distancia variable): K=1.925, variación 14% — **sin vaca_370: K=2.003, variación 9%, error promedio 7.2%**
- Desfile (lateral, distancia estable): K=2.312, variación 17%
- 26marz (semi-cenital): K=1.693, variación 35% — **sin individuos 250/253 (yapa): K=1.932, variación 24%, error promedio 22.1%**

La K **no es universal**. Cada configuración de cámara (ángulo, distancia, altura) produce una K diferente. El ángulo de captura afecta directamente el volumen calculado.

**Caso vaca_370 (outlier en Grandes):** Este individuo tiene la altura más alta (127cm) pero el peso más bajo (370kg) del dataset, resultando en K=1.380 (muy por debajo del grupo). La altura de 127cm genera una escala grande que al elevarse al cubo infla el volumen desproporcionadamente. Existe una correlación negativa de -0.43 entre altura y K: a mayor altura medida, menor K (volumen más inflado). Esto sugiere que un error de pocos centímetros en la medición de altura tiene un impacto significativo en el volumen final. Si se descarta este outlier, la variación de K baja de 14% a 9% y el error promedio baja a 7.2%, lo cual indica que **la precisión en la medición de la altura es crítica**.

### 2. Efecto del ángulo de cámara
- Las capturas **laterales a nivel** del animal (Grandes y Desfile) producen K más consistentes (13-17% variación).
- Las capturas **semi-cenitales** (26marz) producen K con más variación (35%), porque la silueta vista desde arriba cambia mucho con pequeñas variaciones de ángulo.
- **Conclusión:** la captura lateral a nivel del animal es la más confiable.

**Nota 26marz:** Los individuos 9VisualA250/9VisualC250 y 10VisualA253/10Visualc253 son animales "yapa" (más grandes, ~250kg) mezclados en un dataset de animales chicos (~60-74kg). Al excluirlos, K=1.932 con 24% de variación y 22.1% de error promedio. El alto error se debe a que el modelo de barril fue entrenado con fotos laterales y no segmenta bien desde ángulo semi-cenital elevado (~30-40°). Los animales chicos filmados desde arriba son el dataset más problemático.

### 3. Efecto de la distancia
- En "Grandes" la distancia animal-cámara varía mucho (escala varía hasta 87% dentro del mismo video).
- En "Desfile" la distancia es más estable (animal pasa paralelo a la cámara).
- La escala se eleva al **cubo** para calcular volumen: 20% de variación en escala = 73% de variación en volumen.
- **Conclusión:** la distancia estable es crítica para resultados consistentes.

### 4. Cantidad de fotos
- Con **3-4 fotos** por individuo se obtienen resultados razonables si son de buena calidad.
- Más fotos ayudan a promediar y reducir el impacto de una mala detección individual.
- Los individuos con solo **2-3 fotos válidas** (ej: vacaC2 con 2) tienen más riesgo de error.
- Se recomienda un mínimo de **5-6 fotos** por individuo para un promedio robusto.
- Los mejores resultados se obtuvieron con **17 fotos** (vaca_506) donde el promedio suaviza outliers.

### 5. Calidad de las fotos
- **Fotos donde el animal no se detecta** se pierden (3-5% de rechazo típico).
- **Fotos con animal incompleto** (cortado por bordes) se descartan — genera pérdida de datos.
- **Fotos con sombras fuertes** confunden la segmentación (GrabCut falla, YOLO-seg es más robusto).
- **Fotos con otros animales** pueden confundir la detección (selecciona el animal equivocado).
- **Resolución baja o animal muy lejos** (bbox < 150px) produce escalas poco confiables.
- **Mejor calidad:** animal de perfil completo, bien iluminado, sin oclusiones, llenando al menos 30% del frame.

### 6. Limitación principal: profundidad
- El volumen se calcula asumiendo **sección circular** (diámetro = ancho lateral).
- En realidad la sección es **elíptica** (más ancha que profunda).
- Esto sobreestima sistemáticamente el volumen.
- Con una sola cámara no se puede medir la profundidad real.

### 7. Recomendaciones para mejorar
1. **Captura estandarizada:** cámara fija, animal pasa lateral a distancia constante (~3-5 metros).
2. **Calibración por configuración:** cada setup de cámara/ángulo/distancia necesita su propia K.
3. **Vista cenital:** agregar captura desde arriba para medir profundidad real y eliminar la asunción circular.
4. **Más datos de calibración:** mínimo 15-20 animales pesados por configuración para un K robusto.
5. **Mínimo 5-6 fotos** por individuo, todas de buena calidad (perfil lateral completo).

### 8. Estado actual
- El modelo de segmentación de barril (barril_seg v4) funciona correctamente en capturas laterales.
- La pipeline genera modelos 3D y volúmenes reproducibles.
- Con una K calibrada por configuración, el error promedio es ~14% (Desfile) a ~11% (Grandes sin outliers).
- Para uso práctico se requiere: captura estandarizada + K calibrada + suficientes datos de calibración.
