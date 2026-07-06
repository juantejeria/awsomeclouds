# Constante de Calibración K — Explicación del Problema

## ¿Qué es la constante K?

Buscamos un valor K tal que:

```
Peso Real del Animal = K × Peso Estimado del Barril
```

Si K es estable entre individuos, podemos estimar el peso de cualquier animal nuevo midiendo solo su barril.

## ¿Cómo calculamos el peso del barril?

1. Filmamos al animal caminando.
2. Detectamos la silueta y segmentamos el **barril** (torso sin cabeza, cuello, patas, cola) con un modelo entrenado.
3. Medimos la **altura real** del animal en campo (en cm).
4. Calculamos la **escala**: `escala = altura_real_cm / alto_bbox_px`. Esto convierte píxeles a centímetros.
5. Calculamos el **volumen** del barril rebanando la máscara horizontalmente. Cada rebanada tiene un ancho medido en cm (usando la escala). Asumimos sección circular para estimar el volumen.
6. Convertimos volumen a peso: `peso_kg = volumen_litros × 1.03`.

## El problema de la escala variable

La escala depende del **alto del bounding box en píxeles**. Este alto cambia si:

- El animal está **más lejos** → bbox más chico → escala más grande (más cm por píxel).
- El animal está **más cerca** → bbox más grande → escala más chica.
- El animal **agacha la cabeza** o cambia de postura → el alto del bbox cambia.
- YOLO detecta un **bbox ligeramente diferente** frame a frame.

### ¿Por qué afecta tanto al volumen?

La escala se aplica en **tres dimensiones**: largo, alto y profundidad. El volumen escala con el **cubo** de la escala:

```
Volumen_cm³ = Volumen_px³ × escala³
```

Si la escala varía un 20% entre frames (ej: 0.5 vs 0.6 cm/px), el volumen varía un **73%**:

```
(0.6 / 0.5)³ = 1.728 → 73% más volumen
```

Esto explica por qué dentro del mismo video, el mismo animal puede dar 400L en un frame y 1400L en otro.

### Datos reales observados

| Dataset | Variación de escala | Variación de volumen |
|---------|--------------------|--------------------|
| **Grandes** (cámara fija, animal se acerca/aleja) | 18% a 87% | Volúmenes varían 2x a 4x dentro del mismo animal |
| **Chicos** (fotos laterales, distancia estable) | ~15% | Volúmenes varían ~20% dentro del mismo animal |

## ¿Por qué los videos laterales (paralelos a la cámara) dan mejor K?

Cuando el animal camina **paralelo a la cámara** a distancia aproximadamente constante:

1. **La escala es estable**: el animal no se acerca ni se aleja, así que el bbox mantiene tamaño similar frame a frame. Menos variación de escala → menos variación de volumen.

2. **El ángulo es consistente**: siempre vemos al animal de perfil. La silueta del barril tiene una forma similar en todos los frames. No hay frames donde se ve de frente (barril angosto) ni de 3/4 (barril distorsionado).

3. **El barril se segmenta mejor**: el modelo fue entrenado mayormente con vistas laterales. Cuando el animal está de perfil, el modelo reconoce bien el barril. Cuando está girado, puede fallar (detectar de más o de menos).

4. **El volumen por rebanadas es más preciso**: la asunción de sección circular se aplica sobre el ancho lateral del barril. Si siempre vemos el perfil, ese ancho es consistente. Si el animal gira, el ancho aparente cambia pero no representa el mismo corte anatómico.

### Resultado concreto

En los individuos **chicos** (fotos laterales estables), el peso del barril estimado va de 59 a 110 kg para animales de 197-270 kg. Si tuviéramos los pesos reales, K probablemente tendría una variación de ~10-15%, suficiente para una constante de calibración útil.

En los **grandes** (cámara fija, animal se acerca/aleja con ángulos variables), K varía de 1.47 a 2.95 (variación del 23%), demasiado para una constante confiable.

## ¿Qué necesitamos para una buena constante K?

1. **Videos con distancia estable**: el animal pasa paralelo a la cámara, siempre a la misma distancia aproximada (ej: en una manga o pasillo).
2. **Vista lateral consistente**: el animal se mueve perpendicular a la línea de visión de la cámara.
3. **Pesos reales conocidos**: para los individuos de calibración, pesar al animal el mismo día de la filmación.
4. **Mínimo 15-20 individuos**: para que K sea estadísticamente robusto.

Con estos datos, la relación `Peso Real = K × Peso Barril` debería ser estable con un error estimado de ±10% o menos.

## Mejora futura: eliminar la asunción circular

La asunción de sección circular es la otra fuente principal de error. Si se agrega una captura desde ángulo elevado (vista del lomo desde arriba), se puede medir el ancho real del animal y usar secciones elípticas en vez de circulares. Esto reduciría la dependencia de K y mejoraría la precisión directa del volumen.
