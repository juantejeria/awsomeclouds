# Estimación de Peso de Carcasa Bovina mediante Visión por Computadora

## Resultados Actuales

| Individuo | Peso Real (kg) | Peso Barril Estimado (kg) | Error |
|-----------|---------------|--------------------------|-------|
| vaca_370 | 370 | 245.8 | +16.4% |
| vaca_382 | 382 | 177.8 | -3.5% |
| vaca_398 | 398 | 271.2 | +18.1% |
| vaca_449 | 449 | 168.1 | -12.5% |
| vaca_462 | 462 | 229.1 | -0.4% |
| vaca_487 | 487 | 245.1 | +0.5% |
| vaca_506 | 506 | 171.4 | -16.1% |
| vaca_510 | 510 | 252.2 | -0.5% |

## Metodología

### 1. Captura de datos
- Video del animal caminando, filmado con cámara fija.
- Se extraen múltiples fotos (screenshots) del video.

### 2. Detección del animal
- Se utiliza un modelo YOLO entrenado (`cow.pt`) para detectar el animal en cada frame.
- Se genera un bounding box (bbox) que enmarca al animal.
- Se filtran frames donde el animal no está completo (bbox toca los bordes de la imagen).

### 3. Segmentación de la silueta completa
- Un modelo YOLO-seg (`yolov8s-seg.pt`) genera la silueta completa del animal (con patas, cabeza, cola).

### 4. Segmentación del barril (carcasa)
- Un segundo modelo YOLO-seg (`barril_seg.pt`), entrenado específicamente con anotaciones manuales, segmenta únicamente la zona del **barril**: el torso del animal sin cabeza, cuello, patas ni cola.
- Este modelo fue entrenado con 111 frames anotados manualmente, definiendo el barril como la zona que representa la carcasa comercial del animal.

### 5. Escala y medidas reales
- La altura real de cada individuo fue medida manualmente en campo (en cm).
- La escala se calcula como: `escala = altura_real_cm / alto_bbox_px`.
- Todas las medidas en píxeles se convierten a centímetros usando esta escala.

### 6. Cálculo de volumen
- Sobre la máscara del barril, se calcula el volumen mediante integración por rebanadas horizontales.
- Para cada fila de píxeles de la máscara, se toma el ancho como diámetro y se calcula el área de una sección circular (`π × r²`).
- Se suman todas las rebanadas multiplicadas por su espesor en cm.
- El peso se estima como: `peso_kg = volumen_litros × 1.03` (densidad aproximada del tejido animal).

### 7. Promedio de frames
- Se procesan múltiples frames del mismo individuo.
- El resultado final es el **promedio** de los volúmenes de barril de todos los frames válidos.

## Limitación principal: la profundidad

El cálculo actual asume que cada sección transversal del barril es **circular** (el ancho visto de costado = diámetro del cilindro). En la realidad, la sección transversal de una vaca es **elíptica**: más ancha que profunda.

Esto introduce un error sistemático porque:
- Con una sola cámara lateral, solo vemos **ancho × alto** del barril.
- No podemos medir la **profundidad** (la dimensión que va del lomo a la panza vista desde arriba).
- Al asumir sección circular, sobreestimamos la profundidad y por lo tanto el volumen.

## Cómo mejorar: profundidad real desde ángulo elevado

El sistema ya cuenta con fotos de validación tomadas desde un **ángulo semi-cenital** (cámara elevada viendo el lomo del animal). Con este tipo de captura se puede obtener la profundidad real del barril sin asumir geometría.

### Metodología propuesta
1. **Vista lateral** (ya la tenemos): da el **alto × largo** del barril.
2. **Vista desde ángulo elevado** (fotos en `validate_profundidad`): da el **ancho real del lomo** (profundidad).
3. **Combinación**: para cada rebanada del barril, usar sección elíptica donde:
   - Eje mayor (`a`) = semi-ancho de la vista lateral.
   - Eje menor (`b`) = semi-ancho del lomo de la vista superior.
   - Volumen por rebanada: `π × a × b × espesor`.
4. Esto reemplaza la asunción circular por medidas reales en ambas dimensiones.

### Captura necesaria
- Filmar al animal pasando desde una posición elevada (ej: desde un puente, manga, o cámara montada a 2-3 metros de altura mirando hacia abajo).
- El animal camina por debajo y se captura el ancho del lomo a lo largo del cuerpo.
- No requiere sincronización — puede ser la misma cámara en otra pasada o posición.

### Impacto esperado
- Eliminación de la asunción geométrica circular, que es la principal fuente de error actual.
- Mayor consistencia entre individuos al medir la profundidad real en vez de estimarla.
- Reducción del error de ±16% actual a estimado ±5% o menos.
