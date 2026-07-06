# Modelos 3D de Ganado — Híbrido vs Multi-frame

## Resumen

La app genera modelos 3D de vacas a partir de video para estimar volumen corporal y peso. Existen **dos modos** de reconstrucción que comparten la misma fase de extracción de frames pero difieren en cómo generan la geometría 3D:

| | Modelo Híbrido | Modelo Multi-frame |
|--|----------------|---------------------|
| **Idea central** | Usa **1 frame** (el mejor) y le aplica profundidad sintética elíptica | Procesa **cada frame** con profundidad elíptica, filtra outliers por IQR, y promedia |
| **Velocidad** | Rápido (~2-5 seg) | Medio (~5-15 seg) |
| **Robustez** | Funciona bien incluso con videos difíciles | Más robusto: promedia N estimaciones independientes |
| **Precisión** | Depende de un solo frame | Mayor: el promedio suaviza errores individuales |
| **Corrección de volumen** | Ninguna (ConvexHull directo) | IQR descarta frames con volúmenes outlier |

---

## Pipeline Compartido (Fase 1: Extracción de Frames)

Ambos modos comparten la misma fase inicial:

```
Video MP4
   ↓
Extraer 1 frame cada N (configurable: 3-30 frames/seg)
   ↓
Para cada frame:
   1. Detección YOLO  →  bbox de la vaca
   2. Segmentación GrabCut  →  máscara binaria (silueta)
   3. Recorte de torso  →  elimina cabeza/cuello y patas
   ↓
Lista de (frame, máscara_torso) aceptados
```

### Detección (YOLO)
- Primero intenta con modelo custom entrenado en vacas (`cow.pt`)
- Si no detecta, fallback al modelo COCO genérico (clase 19 = cattle)
- Selecciona el bbox de mayor área

### Segmentación (GrabCut)
- Algoritmo iterativo (10 iteraciones) que separa foreground/background
- Limpieza morfológica: close (rellena huecos) + open (elimina ruido)
- Resultado: máscara binaria de la silueta de la vaca

### Recorte de Torso
- Analiza el **ancho por fila**: las patas son zonas donde el ancho cae < 40% del máximo
- Analiza el **alto por columna**: el cuello/cabeza son zonas donde el alto cae < 35% del máximo
- Busca la secuencia más larga de filas/columnas "gruesas" → esa es el torso/barril
- Validación: si el torso resultante es < 20% del área original, se usa la máscara completa

### Requisito
Se necesitan **mínimo 2 frames aceptados** para continuar a Fase 2.

---

## Modo Híbrido (`modelo_hibrido()`)

### Concepto
Toma el frame con la silueta más grande (= vaca más visible), le aplica una fórmula de profundidad basada en un perfil elíptico, y genera un modelo 3D simétrico.

### Pasos

#### 1. Selección del mejor frame
```python
best = frame con mayor área de máscara (más píxeles blancos)
```

#### 2. Muestreo de puntos 2D
- **Borde**: ~80 puntos equiespaciados sobre el contorno de la silueta
- **Interior**: ~40 puntos en una grilla regular, filtrados a los que caen dentro de la máscara
- Total: ~100-120 puntos 2D únicos

#### 3. Triangulación 2D (Delaunay)
Se genera una malla triangulada sobre los puntos 2D. Se descartan triángulos cuyo centroide cae fuera de la máscara.

#### 4. Escalado a centímetros
```
escala = cow_height_cm / bbox_height_px
puntos_cm = puntos_px × escala
```
La altura de la vaca viene de la calibración previa (2 postes de referencia).

#### 5. Profundidad sintética elíptica
Esta es la fórmula clave del modo híbrido:

```python
y_center = y_min + y_range × 0.4     # Centro vertical (40% desde abajo)

Para cada punto:
    d = min(|punto.y - y_center| / (y_range × 0.5), 1.0)    # Distancia normalizada [0,1]
    depth = y_range × 0.25 × √(1 - d²)                       # Perfil elíptico
```

**Interpretación geométrica:**
- Genera una sección transversal **elíptica** a lo largo del eje vertical
- En el centro (`y_center`, al 40% de la altura): profundidad máxima = 25% del alto
- Hacia arriba/abajo: la profundidad decrece siguiendo una elipse → forma natural del cuerpo
- El 0.4 (no 0.5) sesga el centro hacia abajo, donde el barril del animal es más ancho

```
        ╭──────╮          ← top (poca profundidad)
       /        \
      │          │         ← y_center: profundidad máxima
      │          │
       \        /
        ╰──────╯          ← bottom (poca profundidad)
```

#### 6. Espejo (simetría bilateral)
```
lado_derecho = [x, y, +depth]    → N puntos
lado_izquierdo = [x, y, -depth]  → N puntos (espejados)
modelo_3d = unión de ambos       → 2N puntos

triángulos_derecho = triángulos originales
triángulos_izquierdo = triángulos + offset, con winding invertido [0,2,1]
```

#### 7. Volumen y peso
```
volumen_cm³ = ConvexHull(modelo_3d).volume
volumen_litros = volumen_cm³ / 1000
peso_kg = volumen_litros × 1.03     # Densidad bovina ≈ 1.03 kg/L
```

---

## Modo Multi-frame (`sfm_desde_frames()`)

### Concepto
Procesa **cada frame extraído** de forma independiente con la misma fórmula de profundidad elíptica del modo Híbrido, luego filtra outliers por IQR y promedia los volúmenes. El modelo 3D visual corresponde al frame con volumen más cercano a la mediana.

### Ventaja sobre Híbrido
- No depende de un solo frame: si un frame tiene mala segmentación, el promedio lo compensa
- El filtro IQR descarta automáticamente frames con volúmenes anómalos
- Reporta desviación estándar → indica confianza del resultado

### Pasos

#### 1. Procesar cada frame independientemente
Para cada frame aceptado (de Fase 1):

```python
# Mismo pipeline que Híbrido por frame:
1. Contorno de mask_full (silueta completa)
2. Samplear borde (~100 pts) + interior (~50 pts)
3. Triangulación Delaunay 2D
4. Escalar a cm (bbox YOLO → cow_height_cm)
5. Profundidad elíptica (y_center=0.4, depth_max=0.25)
6. Espejo bilateral (+depth, -depth)
7. ConvexHull → volumen total + volumen barril
```

Resultado por frame: `vol_litros`, `vol_barril_litros`, `largo_cm`, `ancho_cm`

#### 2. Filtro IQR (InterQuartile Range)
```python
vols = [vol de cada frame]
Q1 = percentil(25)
Q3 = percentil(75)
IQR = Q3 - Q1

# Rango válido:
lower = Q1 - 1.5 × IQR
upper = Q3 + 1.5 × IQR

# Descartar frames fuera del rango
validos = [f for f in frames if lower ≤ f.vol ≤ upper]
```

Requiere mínimo 3 frames para aplicar IQR. Con menos, usa todos.

#### 3. Promediar métricas
```python
avg_vol = mean(validos.vol)
avg_vol_barril = mean(validos.vol_barril)
avg_largo = mean(validos.largo)
avg_ancho = mean(validos.ancho)
vol_std = stdev(validos.vol)
```

#### 4. Modelo visual del frame mediano
```python
vol_median = median(validos.vol)
best = frame más cercano a la mediana
# Se usa el modelo 3D de ese frame para visualización
```

#### 5. Volumen y peso (promediados)
```python
peso_kg = avg_vol × 1.03        # Sin corrección ×0.7
peso_barril_kg = avg_vol_barril × 1.03
```

No se aplica factor de corrección porque la profundidad elíptica ya genera volúmenes cercanos a la realidad (a diferencia del SfM puro que sobreestimaba).

---

## Fórmula de Peso

Ambos modos usan la misma conversión final:

```
Peso (kg) = Volumen (litros) × 1.03
```

Donde **1.03 kg/L** es la densidad aproximada del ganado bovino (similar al agua pero ligeramente mayor por músculo y hueso).

---

## Flujo en la UI

```
1. Usuario sube video
2. Navega frame a frame, analiza, calibra con 2 postes
   → Obtiene cow_height_cm
3. Aparecen dos botones:
   ┌──────────────────┐  ┌──────────────────┐
   │  Modelo Híbrido  │  │ Modelo Multi-frame│
   │  (violeta)       │  │   (gris oscuro)   │
   └──────────────────┘  └──────────────────┘
4. Click en cualquiera → barra de progreso:
   - 0-50%: Fase 1 (extracción de frames)
   - 50-100%: Fase 2 (reconstrucción según modo)
5. Resultado: tarjeta con Peso Total, Peso Barril, Volumen, Alto, Puntos 3D
6. Botón "Ver Modelo 3D" → visor Three.js interactivo
```

### Eventos SSE (Server-Sent Events)

| Evento | Fase | Datos |
|--------|------|-------|
| `started` | 1 | total_frames, frames_to_process, fps |
| `extracting` | 1 | frame_num, extracted, accepted, thumb_b64 |
| `sfm_progress` | 2 | step, total_steps, message |
| `complete` | Final | volumen, peso, dimensiones, ply_id |
| `error` | Error | message |

---

## Archivos Generados

```
output_modelos3d_batch/
  └── {vaca_name}/
      ├── {vaca_name}_3d.ply        ← Modelo 3D (vértices + caras + colores)
      └── {vaca_name}_resumen.json  ← Métricas (vol, peso, dims, method)
```

El archivo PLY se puede abrir en MeshLab, Blender, o el visor 3D integrado de la app.

---

## Cuándo usar cada modo

| Escenario | Recomendación |
|-----------|---------------|
| Video corto o pocos frames aceptados | **Híbrido** — funciona con 1 solo frame bueno |
| Necesito resultado rápido para screening | **Híbrido** — más rápido (~2-5 seg) |
| Quiero la estimación más robusta posible | **Multi-frame** — promedia N frames, descarta outliers |
| Video largo con muchos frames buenos | **Multi-frame** — más datos = menor varianza |
| Quiero saber la confianza del resultado | **Multi-frame** — reporta desviación estándar |
| Video con frames de calidad variable | **Multi-frame** — el IQR filtra los malos automáticamente |

---

## Constantes y Parámetros Clave

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Densidad bovina | 1.03 kg/L | Conversión volumen → peso |
| IQR multiplier | 1.5 | Factor para rango válido (Q1-1.5×IQR, Q3+1.5×IQR) |
| Mín frames IQR | 3 | Mínimo para aplicar filtro IQR |
| Profundidad centro | 40% altura | Centro del perfil elíptico (modo híbrido) |
| Profundidad máx | 25% del alto | Radio máximo del perfil (modo híbrido) |
| Torso: ancho mín | 40% del max | Umbral para detectar patas |
| Torso: alto mín | 35% del max | Umbral para detectar cabeza/cuello |
