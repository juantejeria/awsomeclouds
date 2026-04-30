# Sistema de Escala, Altura y Volumen (estado actual)

Documento de referencia del pipeline completo de medición al 2026-04-22, antes de
pasar a la versión de "barril consenso" (combinación multi-frame).

## 1. Objetivo

Medir **altura (cm)** y **volumen del barril (L)** de una vaca que pasa por una
zona delimitada, usando dos postes con cintas rojas de 50 cm como referencia de
escala. Resultado descargable + modelo 3D interactivo.

## 2. Setup físico

- Dos postes verticales con **cinta roja de 50 cm** cada uno
- Cámara de costado, idealmente perpendicular al pasaje
- Postes a **profundidades distintas** respecto a cámara (uno cerca, otro lejos)
- Piso de pasto/tierra continuo

## 3. Calibración: generar el "rectángulo fijo"

Endpoint `POST /calibrate_frame`

Pasos:
1. YOLO `sticker.pt` + fallback HSV detecta los 2 postes
2. Dentro de cada bbox de poste, se detecta el tramo rojo continuo (la cinta 50 cm)
3. Desde la base de la cinta se escanea hacia abajo buscando pasto verde
   (HSV) → determina el nivel de piso (`floor`) al pie de cada poste
4. Se arma el rectángulo con 4 vértices:
   - **Top**: `(post1_cx, top_tape_1)` y `(post2_cx, top_tape_2)`
   - **Bottom (línea del piso)**: `(post1_cx, floor_1)` y `(post2_cx, floor_2)`
     — suele estar ligeramente inclinada por la perspectiva
5. La respuesta incluye:
   - `rectangle_ref` con coords en espacio RESIZED (letterbox)
   - `original_coords` con las mismas coords mapeadas a resolución nativa del video

## 4. Fijación de referencia

Endpoint `POST /lock_reference` + frontend `AppState.lockedReference`

- Frontend guarda `original_coords` y genera un `video_id` único por sesión
- Cada request al backend envía `locked_reference_json` **inline** (con
  `original_coords`) para sobrevivir reloads del servidor
- Backend cachea en `_locked_references[video_id]` como respaldo

## 5. Detección por frame (`POST /detect_cow_fast`)

Es el endpoint crítico — corre por cada frame durante "Detectar pasadas".

### 5.1 Detección de la vaca

Se usa `WeightEstimator._detect_all_cows()`, que internamente hace una cascada:
1. `cow.pt` sobre imagen original (con augment)
2. `cow.pt` sobre letterbox
3. `cow.pt` sin augment
4. COCO `yolov8n.pt` con `classes=[19]` (cow) — fallback
5. Two-stage: COCO bbox → `cow.pt` sobre el crop → refina bbox + keypoints

Output: bbox en coords **RESIZED (letterbox)**.

### 5.2 Ajuste del BOTTOM del bbox con `silueta_seg.pt`

El bbox de YOLO a veces se extiende por debajo de los pies → altura inflada.
Para corregir:

1. Crop de la región del bbox + 5% padding
2. `silueta_seg.pt` (YOLO-seg, modelo entrenado para toda la vaca) sobre el crop
3. Toma la máscara de mayor área
4. `ry2 = bottom absoluto de la máscara` (última fila con mask > 0)
5. **Sanity check**: si el nuevo `ry2` difiere > 30% del alto de YOLO, se descarta
   y queda el `ry2` original
6. Retorna flag `silueta_bottom_used`

### 5.3 Ajuste del TOP del bbox con `barril_seg.pt`

El top de YOLO incluye la cabeza (que se mueve independientemente del cuerpo).
Para corregir:

1. Mismo crop que silueta
2. `barril_seg.pt` (YOLO-seg, modelo entrenado para el TORSO sin cabeza) sobre el crop
3. Toma la máscara de mayor área
4. `ry1 = top absoluto de la máscara` (primera fila con mask > 0)
5. También se recortan los laterales (`rx1, rx2`) al barril
6. **Se guarda `barril_binmask` y `barril_crop_origin`** para cálculo de volumen posterior
7. Retorna flag `barril_top_used`

### 5.4 Mapeo a coords originales

Bbox final `(x1, y1, x2, y2)` se convierte de letterbox a original:
```
x = (x_resized - pad_x) / scale_factor
```

### 5.5 Cruce geométrico bbox-piso (escala)

Segmentos a intersectar:
- **Piso del rectángulo**: `(cx1, floor1) → (cx2, floor2)` — inclinado
- **Bottom del bbox**: `(x1, y2) → (x2, y2)` — horizontal

Rango X común: `[max(x1, cx1), min(x2, cx2)]`. Si no se superponen → descartado.

En ese rango:
```
fy_lo = floor_at(X_lo)
fy_hi = floor_at(X_hi)
d_lo = fy_lo - y2
d_hi = fy_hi - y2
```

Si `d_lo × d_hi <= 0` → las dos rectas se cruzan en un punto dentro del rango.

Se resuelve:
```
alpha = d_lo / (d_lo - d_hi)
x_cross = X_lo + alpha × (X_hi - X_lo)
t = (x_cross - cx1) / (cx2 - cx1)
```

`t` es la **posición del cruce** a lo largo del segmento del piso (0 = post1,
1 = post2).

### 5.6 Interpolación de escala

```
scale_1 = 50 / tape_px_1   (cm/px a la profundidad del post 1)
scale_2 = 50 / tape_px_2   (cm/px a la profundidad del post 2)
cm_per_px = (1 - t) × scale_1 + t × scale_2
```

Esta escala es **específica del frame**, depende de dónde pisa la vaca.

### 5.7 Altura del frame

```
altura_cm = (y2 - y1) × cm_per_px
```

Donde `y2 = pies (silueta)`, `y1 = cruz (barril)`.

### 5.8 Volumen del barril del frame

Usando la `barril_binmask` y `cm_per_px`:

```
K_DEPTH = 0.25  (ratio profundidad/altura)
cm_per_resized_px = cm_per_px / scale_factor
vol = 0
para cada columna X de la máscara del barril:
    h_px = última fila − primera fila con mask
    h_cm = h_px × cm_per_resized_px
    a = h_cm / 2
    b = h_cm × K_DEPTH
    area = π × a × b
    vol += area × cm_per_resized_px
barril_volumen_litros = vol / 1000
```

Cada frame produce **su propio volumen** con **su propia escala**.

### 5.9 Validaciones

Flags devueltos:
- `within_rectangle`: el cruce bbox-piso es válido
- `silueta_bottom_used`: se ajustó ry2 con silueta
- `barril_top_used`: se ajustó ry1 con barril
- `bbox_aligned_with_floor`: `|y2 - floor_at_cow_cx| ≤ 3% × bbox_h`
- `y_diff_floor`: diferencia en px (para debug)

## 6. "Detectar pasadas" (frontend — passing loop)

Endpoint: `POST /detect_cow_fast` por cada frame.

Flow (`processPassingLoop` en `engine.js`):

1. Configurable: intervalo entre frames (default = fps/10 = ~10 muestras/s),
   rango `[start_frame, end_frame]`
2. Seek + capture sequential, fetch concurrente (MAX_CONCURRENT = 2)
3. Por frame válido (`within_rectangle = true`):
   - Guarda `animal_bbox_original`, `altura_cm`, `barril_volumen_litros`,
     `silueta_bottom_used`, `bbox_aligned_with_floor`, etc.
   - Renderiza thumbnail client-side con bbox verde, punto de cruce amarillo,
     label "XXX cm"
   - Agrega al gallery (2 columnas, border verde)

### 6.1 Criterio de inclusión al promedio

Al terminar la pasada:
- `silueta_bottom_used = true` → entra directo (bottom del bbox = pezuña real)
- `bbox_aligned_with_floor = true` (sin silueta, pero bbox pegado al piso)
  → rescatado (border verde, label "rescatado")
- Ninguno → descartado del promedio (border naranja, label explicativo)

### 6.2 Agregaciones

```
altura_promedio = mean(altura_cm para frames counted_in_avg)
barril_promedio = mean(barril_volumen_litros para frames counted_in_avg
                        donde barril > 0)
```

## 7. Descarga de resultado (PNG)

Endpoint: `POST /generate_result_card`

Input: frame blob + cow_name + altura_cm + barril_L + locked_reference

Pipeline:
1. Detecta vaca en el frame → crop
2. Corre `silueta_seg.pt` → overlay azul claro con contorno
3. Corre `barril_seg.pt` → overlay naranja con contorno
4. Compone PNG 800+ px de ancho con:
   - Header oscuro: nombre de la vaca + altura promedio + volumen barril
   - 2 thumbnails lado a lado (silueta | barril)
   - Footer: frame n + cantidad de mediciones

Frontend trigger: selecciona frame con altura más cercana al promedio → seek →
captura → POST → descarga automática `resultado_<cow_name>.png`.

## 8. Generación del modelo 3D interactivo

Endpoint: `POST /generate_3d_from_frame`

Input: **un solo frame representativo** (altura más cercana al promedio) +
cow_name + altura + barril_L + locked_reference.

Pipeline:
1. Detecta vaca → crop
2. `silueta_seg.pt` → máscara completa del cuerpo
3. Calcula `cm_per_px` de ese frame (mismo método que en detect_cow_fast)
4. Extrae contorno exterior de la máscara (`cv2.findContours` + `approxPolyDP` simplificado)
5. Convierte puntos de px → cm usando `cm_per_px`
6. Centra en x=0, y=0 al pie de la silueta
7. Usa `generar_ply_volumen.rebanadas_desde_contorno` (80 rebanadas elípticas)
   + `malla_elipsoidal` (32 vertices por rebanada) para generar mesh 3D
8. Guarda en `output_modelos3d_live/<cow_name>/`:
   - `<cow_name>_3d.ply`
   - `<cow_name>_volumen.ply`
   - `<cow_name>_resumen.json` con altura, escala, metadatos

Frontend:
1. Llama `window.loadModelosDisponibles(cow_name)` → refresca el dropdown del
   viewer 3D y auto-selecciona la vaca recién creada
2. Scroll al `#viewer3dCard`

El viewer 3D lee `output_modelos3d_Recorte26marz_altdiag/` **y** `output_modelos3d_live/`.

## 9. Limitaciones conocidas al 2026-04-22

1. **El volumen del barril es un promedio de N volúmenes independientes**.
   Cada frame calcula su barril con su propia escala; al final se promedia.
   Ventaja: simple, da std para diagnóstico. Desventaja: frames ruidosos
   contribuyen parcialmente. Para la mayoría de casos es suficiente porque el
   barril (torso) no varía mucho con movimiento de cabeza/patas.

2. **El modelo 3D se construye desde UN SOLO frame** (el representativo).
   No combina los N frames. La forma 3D está limitada a lo que ese frame captura.

3. **Vacas oscuras**: cuando `silueta_seg`/`barril_seg` fallan, YOLO puede dejar
   bbox extendido hacia abajo → descartado del promedio por `bbox_aligned_with_floor`
   falso.

4. **Postes muy cercanos horizontalmente**: si su separación en imagen es chica,
   la interpolación `t` es muy sensible. No se usa para el cruce geométrico
   (que depende de Y, no X), pero afecta la escala interpolada.

## 10. Próximo paso planificado: "Barril consenso" (multi-frame)

Idea:
- Cada frame ya calcula su silueta/barril en cm (con su escala propia)
- Combinar N contornos de barril en uno solo (alineados por pie y centro X)
- Tomar mediana por columna X → contorno consenso
- Rebanadas elípticas del contorno consenso → UN volumen, UN mesh 3D

Ventajas:
- Robustez estadística a nivel contorno, no volumen
- El PLY integra todos los frames válidos

Esto reemplazaría el "barril promedio" actual Y el "modelo 3D desde un frame".

## 11. Endpoints clave

| Endpoint | Propósito |
|---|---|
| `POST /calibrate_frame` | Detecta postes + piso, devuelve rectangle_ref |
| `POST /lock_reference` | Fija la ref para un video_id |
| `POST /detect_cow_fast` | Por-frame: bbox + altura + barril_vol con ref fija |
| `POST /analyze_frame` | Análisis manual completo (cow + keypoints + peso + altura) |
| `POST /generate_result_card` | Genera PNG descargable con nombre + altura + barril + silueta/barril thumbnails |
| `POST /generate_3d_from_frame` | Genera PLYs para el viewer 3D live desde un frame |

## 12. Archivos de modelo usados

- `models_yolo/cow.pt` — detección + keypoints de vaca
- `models_yolo/sticker.pt` — detección de postes
- `yolov8n.pt` — fallback COCO (class 19 = cow)
- `yolov8s-seg.pt` — segmentación genérica (usado en `_detect_all_cows`)
- `silueta_seg.pt` — silueta completa de la vaca (entrenado custom)
- `barril_seg.pt` — torso/barril sin cabeza (entrenado custom)

## 13. Archivos de código

- `app.py` → endpoints Flask, globals, modelos cargados al boot
- `weight_estimation.py` → `WeightEstimator`, `_detect_all_cows`, `_load_and_resize`
- `depth_estimation.py` → detección de postes
- `generar_ply_volumen.py` → rebanadas elípticas + malla + escribir PLY
- `static/js/engine.js` → UI principal, passing detection, descarga, 3D-from-pass
- `static/js/viewer3d.js` → viewer 3D interactivo (three.js)
- `templates/index.html` → UI

## 14. Datos almacenados

- `alturas_individuos.json`:
  - `alturas_VisualA_cm`: alturas medidas por este sistema
  - `pesos_VisualA_kg`: pesos de balanza (referencia real)
  - Secciones históricas: `alturas_Recorte26marz_cm`, `alturas_desfile26marz_cm`, etc.

- `output_modelos3d_VisualA/`: modelos 3D batch del script `generar_modelos3d_grandes.py`
- `output_modelos3d_live/`: modelos 3D generados en vivo desde pasadas
- `_locked_references` (en memoria): video_id → post1/post2 + original_coords
