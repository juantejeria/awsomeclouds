# Flujo de generación de volúmenes — webapp → código

Paso a paso del proceso real, indicando para cada acción de la UI qué ruta de Flask la atiende, qué función del backend hace el trabajo, qué entra y qué sale.

Convención de citas: `archivo:línea` → archivos viven en `/Users/usuario/JuanVaca/cattle-recognition/`.

---

## 0. Constantes globales que usa todo el flujo

- `SAVED_FRAMES_DATASET` (`app.py:131`) — nombre de la carpeta destino dentro de `checkpoints/`. Default `6mayo`; se sobreescribe con env var:
  ```
  SAVED_FRAMES_DATASET=20mayo ./run_app.sh 5001
  ```
- `MODELO_LIVE_DIR = 'output_modelos3d_6mayo'` (`app.py:3000`) — carpeta donde aterrizan los modelos 3D generados desde la UI.
- `TAPE_CM = 110.0` — alto de la cinta roja en los postes (dataset 20mayo). Está hardcoded dentro de `detect_reference_points` (`app.py:541`).
- `K_DEPTH = 0.25` y `N_SLICES = 80`, `N_VERTICES = 40` (`generar_ply_volumen.py:24-30`) — profundidad relativa del barril y resolución del mallado elipsoidal.
- Modelos YOLO cargados al arranque: `barril_seg.pt`, `silueta_seg.pt`, `yolov8n.pt` (clase 19 = cattle).

---

## 1. Subir video

**UI:** input `#videoUpload` en `templates/index.html`.

**Ruta Flask:** `POST /predict_video` (`app.py:424`) → `predict_video()` (`app.py:425-539`).

Lo que hace:

1. Guarda el archivo con `get_file_path_and_save(request, is_video=True)` (`app.py:261`) en `static/videos/`.
2. Instancia `VideoProcessor` (`video_processor.py`) con el YOLO COCO + `weight_estimator`.
3. Itera frames con `sample_rate` configurable y devuelve un JSON con `cows` (detecciones agregadas) y `stats`.

El video queda referenciado por `video_id` para el resto de la sesión; las rutas posteriores (lock_reference, save_frames_around) lo usan como clave.

---

## 2. Analizar frame — extraer la vaca y los postes

Esto es la inspección puntual sobre un frame seleccionado.

### 2.a Escaneo del frame
**Ruta Flask:** `POST /scan_frame` (`app.py:2300`) → `scan_frame()` (`app.py:2301-2337`).

- Cachea el frame en memoria con UUID (TTL 300 s) vía `_cache_frame()` (`app.py:133`).
- Llama `weight_estimator.scan_detections()` para devolver lista de vacas y de postes detectados, con thumbnails.
- Devuelve `frame_image_id`, que es lo que usa el siguiente paso para no reenviar la imagen.

### 2.b Analizar la vaca y postes elegidos
**Ruta Flask:** `POST /analyze_frame` (`app.py:2339`) → `analyze_frame()` (`app.py:2340-2498`).

- Recibe `frame_image_id` + `cow_index` + `post_indices`.
- Llama `estimate_weight(..., return_eye_coords=True, return_keypoint_coords=True)` definido en `weight_estimation.py`.
- Devuelve: peso, altura_cm, keypoints (ojos, garras), `cm_per_px` y un `rectangle_ref` con la escala lista para fijar.

---

## 3. Calibrar postes (solo puntos de referencia)

Para el flujo de "solo calibrar postes" (sin re-detectar vacas en cada calibración).

### 3.a Detectar postes en el frame
**Ruta Flask:** `POST /detect_reference_points` (`app.py:541`) → `detect_reference_points()` (`app.py:542-785`).

Lo que hace:
- Usa `depth_estimator.detect_postes()` (en `depth_estimation.py`) para localizar los postes verticales.
- En cada poste detecta la cinta roja (110 cm) y mide su píxeles → `tape_px`.
- Estima dónde está el piso bajo cada poste y construye un rectángulo amarillo anotado.
- Devuelve `poste1` y `poste2` con `(cx, top_tape, floor, tape_px)` y la escala cm/px efectiva en cada lado de la imagen.

### 3.b Versión rápida (solo calibrar)
**Ruta Flask:** `POST /calibrate_frame` (`app.py:838`) → `calibrate_frame()` (`app.py:839-1002`).

Igual que 3.a pero saltándose la detección de la vaca; ideal cuando ya sabes qué postes seleccionar. Devuelve el `rectangle_ref` listo para fijar.

### 3.c Fijar la referencia para todo el video
**Ruta Flask:** `POST /lock_reference` (`app.py:795`) → `lock_reference()` (`app.py:796-819`).

Guarda el rectángulo en el diccionario global `_locked_references[video_id]`. Desde aquí, cualquier otra ruta (detectar pasada, guardar 21 frames, generar 3D) reusa esa escala sin reprocesar postes.

Rutas auxiliares: `GET /get_reference/<video_id>` (`app.py:822`) y `POST/DELETE /clear_reference/<video_id>` (`app.py:830`).

---

## 4. Detectar pasada (promedio de alturas)

Loop en el cliente que llama frame por frame.

**Ruta Flask:** `POST /detect_cow_fast` (`app.py:1833`) → `detect_cow_fast()` (`app.py:1834-1970`).

Por frame:
1. Detecta vaca rápida (sin keypoints).
2. Aplica la `locked_reference` (interpolación lineal entre poste1 y poste2 según `x` de la vaca) para sacar `cm_per_px`.
3. Devuelve `cow_height_cm`, bbox y `cm_per_px`.

El JS acumula esas alturas y calcula el promedio de la pasada, que es el que usarás como `altura_cm` autoritativa para el modelo 3D.

---

## 5. Guardar 21 frames (central ± 10)

**Ruta Flask:** `POST /save_frames_around` (`app.py:1704`) → `save_frames_around()` (`app.py:1705-1830`).

Dos modos:
- **Backend (preferido):** recibe el video multipart + `central_frame` + `fps`, abre con `cv2.VideoCapture`, extrae 21 frames con calidad JPEG 95.
- **Legacy:** recibe los 21 frames ya decodificados desde el canvas del navegador (calidad menor).

Destino: `checkpoints/<SAVED_FRAMES_DATASET>/<timestamp_o_nombre>/`, con nombres `frame_m10_…`, …, `frame_0_…`, …, `frame_p10_…`, y un `context.json` con la `locked_reference` para que cualquier script offline pueda reusar la escala.

Listado de carpetas guardadas: `GET /list_saved_folders` (`app.py:1559`); listado de frames dentro de una carpeta: `POST /list_saved_frames` (`app.py:1597`).

---

## 6. Generar máscara y modelo 3D desde los 21 frames

### 6.a Modelo por frame
**Ruta Flask:** `POST /generate_3d_from_frame` (`app.py:1171`) → `generate_3d_from_frame()` (`app.py:1172-1419`).

Pipeline interno (mismo bloque que usa `diagnostico_barril_batch_22abril.py` para validar offline):

1. **Detección de vaca**: YOLO COCO clase 19, conf ≥ 0.2.
2. **Máscara de barril**: `barril_seg.pt` (conf 0.25) sobre el crop con padding. Función ayudante `_reparar_mascara_oclusion()` (`app.py:161`) repara las verticales que cortan el barril (postes que ocluyen el torso).
3. **Máscara de silueta** (para pezuñas): `silueta_seg.pt`.
4. **Escala cm/px**: interpolación lineal del `locked_reference` según la posición `x` del centroide de la vaca.
5. **Contorno 2D**: muestreo del borde de la máscara del barril + grid interior y triangulación Delaunay (función `triangular` en `generar_modelos3d_grandes.py:781`).
6. **Generación de los 3 PLY** vía `guardar_ply()` (`generar_modelos3d_grandes.py:848`):
   - `_3d.ply` — silueta espejada (front/back simétricos).
   - `_lateral.ply` — contorno 2D plano.
   - `_volumen.ply` — rebanadas elípticas (ver 6.c).

Destino: `output_modelos3d_6mayo/<cow_name>/` (constante `MODELO_LIVE_DIR`, `app.py:3000`) + un JSON resumen con `altura_cm` y `vol_barril_litros`.

### 6.b Modelo de consenso multi-frame (los 21 frames juntos)
**Ruta Flask:** `POST /generate_3d_consensus` (`app.py:1004`) → `generate_3d_consensus()` (`app.py:1005-1166`).

- Recibe N contornos normalizados (`barril_contour_norm`) de distintos frames de la pasada.
- Calcula la **mediana por posición** de los contornos → un contorno consenso, robusto a frames ruidosos.
- Llama el mismo bloque elipsoidal (6.c) sobre el contorno consenso y escribe `_volumen.ply` en `output_modelos3d_6mayo/<cow_name>/`.
- En el JSON queda `metodo: "consenso_multi_frame"` y `frames_usados: N`.

Este es el output que toma como bueno la app después de procesar los 21 frames.

### 6.c Construcción de la malla elipsoidal y el volumen (PLY)

Vive en `generar_ply_volumen.py` y se llama tanto desde 6.a como desde 6.b:

- `rebanadas_desde_contorno()` (`generar_ply_volumen.py:55`) — recorre el contorno de izquierda a derecha, lo parte en `N_SLICES = 80` rebanadas, y por cada rebanada calcula `(x, y_centro, h)`.
- `malla_elipsoidal()` (`generar_ply_volumen.py:81`) — para cada rebanada genera una elipse con `N_VERTICES = 40` vértices, ancho `h` y profundidad `K_DEPTH * h` (con `K_DEPTH = 0.25`). Une rebanadas consecutivas en triángulos.
- `escribir_ply()` (`generar_ply_volumen.py:142`) — vuelca a PLY ASCII.

El **volumen** se calcula por integración trapezoidal de las áreas elípticas a lo largo de `x`: `V = Σ (π · a_i · b_i) · Δx`, donde `a = h/2`, `b = K_DEPTH · h/2`. Se reporta en litros (cm³ / 1000).

La equivalente "offline" para una carpeta completa es `generar_ply_volumen.py:main()` (`:199`), que procesa todas las subcarpetas de `output_modelos3d_*/`.

---

## 7. Archivos PLY que quedan en disco

Para cada vaca/individuo procesado por la app aparece:

```
output_modelos3d_6mayo/<cow_name>/
  _3d.ply            ← silueta 3D espejada (vista artística)
  _lateral.ply       ← contorno 2D del barril (debug visual)
  _volumen.ply       ← malla elipsoidal usada para volumen
  _resumen.json      ← {altura_cm, vol_barril_litros, metodo, frames_usados, ...}
```

Cómo los expone la app a la UI:

- `GET /api/modelos_disponibles` (`app.py:3031`) — descubre todas las carpetas modelo (`_discover_modelo_dirs`, `app.py:3004`) y devuelve metadatos leídos por `_load_resumen` (`app.py:3022`).
- `GET /api/modelo3d/<vaca>/<archivo>` (`app.py:3161`) — sirve el PLY al viewer.
- `POST /api/modelo3d/<vaca>/recalcular` (`app.py:3092`) — recalcula el volumen con otro `K_DEPTH` o `N_SLICES` sin regenerar el contorno.

---

## 8. Diagnóstico

El diagnóstico hoy vive **offline** (CLI). No hay un botón Flask que lo dispare; la UI solo lista las carpetas para que tú lances el script desde shell.

### 8.a Diagnóstico por carpeta de 21 frames
`diagnostico_21frames_barril.py`:

```
python diagnostico_21frames_barril.py checkpoints/20mayo/<carpeta> [--out salida.png]
```

- `correr_barril_pipeline()` (línea 15) — replica exactamente lo que hace `/generate_3d_from_frame` (mismo bbox → crop+pad → `barril_seg`).
- Pinta cada frame con su máscara superpuesta (`overlay_mask`, línea 48) y los une en un grid (`make_tile`, línea 60). El PNG sale en `grids_21frames_20mayo/` (o el dataset que toque).

Sirve para detectar frames donde el barril sale truncado o donde el modelo agarra otra vaca.

### 8.b Diagnóstico batch sobre un dataset entero
`diagnostico_barril_batch_22abril.py`:

- Recorre todos los videos de `checkpoints/22abril/`, samplea a 15 fps y corre el mismo pipeline real.
- Marca como "truncado" todo frame donde la cobertura X de la máscara cae por debajo del 75 % del crop (`COBERTURA_THRESHOLD`).
- Salidas:
  - `checkpoints/22abril_diagnostico/<video>_grid.png` — grid con truncados + muestra de OK.
  - `checkpoints/22abril_diagnostico/resumen.csv` — una fila por frame con cobertura y estado.
- Variantes equivalentes para otros datasets están en `batch_14mayo_diagnostico.sh`, `batch_20mayo_diagnostico.sh`, `batch_6mayo_diagnostico.sh` y sus `*_v7.sh` (que cambian el `barril_seg.pt` por `barril_seg_v7.pt`).

### 8.c Diagnósticos auxiliares
- `diagnose_model.py` — valida el modelo de reconocimiento facial por granja (no toca volumen).
- `diagnose_weight.py` — compara peso real vs estimado contra un dataset etiquetado y emite CSV.
- `diagnostico_6mayo.csv` — tabla agregada `vaca | altura_real | altura_estimada | peso_real | peso_estimado | volumen_cm3` que vas alimentando.

---

## Resumen en una tabla

| # | Acción UI | Ruta Flask | Función backend | Archivo:línea | Salida |
|---|-----------|------------|-----------------|---------------|--------|
| 1 | Subir video | `POST /predict_video` | `predict_video` | `app.py:425` | `static/videos/...`, `video_id` |
| 2a | Escanear frame | `POST /scan_frame` | `scan_frame` | `app.py:2301` | `frame_image_id` + lista vacas/postes |
| 2b | Analizar frame | `POST /analyze_frame` | `analyze_frame` | `app.py:2340` | peso, altura, keypoints, `rectangle_ref` |
| 3a | Detectar postes | `POST /detect_reference_points` | `detect_reference_points` | `app.py:542` | `poste1`, `poste2` con escala |
| 3b | Calibrar (solo postes) | `POST /calibrate_frame` | `calibrate_frame` | `app.py:839` | `rectangle_ref` |
| 3c | Fijar referencia | `POST /lock_reference` | `lock_reference` | `app.py:796` | `_locked_references[video_id]` |
| 4 | Detectar pasada | `POST /detect_cow_fast` | `detect_cow_fast` | `app.py:1834` | `cow_height_cm` por frame (promedio en cliente) |
| 5 | Guardar 21 frames | `POST /save_frames_around` | `save_frames_around` | `app.py:1705` | `checkpoints/<dataset>/<folder>/*.jpg` + `context.json` |
| 6a | Generar 3D por frame | `POST /generate_3d_from_frame` | `generate_3d_from_frame` | `app.py:1172` | `_3d.ply`, `_lateral.ply`, `_volumen.ply` |
| 6b | Consenso 21 frames | `POST /generate_3d_consensus` | `generate_3d_consensus` | `app.py:1005` | `_volumen.ply` consenso + `_resumen.json` |
| 7 | Listar/servir PLY | `/api/modelos_disponibles`, `/api/modelo3d/<vaca>/<archivo>` | `modelos_disponibles`, `get_modelo_3d` | `app.py:3031`, `3161` | PLY al viewer |
| 8 | Diagnóstico | CLI (no hay ruta) | `diagnostico_21frames_barril.py`, `diagnostico_barril_batch_22abril.py` | scripts en raíz | grids PNG + CSV |
