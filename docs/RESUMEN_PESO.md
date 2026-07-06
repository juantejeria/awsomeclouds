# Resumen: Calculo de Peso y Screening

## Indice

1. [Arquitectura general](#arquitectura-general)
2. [Formula de peso (Schaeffer)](#formula-de-peso-schaeffer)
3. [Keypoints: los 4 puntos criticos](#keypoints-los-4-puntos-criticos)
4. [Conversion de pixeles a centimetros](#conversion-de-pixeles-a-centimetros)
5. [Coeficientes de correccion por raza/categoria/edad](#coeficientes-de-correccion)
6. [Flujo por endpoint (que pasa en cada "orden")](#flujo-por-endpoint)
7. [Batch Screening (procesamiento masivo)](#batch-screening)
8. [Resumen de librerias](#resumen-de-librerias)

---

## Arquitectura general

```
Imagen/Frame
    |
    v
[YOLO cow.pt] --> Deteccion de vaca (bbox + 9 keypoints)
    |
    +--> [YOLO eye.pt]     --> Ojos (escala por distancia inter-ocular)
    +--> [YOLO sticker.pt] --> Postes rojos (escala por altura conocida)
    |
    v
Escala: cm_per_px (cuantos cm mide 1 pixel)
    |
    v
Medidas fisicas: BL (largo cuerpo), GirthVert (alto torax) en cm
    |
    v
Formula Schaeffer --> raw_weight_kg
    |
    v
Correccion: raw_weight * K_breed * K_category * K_age --> peso final
```

---

## Formula de peso (Schaeffer)

El sistema usa una **formula de Schaeffer adaptada** para estimar el peso a partir de mediciones visuales.

### Caso 1: Con circunferencia toracica completa (Depth Anything V2 disponible)

Si el sistema puede estimar la circunferencia eliptica del torax usando profundidad monocular:

```
raw_weight_kg = (HG_cm^2 * BL_cm) / 10838
```

- **HG_cm** = Heart Girth = circunferencia toracica estimada (elipse con Ramanujan)
- **BL_cm** = Body Length = largo del cuerpo (KP1 pinbone -> KP2 shoulderbone)
- **10838** = 300 * 2.54^3 / 0.4536 (conversion metrica del divisor original)

> Codigo: `weight_estimation.py:1655-1659`

### Caso 2: Fallback con girth vertical (mas comun)

Cuando no hay profundidad disponible, usa directamente la medida vertical del torax:

```
raw_weight_kg = (BL_cm * GirthVert_cm^2 * 0.4536) / 300
```

- **GirthVert_cm** = distancia vertical del torax (KP3 girth bottom -> KP4 girth top)
- **0.4536** = conversion libras a kg
- **300** = divisor original de Schaeffer

> Codigo: `weight_estimation.py:1663`

### Peso final con correccion

```
weight_final = raw_weight * K_breed * K_category * K_age
```

> Codigo: `weight_estimation.py:1674` y `breed_coefficients.py:122-130`

---

## Keypoints: los 4 puntos criticos

El modelo `cow.pt` detecta 9 keypoints por animal. Para el calculo de peso se usan **4**:

| KP  | Nombre       | Ubicacion            | Uso                   |
|-----|-------------|----------------------|-----------------------|
| KP1 | Pinbone     | Cadera trasera       | Inicio de dist1 (BL)  |
| KP2 | Shoulderbone| Hombro delantero     | Fin de dist1 (BL)     |
| KP3 | Girth Bottom| Base del torax       | Inicio de dist2 (GV)  |
| KP4 | Girth Top   | Lomo/cruz (withers)  | Fin de dist2 (GV)     |

**Confianza minima**: cada keypoint debe tener `conf >= 0.3` (`MIN_KP_CONF`).

**Validacion adicional**: cada keypoint debe caer dentro del bounding box del animal con un margen de 15%.

> Codigo: `weight_estimation.py:1089-1150`

### Fallback por bounding box

Si no se detectan keypoints con suficiente confianza, el sistema estima las distancias a partir del bbox:

```python
dist1 = bbox_width  * 0.80   # ~80% del ancho = largo cuerpo
dist2 = bbox_height * 0.60   # ~60% del alto  = girth vertical
```

> Codigo: `weight_estimation.py:1205-1209`

### Recuperacion de keypoints en dos etapas

Si la vaca seleccionada no tiene keypoints, el sistema recorta la region del animal (con 25% de margen) y vuelve a correr `cow.pt` sobre el crop para intentar recuperarlos.

> Codigo: `weight_estimation.py:799-842`

---

## Conversion de pixeles a centimetros

El paso critico es calcular `cm_per_px`: cuantos centimetros representa un pixel en la imagen.

### Metodo 1: Ojos (`scale_method='eyes'`)

```
cm_per_px = 20.0 / distancia_inter_ocular_px
```

- Asume distancia entre ojos de vaca adulta = **20 cm** (constante)
- Requiere que `eye.pt` detecte ambos ojos con segmentacion

> Codigo: `weight_estimation.py:1490-1493`

### Metodo 2: Postes rojos (`scale_method='poste'`)

```
cm_per_px = 122.0 / altura_poste_px
```

- Asume postes de **122 cm** de altura (configurable por establecimiento)
- Detecta postes rojos por filtrado HSV (H: 0-15 y 165-180) + YOLO `sticker.pt`
- Si hay multiples postes, promedia sus alturas en pixeles

> Codigo: `weight_estimation.py:1382-1445`

### Metodo 3: Ambos (`scale_method='both'`, default)

Busca ojos primero (prioridad). Si no los encuentra, usa postes como fallback.

### Metodo 4: Override directo (batch mode)

En el screening batch, se pasa un `override_cm_per_px` precalibrado desde el frame de referencia, salteando toda la deteccion de postes/ojos.

> Codigo: `weight_estimation.py:1254-1255, 1382-1385`

### Correccion por profundidad (opcional)

Cuando hay 2+ postes y Depth Anything V2 esta cargado, el sistema corrige la escala segun la profundidad relativa del animal respecto a los postes.

---

## Coeficientes de correccion

Archivo: `breed_coefficients.py`

### Por raza (K_breed) - Referencia: Angus = 1.00

| Raza       | K     | | Raza      | K     |
|------------|-------|-|-----------|-------|
| Angus      | 1.00  | | Brangus   | 0.96  |
| Hereford   | 0.98  | | Bradford  | 0.95  |
| Shorthorn  | 0.96  | | Brahman   | 0.93  |
| Charolais  | 1.04  | | Nelore    | 0.91  |
| Limousin   | 0.98  | | Gyr       | 0.88  |
| Simmental  | 1.01  | | Holando   | 0.93  |

### Por categoria (K_category) - Referencia: Novillo = 1.00

| Categoria   | K     |
|-------------|-------|
| Ternero/a   | 0.84  |
| Recria      | 0.90  |
| Vaquillona  | 0.95  |
| Novillito   | 0.97  |
| Novillo     | 1.00  |
| Vaca        | 0.95  |
| Toro        | 1.08  |

### Por edad (K_age) - Referencia: 24-36 meses = 1.00

| Rango       | K     |
|-------------|-------|
| 0-6 meses   | 0.85  |
| 6-12 meses  | 0.92  |
| 12-18 meses | 0.96  |
| 18-24 meses | 0.98  |
| 24-36 meses | 1.00  |
| 36+ meses   | 1.00  |

### Rangos de peso esperado por categoria (para deteccion de outliers)

| Categoria   | Min (kg) | Max (kg) |
|-------------|----------|----------|
| Ternero/a   | 50       | 180      |
| Recria      | 120      | 250      |
| Vaquillona  | 170      | 350      |
| Novillito   | 180      | 400      |
| Novillo     | 300      | 550      |
| Vaca        | 300      | 600      |
| Toro        | 400      | 900      |

---

## Flujo por endpoint

### 1. `/predict` (imagen individual)

**Archivo**: `app.py:167-272`

```
1. Recibe imagen + opciones (scale_method, breed, category, age_range)
2. [Opcional] Reconocimiento facial del animal (CNN VGGFace/ResNet50)
3. weight_estimator.estimate_weight(img, scale_method, breed, category, age_range)
   a. Redimensiona imagen a 1040x640 (letterbox)
   b. Detecta vacas con cow.pt -> bboxes + keypoints
   c. Selecciona vaca (cow_index=0 por defecto)
   d. Extrae KP1-KP4, calcula dist1 y dist2 en pixeles
   e. Detecta ojos y/o postes para calcular cm_per_px
   f. Convierte dist1/dist2 a centimetros
   g. Aplica formula Schaeffer -> raw_weight
   h. Multiplica por coeficientes de raza/categoria/edad -> weight_final
4. Retorna JSON con peso, imagen anotada (base64), metadata
```

### 2. `/scan_frame` (Fase 1 del flujo en dos fases)

**Archivo**: `app.py:700-737`

```
1. Recibe un frame de video
2. Cachea el frame (para reutilizar en /analyze_frame)
3. weight_estimator.scan_detections(img)
   - Detecta TODAS las vacas (bboxes + thumbnails)
   - Detecta TODOS los postes rojos
   - NO calcula peso todavia
4. Retorna JSON con:
   - Lista de vacas encontradas (con thumbnails recortados)
   - Lista de postes encontrados
   - Imagen preview con anotaciones
5. El usuario selecciona QUE vaca y QUE postes usar
```

### 3. `/analyze_frame` (Fase 2 del flujo en dos fases)

**Archivo**: `app.py:739-876`

```
1. Recibe: frame_image_id (cacheado) + cow_index + post_indices + breed/category/age
2. Recupera frame del cache
3. weight_estimator.estimate_weight(
       img, cow_index=N, post_indices=[i,j],
       breed, category, age_range
   )
   - Calcula peso SOLO de la vaca seleccionada
   - Usa SOLO los postes seleccionados para la escala
4. Retorna JSON con peso estimado, detalles, imagen anotada
   - Incluye cm_per_px calibrado (que se usa luego en batch)
```

### 4. `/batch_screen` (Screening masivo)

**Archivo**: `app.py:878-1098`

```
1. Recibe: video + cm_per_px (pre-calibrado) + frame_interval + min_cow_score
           + breed/category/age + post_indices
2. Abre el video con OpenCV
3. Emite SSE evento "started" con metadata del video
4. Por cada frame (cada N frames segun frame_interval):
   a. Extrae frame, lo guarda como JPEG temporal
   b. Llama estimate_weight() con override_cm_per_px
      (saltea deteccion de postes/ojos, usa escala fija)
   c. Valida cow_score >= min_cow_score (sino: skip)
   d. Valida keypoints encontrados (sino: skip)
   e. Calcula peso y verifica si esta en rango (WEIGHT_RANGES)
   f. Genera thumbnail anotado (640px, JPEG 85%)
   g. Emite SSE evento "frame_result" o "frame_skip"
5. Al final: calcula estadisticas (media, mediana, desvio, min, max)
6. Emite SSE evento "complete" con resumen
```

### 5. `/predict_video` (video completo con tracking)

**Archivo**: `app.py:274-390`

```
1. Recibe video + opciones
2. VideoProcessor.process_video_simple():
   a. Itera frames a sample_rate dado
   b. Detecta vacas con cow.pt (bboxes)
   c. Tracking por IoU: asocia detecciones entre frames
   d. Por cada track (animal individual):
      - Recorta ROI expandido
      - Reconocimiento CNN (votacion por identidad)
      - estimate_weight() para frame de peso
   e. Agrega resultados por track:
      - Identidad: votacion ponderada
      - Peso: promedio de frames validos
3. Retorna JSON con resultados por animal
```

---

## Batch Screening

El screening es el flujo completo para procesar un video y obtener estadisticas de peso del rodeo:

```
[Calibracion]                         [Screening]
     |                                     |
  /scan_frame                        /batch_screen
     |                                     |
  Usuario ve vacas                   Por cada N frames:
  y postes detectados                  - estimate_weight()
     |                                   con cm_per_px fijo
  /analyze_frame                       - Valida keypoints
     |                                 - Clasifica in_range/outlier
  Obtiene cm_per_px     ---------->    |
  calibrado                          Resumen final:
                                       - Promedio/mediana
                                       - Desvio estandar
                                       - Cantidad validos/outliers
```

### Eventos SSE del screening

| Evento         | Cuando                                  | Datos principales                           |
|----------------|-----------------------------------------|---------------------------------------------|
| `started`      | Inicio del procesamiento                | total_frames, frames_to_process, fps        |
| `frame_result` | Frame procesado exitosamente            | weight_kg, in_range, cow_score, dist1/2_px  |
| `frame_skip`   | Frame descartado                        | frame_num, reason (low_cow_score, etc.)     |
| `complete`     | Fin del procesamiento                   | summary: avg, median, stdev, valid_count    |
| `error`        | Error fatal                             | message                                     |

---

## Resumen de librerias

### Librerias principales

| Libreria          | Version | Uso en el proyecto |
|-------------------|---------|-------------------|
| **Flask**         | 3.1.2   | Servidor web. Maneja todos los endpoints REST (`/predict`, `/scan_frame`, `/analyze_frame`, `/batch_screen`). Sirve templates HTML y streaming SSE. |
| **Ultralytics**   | 8.3.236 | Framework YOLO. Carga y ejecuta los 3 modelos de deteccion: `cow.pt` (vacas + keypoints), `eye.pt` (ojos), `sticker.pt` (postes rojos). Toda la deteccion pasa por aca. |
| **TensorFlow**    | 2.13.0  | Backend de deep learning para la CNN de reconocimiento facial. Ejecuta el modelo VGGFace (ResNet50) para identificar animales individuales. |
| **Keras**         | 2.13.1  | API de alto nivel sobre TensorFlow. Define y carga los modelos de reconocimiento (`chckpt.best.h5`). Se usa junto con `keras-vggface` (fork local). |
| **OpenCV**        | 4.8.1   | Procesamiento de imagen/video. Lee frames de video (`VideoCapture`), redimensiona, filtra por color HSV (deteccion de postes rojos), dibuja anotaciones, codifica JPEG. |
| **NumPy**         | 1.24.3  | Operaciones numericas. Manipulacion de arrays de keypoints, bboxes, mapas de profundidad. Calculos de distancia euclidiana y transformaciones de coordenadas. |
| **Pillow**        | 11.3.0  | Carga y conversion de imagenes. Lee imagenes desde archivos, convierte entre formatos (PIL <-> numpy), preprocesamiento antes de los modelos. |
| **PyTorch**       | 2.2.2   | Backend para Ultralytics YOLO y Depth Anything V2. Los modelos YOLO corren internamente sobre PyTorch. Tambien se usa para el pipeline de profundidad monocular. |
| **Transformers**  | 4.57.6  | HuggingFace Transformers. Carga el modelo Depth Anything V2 (`depth-anything/Depth-Anything-V2-Small-hf`) para estimacion de profundidad monocular. Usado para calcular circunferencia toracica eliptica. |

### Librerias de soporte

| Libreria          | Version | Uso en el proyecto |
|-------------------|---------|-------------------|
| **scikit-learn**  | 1.6.1   | Utilidades de ML (metricas, preprocesamiento). Usado en entrenamiento y evaluacion de la CNN de reconocimiento. |
| **scikit-image**  | 0.24.0  | Analisis de imagenes. SSIM para deteccion de imagenes duplicadas en el dataset (`ssim.py`). |
| **matplotlib**    | 3.9.4   | Graficos y visualizacion. Genera graficos del historial de entrenamiento, visualiza filtros de convolucion y Grad-CAM. |
| **h5py**          | 3.14.0  | Lectura/escritura de archivos HDF5. Carga los checkpoints de Keras (`.h5`) con los pesos de la CNN de reconocimiento. |
| **configparser**  | 7.2.0   | Lee `config.ini` con configuracion de la app (secret key, umbrales, rutas). |
| **polars**        | 1.36.1  | Dataframes rapidos. Procesamiento tabular de datos (potencialmente logs o datasets). |
| **Jinja2**        | 3.1.6   | Motor de templates HTML. Renderiza las paginas web (`base.html`, `index.html`, `chooser.html`). |
| **pyssim**        | 0.7.1   | Calculo de SSIM (Structural Similarity Index). Usado en `ssim.py` para detectar imagenes duplicadas o muy similares en el dataset. |

### Modelos YOLO (archivos `.pt`)

| Modelo        | Tamano | Funcion |
|---------------|--------|---------|
| `cow.pt`      | 22 MB  | Deteccion de vacas + 9 keypoints anatomicos. Es el modelo principal para la estimacion de peso. |
| `eye.pt`      | 5.9 MB | Segmentacion de ojos. Permite calcular la distancia inter-ocular para la escala cm/px. |
| `sticker.pt`  | 23 MB  | Deteccion de postes/stickers rojos de referencia. Alternativa a ojos para calibrar la escala. |
| `yolov8n.pt`  | 6.5 MB | Modelo COCO preentrenado (fallback). Detecta clase "cow" (id=19) cuando `cow.pt` falla. Se auto-descarga. |

### Fork local: `keras_vggface/`

Fork modificado de `keras-vggface` adaptado para TensorFlow 2.x. Provee las arquitecturas VGGFace (VGG16, ResNet50, SENet50) para reconocimiento facial de ganado. Se usa con `version=2` (ResNet50/SENet50).

---

## Archivos clave

| Archivo                  | Lineas | Responsabilidad |
|--------------------------|--------|-----------------|
| `weight_estimation.py`   | ~1750  | Motor de calculo de peso. Clase `WeightEstimator` con toda la logica: deteccion, keypoints, escala, formula, visualizacion. |
| `app.py`                 | ~1100  | Servidor Flask. Todos los endpoints, logica de request/response, streaming SSE, cache de frames. |
| `breed_coefficients.py`  | 131    | Coeficientes de correccion (raza, categoria, edad) y rangos de peso esperados. |
| `depth_estimation.py`    | ~700   | Deteccion de postes, calibracion de escala con postes, estimacion de profundidad. |
| `video_processor.py`     | ~1800  | Procesamiento de video: tracking por IoU, reconocimiento multi-animal, agregacion por track. |
| `testing.py`             | ~300   | Inferencia CNN: `ImageScore` que carga imagen, preprocesa y predice identidad con VGGFace. |
| `training.py`            | ~400   | Entrenamiento de la CNN de reconocimiento. Data augmentation, fine-tuning, guardado de checkpoints. |
