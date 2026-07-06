# models_yolo/

Modelos YOLO pre-entrenados usados para detección y segmentación en el pipeline de estimación de peso.

## Modelos

| Archivo       | Tarea                         | Tipo de salida                     | Usado por                  |
|---------------|-------------------------------|------------------------------------|----------------------------|
| `cow.pt`      | Detección de cuerpo de vaca   | Bounding box + 9 keypoints        | `weight_estimation.py`     |
| `eye.pt`      | Segmentación de ojos          | Máscaras de instancia + bbox      | `weight_estimation.py`     |
| `sticker.pt`  | Detección de postes rojos     | Bounding boxes                     | `depth_estimation.py`      |

## Detalles por modelo

### `cow.pt` — Detección de cuerpo + keypoints

Detecta el cuerpo completo de la vaca y predice 9 keypoints anatómicos:

| Keypoint | Nombre           | Uso en el cálculo                              |
|----------|------------------|-------------------------------------------------|
| KP0      | head/poll        | (no usado para peso)                            |
| KP1      | pinbone (hip)    | dist1 = KP1 → KP2 (Body Length)                |
| KP2      | shoulderbone     | dist1 = KP1 → KP2 (Body Length)                |
| KP3      | girth bottom     | dist2 = KP3 → KP4 (Girth Vertical)             |
| KP4      | girth top/withers| dist2 = KP3 → KP4 (Girth Vertical)             |
| KP5      | leg/hoof         | (no usado para peso)                            |
| KP6      | back/topline     | (no usado para peso)                            |
| KP7      | belly midpoint   | (no usado para peso)                            |
| KP8      | back near shoulder| (no usado para peso)                           |

Confianza mínima por keypoint: 0.3 (configurable con `MIN_KP_CONF` en `weight_estimation.py`).

### `eye.pt` — Segmentación de ojos

Modelo de segmentación de instancias que detecta ojos de ganado. Se ejecuta sobre un ROI recortado de la región de cabeza (50% superior del bbox del animal) para mayor precisión.

Clases del modelo:
- `Eye` — aceptada para cálculo de distancia inter-ocular
- `Nose` — rechazada (filtrada automáticamente)

La distancia entre los centros de dos ojos detectados se usa como referencia de escala (distancia real asumida: ~20 cm).

### `sticker.pt` — Detección de postes/stickers rojos

Detecta los postes con franjas rojas instalados en la manga de trabajo. La altura real de la franja roja (122 cm por defecto) permite calcular la escala cm/px.

Incluye validación por color: solo acepta detecciones que contengan suficiente rojo (HSV) dentro del bbox.

Si YOLO falla, `depth_estimation.py` tiene un fallback basado puramente en detección de color rojo + contornos.

## Umbrales de confianza recomendados

Los umbrales se configuran en `app.py` al inicializar `WeightEstimator`:

| Parámetro                 | Valor actual | Descripción                          |
|---------------------------|-------------|---------------------------------------|
| `YOLO_CONF`               | 0.05        | Conf base (muy permisivo)             |
| `EYE_CONF_MULTIPLIER`     | 0.1         | → eye_conf = 0.005                    |
| `KEYPOINT_CONF_MULTIPLIER`| 0.1         | → keypoint_conf = 0.005              |
| `YOLO_IOU`                | 0.45        | IoU para NMS                          |

Valores bajos maximizan detecciones a costa de más falsos positivos (filtrados luego por validaciones de tamaño, posición y alineación).

## Reentrenamiento / actualización

Para actualizar un modelo:

1. Entrenar con [Ultralytics YOLO](https://docs.ultralytics.com/modes/train/)
2. Reemplazar el archivo `.pt` correspondiente en esta carpeta
3. Reiniciar la aplicación Flask

Los modelos se cargan una sola vez al iniciar `app.py` y se reutilizan para todas las peticiones.
