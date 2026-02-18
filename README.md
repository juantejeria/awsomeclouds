# cattle-recognition

Sistema de **reconocimiento individual de ganado** y **estimación de peso en vivo** mediante Deep Learning, modelos YOLO y visión por computadora.

## Arquitectura General

```
                         ┌──────────────────────────────────────────┐
                         │            Flask Web App (app.py)        │
                         │   http://localhost:5001                  │
                         └────────┬───────────┬───────────┬────────┘
                                  │           │           │
                    ┌─────────────┘     ┌─────┘     ┌─────┘
                    ▼                   ▼           ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
            │  /predict     │  │/predict_video│  │/detect_reference │
            │  (imagen)     │  │  (video)     │  │  _points         │
            └──────┬───────┘  └──────┬───────┘  └──────┬───────────┘
                   │                 │                  │
          ┌────────┴────────┐  ┌─────┴──────┐    ┌─────┴──────┐
          ▼                 ▼  ▼            ▼    ▼            │
  ┌──────────────┐ ┌────────────┐ ┌──────────────┐ ┌──────────────┐
  │  testing.py  │ │weight_est. │ │video_proc.py │ │depth_est.py  │
  │  (CNN face)  │ │   .py      │ │  (tracking)  │ │  (postes)    │
  └──────┬───────┘ └──────┬─────┘ └──────┬───────┘ └──────────────┘
         │                │              │
         ▼                ▼              ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ keras_vggface│ │ YOLO models  │ │ breed_coeff. │
  │ (VGG/ResNet/ │ │ cow.pt       │ │   .py        │
  │  SENet)      │ │ eye.pt       │ │ (K_breed *   │
  │              │ │ sticker.pt   │ │  K_cat *     │
  └──────────────┘ └──────────────┘ │  K_age)      │
                                    └──────────────┘
```

## Requisitos

- **Python 3.9** (probado con CPython 3.9.x en macOS)
- TensorFlow / Keras (para reconocimiento facial de ganado)
- Ultralytics YOLO (para detección de cuerpo, ojos y postes)
- OpenCV, NumPy, scikit-learn, Pillow, Flask

### Instalación

```bash
# Clonar el repositorio
git clone <url-del-repo> cattle-recognition
cd cattle-recognition

# Crear entorno virtual
python3.9 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt   # si existe
# o manualmente:
pip install flask tensorflow ultralytics opencv-python-headless \
            numpy scikit-learn scikit-image pillow
```

## Configuración

### `config.ini`

```ini
[app]
app-secret-key=<clave-secreta-para-flask>

[detection]
confidence_threshold=0.5   # Umbral para considerar un animal "conocido"
```

### Modelos YOLO

Deben existir en `models_yolo/`:

| Archivo       | Función                              |
|---------------|--------------------------------------|
| `cow.pt`      | Detección de cuerpo + keypoints      |
| `eye.pt`      | Segmentación de ojos (instancias)    |
| `sticker.pt`  | Detección de postes/stickers rojos   |

### Checkpoints de reconocimiento

Cada granja necesita una carpeta en `checkpoints/`:

```
checkpoints/
  Productor A/
    chckpt.best.h5   # Modelo Keras entrenado
    labels.json       # {"animal_1": 0, "animal_2": 1, ...}
```

## Ejecución

```bash
# Opción 1: Script de arranque
./run_app.sh

# Opción 2: Directamente
source venv/bin/activate
python app.py
```

El servidor arranca en **http://localhost:5001**.

## Estructura del Proyecto

```
cattle-recognition/
├── app.py                      # Servidor Flask: endpoints de la API
├── weight_estimation.py        # Estimación de peso (Schaeffer adaptada)
├── depth_estimation.py         # Detección de postes y escala cm/px
├── video_processor.py          # Procesamiento de video con tracking IoU
├── testing.py                  # Carga de modelo e inferencia CNN
├── training.py                 # Entrenamiento de modelos VGGFace
├── breed_coefficients.py       # Multiplicadores por raza/categoría/edad
│
├── config.ini                  # Configuración de la app
├── run_app.sh                  # Script de arranque
│
├── models_yolo/                # Modelos YOLO pre-entrenados
│   ├── cow.pt                  #   Detección de cuerpo + 9 keypoints
│   ├── eye.pt                  #   Segmentación de ojos
│   └── sticker.pt              #   Detección de postes rojos
│
├── keras_vggface/              # Fork local de keras-vggface (TF2)
│   ├── __init__.py
│   ├── vggface.py              #   Punto de entrada VGGFace()
│   ├── models.py               #   Definición de VGG16/ResNet50/SENet50
│   ├── utils.py                #   Preprocesamiento de imágenes
│   └── version.py
│
├── checkpoints/                # Modelos entrenados por granja
│   └── Productor A/
│       ├── chckpt.best.h5
│       └── labels.json
│
├── dataset/                    # Imágenes de entrenamiento
│   ├── animal_1/               #   ~10-20 imágenes recortadas de la cara
│   ├── animal_2/
│   └── ...
│
├── templates/                  # Templates HTML (Jinja2)
│   ├── base.html               #   Template base (Bootstrap 4)
│   ├── chooser.html            #   Selector de granja
│   └── index.html              #   Interfaz principal
│
├── static/
│   ├── css/main.css            # Estilos
│   ├── js/engine.js            # Lógica del frontend
│   └── img/                    # Imágenes estáticas
│
├── diagnose_model.py           # Diagnóstico de precisión del modelo CNN
├── diagnose_weight.py          # Diagnóstico de logs de estimación de peso
├── grad_CAM.py                 # Visualización Grad-CAM
├── ssim.py                     # Limpieza de duplicados por SSIM
├── organize_dataset.py         # Organización de imágenes en clases
├── drawing.py                  # Grafos de arquitectura de red
└── convfilter_visualization.py # Visualización de filtros convolucionales
```

## Flujo de Datos

### Imagen: `/predict` (POST)

```
Imagen (JPG/PNG)
    │
    ├─► YOLO cow.pt  ──► bbox del animal + 9 keypoints
    │                         │
    │                         ├─► KP1 (pinbone) ─────┐
    │                         ├─► KP2 (shoulderbone) ─┤─► dist1 (Body Length)
    │                         ├─► KP3 (girth bottom) ─┤
    │                         └─► KP4 (girth top) ────┘─► dist2 (Girth Vertical)
    │
    ├─► YOLO eye.pt  ──► segmentación de ojos ──► dist_ojos (px)
    │                                                │
    │                                                └─► escala: 20cm / dist_ojos_px
    │
    ├─► YOLO sticker.pt ──► postes rojos ──► altura_px
    │                                            │
    │                                            └─► escala: 122cm / altura_px
    │
    ├─► CNN VGGFace (ResNet50) ──► reconocimiento: {animal_N: probabilidad}
    │
    └─► Fórmula de peso ──► peso_kg
```

### Video: `/predict_video` (POST)

```
Video (MP4/AVI/MOV)
    │
    └─► VideoProcessor.process_video_simple()
            │
            ├─► Frame N ──► YOLO cow.pt ──► bboxes
            │                   │
            │                   └─► Tracking IoU ──► cow_0, cow_1, ...
            │                         │
            │                         ├─► ROI expandido ──► CNN reconocimiento
            │                         │
            │                         └─► Frame completo ──► estimate_weight()
            │                               │
            │                               ├─► Calibración de altura (cm_per_px)
            │                               └─► Fallback: cow-height como "regla"
            │
            └─► Agregación por track_id
                    ├─► Votación ponderada (identidad)
                    ├─► Media/mediana de pesos
                    └─► Frames con peso (galería)
```

## Fórmula de Peso (Schaeffer Adaptada)

La estimación de peso se basa en la fórmula de Schaeffer modificada:

### Fórmula original de Schaeffer

```
Weight_lbs = (HG² × BL) / 300
```

Donde:
- **HG** = Heart Girth (circunferencia completa detrás de las patas delanteras)
- **BL** = Body Length (largo del cuerpo: hombro a cadera)

### Adaptación en este sistema

En nuestro caso, no medimos la circunferencia completa sino el **diámetro vertical** de la zona del girth:

```
raw_weight_kg = (BL × GirthVert² × lb) / 406
```

Donde:
- **BL** = `dist1cm` = distancia KP1 (pinbone) → KP2 (shoulderbone) en cm
- **GirthVert** = `dist2cm` = distancia KP3 (girth bottom) → KP4 (girth top) en cm
- **lb** = 0.45359237 (conversión libras → kg)
- **406** = divisor calibrado (calibrado con animal conocido de 446 kg)

### Conversión de píxeles a centímetros

La escala `cm/px` se obtiene de una referencia de tamaño conocido:

| Método      | Referencia                                     | Cálculo                            |
|-------------|------------------------------------------------|------------------------------------|
| **Ojos**    | Distancia inter-ocular del ganado (~20 cm)     | `cm_per_px = 20 / dist_ojos_px`   |
| **Poste**   | Franja roja del poste (122 cm de altura)       | `cm_per_px = 122 / altura_px`     |

### Corrección por raza, categoría y edad

El peso final se ajusta con multiplicadores (`breed_coefficients.py`):

```
peso_final = raw_weight × K_breed × K_category × K_age
```

Ejemplo: Brahman × Ternero × 0-6 meses = `0.93 × 0.84 × 0.85 = 0.664`

## Endpoints de la API

| Método | Ruta                     | Descripción                                           |
|--------|--------------------------|-------------------------------------------------------|
| GET    | `/`                      | Página principal: selector de granja                  |
| POST   | `/load_model`            | Carga el modelo CNN de la granja seleccionada         |
| POST   | `/predict`               | Procesa una imagen (reconocimiento + peso)            |
| POST   | `/predict_video`         | Procesa un video completo con tracking multi-vaca     |
| POST   | `/detect_reference_points` | Detecta postes rojos y muestra mediciones           |
| POST   | `/detect_posts_all`      | Detecta todos los postes del alambrado (experimental) |

### Parámetros de `/predict` y `/predict_video`

| Parámetro            | Tipo    | Default        | Descripción                                    |
|----------------------|---------|----------------|-------------------------------------------------|
| `file`               | File    | (requerido)    | Imagen o video a procesar                       |
| `enable_recognition` | string  | `"true"`       | Activar reconocimiento facial                   |
| `enable_weight`      | string  | `"true"`       | Activar estimación de peso                      |
| `scale_method`       | string  | `"both"`       | `"both"`, `"eyes"`, `"poste"`                   |
| `breed`              | string  | `"desconocido"`| Raza del animal (para corrección de peso)       |
| `category`           | string  | `"desconocido"`| Categoría (ternero, novillo, vaca, toro, etc.)  |
| `age_range`          | string  | `"desconocido"`| Rango de edad (`"0-6"`, `"6-12"`, ..., `"36+"`) |
| `sample_rate`        | int     | `1`            | (video) Procesar 1 frame cada N frames          |
| `debug`              | string  | `"false"`      | (video) Logs detallados en consola              |

## Entrenamiento de un modelo nuevo

```bash
# 1. Preparar dataset: imágenes recortadas de la cara en dataset/animal_N/
# 2. Entrenar (por defecto usa ResNet50, 20 epochs):
python training.py --granja "Mi Granja" --model resnet50 --epochs 30 --batch_size 16

# 3. El checkpoint se guarda en checkpoints/Mi Granja/chckpt.best.h5
#    El mapeo de clases se guarda en checkpoints/Mi Granja/labels.json
```

## Herramientas de Diagnóstico

```bash
# Verificar precisión del modelo CNN con el dataset
python diagnose_model.py "Productor A"

# Analizar logs de peso para entender qué puntos faltan
python diagnose_weight.py app.log

# Visualizar Grad-CAM (qué mira la red)
python grad_CAM.py --granja "Productor A" --img dataset/animal_1/foto.png --model resnet50

# Limpiar imágenes duplicadas del dataset por similitud SSIM
python ssim.py --dir dataset/animal_1 --threshold 0.95
```

## Troubleshooting

### El peso no se calcula

1. **Verificar que los 3 modelos YOLO existen** en `models_yolo/` (`cow.pt`, `eye.pt`, `sticker.pt`)
2. **Activar debug** en la UI al procesar video y revisar la consola del servidor
3. **Ejecutar `diagnose_weight.py`** para analizar los logs
4. Causas comunes:
   - No se detectan keypoints → la vaca no está de cuerpo entero en la imagen
   - No se detectan ojos → la vaca no está de perfil (vista lateral)
   - No se detectan postes → no hay postes rojos visibles o el modelo `sticker.pt` no los reconoce
   - Keypoints con confianza < 0.3 → mejorar calidad de imagen o iluminación

### El reconocimiento es incorrecto

1. **Verificar el dataset**: cada `animal_N/` debe tener ~10-20 imágenes variadas de la cara
2. **Reentrenar** con más epochs o más imágenes
3. **Ejecutar `diagnose_model.py`** para ver la precisión por animal
4. Si la confianza es siempre baja (< 50%), el modelo puede necesitar más datos de entrenamiento

### Puerto 5001 ocupado

macOS usa el puerto 5000 para AirPlay. El servidor usa 5001 por defecto. Si está ocupado, modificar la línea final de `app.py`:

```python
app.run(host='0.0.0.0', port=5002, debug=True)
```

### Error al cargar modelo

- Verificar que `checkpoints/<granja>/chckpt.best.h5` existe
- Verificar que `checkpoints/<granja>/labels.json` existe y tiene el formato correcto
- Verificar compatibilidad de la versión de TensorFlow/Keras con el checkpoint
