## Reconocimiento facial de vacas (simple, funcional)

Sistema de ejemplo para **identificar 10 vacas** a partir de fotos o videos usando **redes convolutivas** (CNN) mediante **fine-tuning** de una ResNet (transfer learning). Incluye:

- Entrenamiento con tu dataset (10 animales)
- Inferencia en imagen
- Inferencia en video (muestreo de frames + votación)
- UI web local para subir foto/video y ver el resultado

> Nota importante: para que sea “facial” de verdad, el dataset debería estar **centrado en la cara** de la vaca (recortes consistentes). Este proyecto asume imágenes ya enfocadas en la cara o, al menos, que la vaca ocupa la mayor parte del encuadre.

---

## Estructura del dataset (múltiples establecimientos)

El sistema soporta **múltiples establecimientos**, cada uno con su propio dataset y modelo entrenado.

### Estructura recomendada:

```
data/
  establecimiento_01/
    cows/
      cow_01/
        img001.jpg
        img002.jpg
        ...
      cow_02/
        ...
      ...
      cow_10/
        ...
  establecimiento_02/
    cows/
      cow_01/
        ...
      ...
artifacts/
  establecimiento_01/
    model.pt
    classes.json
    config.json
  establecimiento_02/
    model.pt
    classes.json
    config.json
```

### Estructura simple (un solo establecimiento):

Si solo tenés un establecimiento, podés usar:

```
data/
  cows/
    cow_01/
    cow_02/
    ...
artifacts/
  establecimiento_01/  # o cualquier nombre
    model.pt
    classes.json
    config.json
```

**Recomendación mínima por clase**: **30–60 imágenes por vaca** (mejor si tienes 100+). Variar iluminación, ángulos y distancia.

### Opción: Extraer frames de videos

Si tenés videos de las vacas, podés extraer frames automáticamente:

**Opción 1: Videos en carpetas separadas**
```bash
# Estructura: data/videos/cow_01/video1.mp4, data/videos/cow_02/video2.mp4, etc.
python extract_frames.py \
  --videos_dir data/videos \
  --output_dir data/cows \
  --stride 30 \
  --max_frames_per_video 50
```

**Opción 2: Videos directamente en las carpetas de vacas**
```bash
# Estructura: data/cows/cow_01/video1.mp4, data/cows/cow_02/video2.mp4, etc.
python extract_frames.py \
  --videos_dir data/cows \
  --output_dir data/cows \
  --stride 30 \
  --max_frames_per_video 50 \
  --extract_from_same_dir
```

**Opción 3: Un solo video para una vaca específica**
```bash
python extract_frames.py \
  --video_path ruta/al/video.mp4 \
  --cow_class cow_01 \
  --output_dir data/cows \
  --stride 30
```

#### Filtrado de frames por detección de animales (YOLO)

Para mejorar la calidad del dataset, podés filtrar frames que no contengan animales detectados. Esto evita entrenar con frames donde el animal no está visible o está muy lejos.

**El sistema usa YOLOv8 pre-entrenado** que puede detectar:
- Vacas específicamente (clase "cow")
- Otros animales: perros, gatos, caballos, ovejas, etc.

**Filtrado básico (solo requiere detectar algún animal):**
```bash
python extract_frames.py \
  --videos_dir data/videos \
  --output_dir data/cows \
  --stride 30 \
  --filter_faces
```

**Filtrado específico para vacas:**
```bash
# Solo acepta frames donde se detecte específicamente una vaca
python extract_frames.py \
  --videos_dir data/videos \
  --output_dir data/cows \
  --stride 30 \
  --filter_faces \
  --require_features cow
```

**Opciones disponibles:**
- `cow`: Solo acepta frames con vacas detectadas específicamente
- `animal`: Acepta cualquier animal detectado (vaca, perro, gato, caballo, etc.)

Si no especificás `--require_features`, se acepta cualquier animal detectado.

> **Nota:** YOLO es mucho más preciso que Haar Cascades para detectar animales. El modelo se descarga automáticamente la primera vez que lo usás (aprox. 6MB para YOLOv8n). Si un frame válido no se detecta, podés agregarlo manualmente al dataset.

Parámetros:
- `--stride`: cada cuántos frames extraer (30 = cada 1 segundo en video 30fps)
- `--max_frames_per_video`: máximo frames por video (evita datasets gigantes)

---

## Instalación

Desde la raíz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Entrenamiento

Entrena un modelo **por establecimiento**. Cada establecimiento necesita su propio entrenamiento:

```bash
# Entrenar establecimiento_01
python -m src.train \
  --data_dir data/establecimiento_01/cows \
  --artifacts_dir artifacts/establecimiento_01 \
  --epochs 10 \
  --batch_size 16 \
  --img_size 224

# Entrenar establecimiento_02
python -m src.train \
  --data_dir data/establecimiento_02/cows \
  --artifacts_dir artifacts/establecimiento_02 \
  --epochs 10 \
  --batch_size 16 \
  --img_size 224
```

**Estructura simple (un solo establecimiento):**

```bash
python -m src.train \
  --data_dir data/cows \
  --artifacts_dir artifacts/establecimiento_01 \
  --epochs 10 \
  --batch_size 16
```

Al finalizar crea en `artifacts/[nombre_establecimiento]/`:

- `model.pt` (modelo entrenado)
- `classes.json` (lista de clases/vacas)
- `config.json` (configuración del entrenamiento)

---

## UI (subir foto o video)

```bash
streamlit run streamlit_app.py
```

La UI te permite:

1. **Seleccionar el establecimiento/dataset** (dropdown con todos los modelos entrenados)
2. **Subir foto o video** de una vaca
3. Ver el resultado:
   - **Vaca predicha** (del dataset seleccionado)
   - **Confianza**
   - **Desconocida** si la confianza cae bajo un umbral (configurable)

**Nota**: La UI busca automáticamente todos los establecimientos entrenados en la carpeta `artifacts/` y te permite elegir en cuál buscar.

---

## “Desconocida” (fuera del dataset)

Este ejemplo usa un criterio simple:

- Si `max_softmax < threshold` → **Desconocida**

Para robustez real, suele ser mejor usar **embeddings** + distancia (metric learning), pero esto cumple tu requisito de “simple y funcional”.


