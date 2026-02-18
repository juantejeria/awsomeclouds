# checkpoints/

Modelos de reconocimiento entrenados, organizados por granja.

## Estructura

```
checkpoints/
└── Productor A/
    ├── chckpt.best.h5    # Modelo Keras (VGGFace + capas de clasificación)
    └── labels.json        # Mapeo clase → índice
```

### `chckpt.best.h5`

Modelo completo de Keras guardado con `model.save()`. Incluye arquitectura, pesos y estado del optimizador.

Arquitectura típica: VGGFace (ResNet50 sin top) + Flatten + Dense(512) + Dense(256) + Dense(128) + Dense(N, softmax).

### `labels.json`

Mapeo entre el nombre de la carpeta del dataset y el índice numérico de la clase:

```json
{
    "animal_1": 0,
    "animal_2": 1,
    "animal_3": 2,
    "animal_4": 3,
    "animal_5": 4,
    "animal_6": 5
}
```

## Cómo se genera un checkpoint

1. Preparar imágenes en `dataset/animal_N/`
2. Ejecutar el entrenamiento:
   ```bash
   python training.py --granja "Mi Granja" --model resnet50 --epochs 30 --batch_size 16
   ```
3. `training.py` crea automáticamente:
   - `checkpoints/Mi Granja/chckpt.best.h5` (mejor modelo por loss)
   - `checkpoints/Mi Granja/labels.json` (clases del dataset)

## Cómo agregar una nueva granja

1. Entrenar un modelo nuevo con `training.py` especificando el nombre de la granja
2. La carpeta se crea automáticamente
3. Al reiniciar `app.py`, la nueva granja aparece en el selector (`chooser.html`)

Solo las granjas que contienen un archivo `.h5` se muestran en la interfaz.

## Notas

- El modelo se carga una vez al seleccionar la granja (`/load_model`) y se mantiene en memoria
- Para cambiar de granja, el usuario debe volver a la página principal (`/`)
- Los logs de TensorBoard del entrenamiento se guardan en `logs/` (no en esta carpeta)
