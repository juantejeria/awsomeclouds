# keras_vggface/

Fork local del paquete [keras-vggface](https://github.com/rcmalli/keras-vggface) adaptado para compatibilidad con TensorFlow 2.x.

## Por qué es un fork local

El paquete original `keras-vggface` fue diseñado para Keras 1.x / TensorFlow 1.x y tiene problemas de compatibilidad con versiones modernas de TensorFlow. Este fork local resuelve esos problemas sin depender de una versión PyPI que puede no estar actualizada.

## Arquitecturas soportadas

| Modelo    | Función       | Clases originales | Uso típico           |
|-----------|---------------|-------------------|----------------------|
| VGG16     | `VGG16()`     | 2622              | version=1 en testing |
| ResNet50  | `RESNET50()`  | 8631              | version=2 en testing |
| SENet50   | `SENET50()`   | 8631              | version=2 en testing |

## Archivos

| Archivo       | Descripción                                                     |
|---------------|-----------------------------------------------------------------|
| `__init__.py`  | Exporta `VGGFace` y `__version__`                              |
| `vggface.py`   | Función `VGGFace()`: punto de entrada para instanciar modelos  |
| `models.py`    | Definición de las arquitecturas VGG16, RESNET50, SENET50       |
| `utils.py`     | `preprocess_input()`: preprocesamiento según versión (1 o 2)   |
| `version.py`   | Versión del paquete                                             |

## Uso desde el proyecto

### En `training.py` (entrenamiento)

```python
from keras_vggface.vggface import VGGFace

# Cargar backbone sin top, sin pesos pre-entrenados
vggface = VGGFace(include_top=False, model='resnet50',
                  input_shape=(224, 224, 3), weights=None)
```

### En `testing.py` (inferencia)

```python
from keras_vggface import utils

# Preprocesar imagen según la versión del modelo
img = utils.preprocess_input(img, version=2)  # version=2 para ResNet50/SENet50
# version=1 para VGG16
```

## Preprocesamiento

- **version=1** (VGG16): sustrae media por canal BGR [93.5940, 104.7624, 129.1863]
- **version=2** (ResNet50/SENet50): sustrae media por canal RGB [91.4953, 103.8827, 131.0912]
