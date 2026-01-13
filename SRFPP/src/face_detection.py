"""
Detección de rostros/partes de animales en imágenes para filtrar frames.
Usa YOLOv8 pre-entrenado para detectar vacas y otros animales.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Lazy import de YOLO para evitar cargar el modelo si no se usa
_yolo_model = None


def _get_yolo_model(model_size: str = "nano"):
    """
    Obtiene el modelo YOLO (carga lazy, solo cuando se necesita).
    
    Args:
        model_size: Tamaño del modelo. Opciones: "nano" (más rápido), "small", "medium", "large"
    
    Returns:
        Modelo YOLO cargado
    """
    global _yolo_model
    
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            # YOLOv8n (nano) es rápido y suficiente para detección básica
            # Modelos disponibles: n (nano), s (small), m (medium), l (large), x (xlarge)
            model_name = f"yolov8{model_size[0]}.pt"  # "nano" -> "n", "small" -> "s", etc.
            _yolo_model = YOLO(model_name)
        except ImportError:
            raise ImportError(
                "ultralytics no está instalado. Instálalo con: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Error al cargar modelo YOLO: {e}")
    
    return _yolo_model


def detect_animal_face(image: Image.Image | np.ndarray, min_confidence: float = 0.3) -> bool:
    """
    Detecta si hay un animal (especialmente vaca) visible en la imagen usando YOLO.
    
    Args:
        image: Imagen PIL o numpy array
        min_confidence: Confianza mínima requerida para considerar una detección válida
    
    Returns:
        True si se detecta un animal (vaca, perro, gato, oveja, cabra, caballo), False en caso contrario
    """
    model = _get_yolo_model()
    
    # Convertir PIL a numpy si es necesario
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert('RGB'))
    else:
        img_array = image
    
    # YOLO espera imágenes en formato RGB
    if len(img_array.shape) == 2:
        # Si es escala de grises, convertir a RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Clases de animales en COCO dataset (YOLO pre-entrenado)
    # 16: dog, 17: horse, 18: sheep, 19: cow, 20: elephant, 21: bear, 22: zebra, 23: giraffe
    # También incluimos: 14: bird, 15: cat
    animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    # Realizar detección
    results = model(img_array, conf=min_confidence, verbose=False)
    
    # Verificar si hay alguna detección de animal
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            # Verificar si alguna detección es un animal
            if any(cls in animal_classes for cls in classes):
                return True
    
    return False


def detect_animal_features(image: Image.Image | np.ndarray, min_confidence: float = 0.3) -> dict[str, bool]:
    """
    Detecta características específicas del animal usando YOLO.
    
    Args:
        image: Imagen a analizar
        min_confidence: Confianza mínima requerida
    
    Returns:
        Diccionario con las características detectadas:
        - "cow": Se detectó una vaca
        - "animal": Se detectó cualquier animal
        - "face": Alias de "animal" (para compatibilidad con código anterior)
        - "eyes": Siempre False (YOLO no detecta partes específicas, solo objetos completos)
        - "profile": Siempre False (YOLO detecta el animal completo, no perfiles específicos)
    """
    model = _get_yolo_model()
    
    # Convertir PIL a numpy si es necesario
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert('RGB'))
    else:
        img_array = image
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Clases de animales
    animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # bird, cat, dog, horse, sheep, cow, etc.
    cow_class = 19  # Vaca específicamente
    
    features = {
        "cow": False,
        "animal": False,
        "face": False,  # Alias para compatibilidad
        "eyes": False,  # YOLO no detecta partes específicas
        "profile": False,  # YOLO detecta el animal completo
    }
    
    # Realizar detección
    results = model(img_array, conf=min_confidence, verbose=False)
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            # Verificar si hay alguna detección de animal
            if any(cls in animal_classes for cls in classes):
                features["animal"] = True
                features["face"] = True  # Alias para compatibilidad
            
            # Verificar específicamente vacas
            if cow_class in classes:
                features["cow"] = True
    
    return features


def has_valid_animal_frame(
    image: Image.Image | np.ndarray, 
    require_features: list[str] | None = None,
    min_confidence: float = 0.3,
) -> bool:
    """
    Verifica si un frame tiene características válidas de animal.
    
    Args:
        image: Imagen a verificar
        require_features: Lista de características requeridas. 
                         Si es None, solo requiere que haya algún animal detectado.
                         Opciones válidas: ["cow", "animal", "face"]
                         Nota: "eyes" y "profile" no están disponibles con YOLO
        min_confidence: Confianza mínima requerida para las detecciones
    
    Returns:
        True si el frame es válido para entrenamiento
    """
    if require_features is None or len(require_features) == 0:
        # Por defecto, solo verificar que haya algún animal
        return detect_animal_face(image, min_confidence=min_confidence)
    
    # Verificar características específicas
    features = detect_animal_features(image, min_confidence=min_confidence)
    
    # Mapear características para compatibilidad
    # "face" es un alias de "animal"
    feature_map = {
        "face": "animal",
        "eyes": "animal",  # Mapear a animal ya que YOLO no detecta ojos específicamente
        "profile": "animal",  # Mapear a animal ya que YOLO detecta el animal completo
    }
    
    # Verificar que todas las características solicitadas estén presentes
    for feat in require_features:
        mapped_feat = feature_map.get(feat, feat)
        if not features.get(mapped_feat, False):
            return False
    
    return True


def detect_cow_specifically(image: Image.Image | np.ndarray, min_confidence: float = 0.3) -> bool:
    """
    Detecta específicamente si hay una vaca en la imagen.
    
    Args:
        image: Imagen a analizar
        min_confidence: Confianza mínima requerida
    
    Returns:
        True si se detecta una vaca, False en caso contrario
    """
    features = detect_animal_features(image, min_confidence=min_confidence)
    return features.get("cow", False)
