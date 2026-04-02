"""
Detección de rostros/partes de animales en imágenes para filtrar frames.

Incluye dos modelos YOLO:
  1. Modelo genérico (yolov8n.pt / COCO) para detectar cuerpos de animales.
  2. Modelo especializado de detección de rostros de vacas, entrenado con un
     dataset de Roboflow ("Cattle Face Detection"). Se carga desde
     ``cattle_face_detection/train_run/weights/best.pt`` (relativo al directorio
     del proyecto).

El pipeline de video utiliza primero el modelo de rostros y, si no encuentra
ninguno, cae al modelo genérico (body fallback).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Model caching: st.cache_resource en Streamlit, lru_cache fuera de él
# ---------------------------------------------------------------------------
try:
    from streamlit import cache_resource as _model_cache
except ImportError:
    from functools import lru_cache

    def _model_cache(func):
        return lru_cache(maxsize=1)(func)


# Ruta por defecto al modelo de rostros entrenado
_COW_FACE_WEIGHTS: Path = (
    Path(__file__).resolve().parent.parent
    / "cattle_face_detection" / "train_run" / "weights" / "best.pt"
)


@_model_cache
def _load_yolo_model(model_size: str = "nano"):
    """Carga el modelo YOLO genérico (cacheado, se ejecuta una sola vez)."""
    try:
        from ultralytics import YOLO

        return YOLO(f"yolov8{model_size[0]}.pt")
    except ImportError:
        raise ImportError(
            "ultralytics no está instalado. Instálalo con: pip install ultralytics"
        )


@_model_cache
def _load_cow_face_model(weights_path_str: str):
    """Carga el modelo YOLO de rostros de vacas (cacheado, se ejecuta una sola vez)."""
    try:
        from ultralytics import YOLO

        return YOLO(weights_path_str)
    except ImportError:
        raise ImportError(
            "ultralytics no está instalado. Instálalo con: pip install ultralytics"
        )


def _get_yolo_model(model_size: str = "nano"):
    """Obtiene el modelo YOLO genérico (carga cacheada)."""
    return _load_yolo_model(model_size)


def _get_cow_face_model(weights_path: str | Path | None = None):
    """
    Obtiene el modelo YOLO especializado en detección de rostros de vacas.
    Retorna None si los pesos no existen.
    """
    weights = Path(weights_path) if weights_path else _COW_FACE_WEIGHTS
    if not weights.exists():
        return None
    return _load_cow_face_model(str(weights))


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


def detect_animal_boxes(
    image: Image.Image | np.ndarray,
    min_confidence: float = 0.3,
    prefer_cow: bool = True,
    padding_factor: float = 0.15,
) -> list[dict]:
    """
    Detecta animales en la imagen y devuelve bounding boxes con metadatos.
    Útil para recortar la región del animal antes de pasarla al modelo de reconocimiento.

    Args:
        image: Imagen PIL o numpy array (RGB)
        min_confidence: Confianza mínima para considerar una detección válida
        prefer_cow: Si True, prioriza detecciones de vaca sobre otros animales
        padding_factor: Factor de padding alrededor del bbox (0.15 = 15% extra por lado)

    Returns:
        Lista de diccionarios con detecciones, cada uno contiene:
        - "bbox": [x1, y1, x2, y2] coordenadas del bounding box original
        - "bbox_padded": [x1, y1, x2, y2] bbox con padding aplicado (clipped a imagen)
        - "animal": nombre del animal detectado
        - "confidence": confianza de la detección YOLO
        - "is_cow": True si es específicamente una vaca
        - "is_face": False (detección de cuerpo, no rostro)
        - "center": [cx, cy] centro del bbox
        
        La lista está ordenada: vacas primero (si prefer_cow), luego por confianza desc.
    """
    model = _get_yolo_model()

    # Convertir PIL a numpy si es necesario
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        img_array = image

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    h, w = img_array.shape[:2]

    # Clases de animales en COCO dataset
    animal_classes = {
        14: "bird", 15: "cat", 16: "dog", 17: "horse",
        18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
        22: "zebra", 23: "giraffe",
    }
    cow_class = 19

    results = model(img_array, conf=min_confidence, verbose=False)

    detections: list[dict] = []

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(boxes, confidences, classes):
            if cls not in animal_classes:
                continue

            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            bw = x2 - x1
            bh = y2 - y1

            # Aplicar padding
            pad_x = bw * padding_factor
            pad_y = bh * padding_factor
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(w, x2 + pad_x)
            py2 = min(h, y2 + pad_y)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_padded": [px1, py1, px2, py2],
                "animal": animal_classes[cls],
                "confidence": float(conf),
                "is_cow": cls == cow_class,
                "is_face": False,
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            })

    # Ordenar: vacas primero (si prefer_cow), luego por confianza descendente
    if prefer_cow:
        detections.sort(key=lambda d: (-int(d["is_cow"]), -d["confidence"]))
    else:
        detections.sort(key=lambda d: -d["confidence"])

    return detections


# ---------------------------------------------------------------------------
# Cattle face detection  (specialized Roboflow-trained model)
# ---------------------------------------------------------------------------

def detect_cow_face_boxes(
    image: Image.Image | np.ndarray,
    min_confidence: float = 0.25,
    padding_factor: float = 0.10,
    weights_path: str | Path | None = None,
) -> list[dict]:
    """
    Detecta **rostros de vacas** en la imagen usando un modelo YOLOv8 entrenado
    específicamente con el dataset "Cattle Face Detection" de Roboflow.

    Args:
        image: Imagen PIL o numpy array (RGB).
        min_confidence: Confianza mínima para considerar una detección válida.
        padding_factor: Factor de padding alrededor del bbox (0.10 = 10% extra).
        weights_path: Ruta al archivo ``best.pt`` del modelo de rostros.
                      Si es *None* se usa la ruta por defecto.

    Returns:
        Lista de diccionarios (misma estructura que ``detect_animal_boxes``),
        cada uno con:
        - ``"bbox"``, ``"bbox_padded"``, ``"animal"`` (= "cow_face"),
          ``"confidence"``, ``"is_cow"`` (True), ``"is_face"`` (True),
          ``"center"``.
        Ordenados por confianza descendente.
        Devuelve lista vacía si el modelo de rostros no está disponible.
    """
    model = _get_cow_face_model(weights_path)
    if model is None:
        return []

    # Convertir PIL a numpy si es necesario
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        img_array = image

    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    h, w = img_array.shape[:2]

    results = model(img_array, conf=min_confidence, verbose=False)

    detections: list[dict] = []

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            bw = x2 - x1
            bh = y2 - y1

            pad_x = bw * padding_factor
            pad_y = bh * padding_factor
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(w, x2 + pad_x)
            py2 = min(h, y2 + pad_y)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_padded": [px1, py1, px2, py2],
                "animal": "cow_face",
                "confidence": float(conf),
                "is_cow": True,
                "is_face": True,
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            })

    detections.sort(key=lambda d: -d["confidence"])
    return detections


def detect_best_boxes(
    image: Image.Image | np.ndarray,
    face_min_confidence: float = 0.25,
    body_min_confidence: float = 0.30,
    face_padding: float = 0.10,
    body_padding: float = 0.15,
) -> list[dict]:
    """
    Detección combinada: intenta primero **rostro de vaca** y si no
    encuentra ninguno cae a **detección de cuerpo** (COCO genérico).

    Esta es la función recomendada para el pipeline de video ya que
    maximiza la probabilidad de obtener un crop relevante para la
    identificación facial.

    Returns:
        Lista de detecciones (misma estructura). Las detecciones de
        rostro llevan ``is_face=True``; las de cuerpo ``is_face=False``.
    """
    # 1. Intentar detección de rostro
    face_dets = detect_cow_face_boxes(
        image,
        min_confidence=face_min_confidence,
        padding_factor=face_padding,
    )
    if face_dets:
        return face_dets

    # 2. Fallback: detección de cuerpo completo
    body_dets = detect_animal_boxes(
        image,
        min_confidence=body_min_confidence,
        prefer_cow=True,
        padding_factor=body_padding,
    )
    return body_dets


def get_face_model(weights_path: str | Path | None = None):
    """
    Public accessor for the cow face YOLO model instance.

    Needed by the multi-animal pipeline which calls ``model.track()``
    directly for ByteTrack-based cross-frame tracking.  Returns *None*
    if the model weights are not available.
    """
    return _get_cow_face_model(weights_path)


def validate_faces_against_bodies(
    face_dets: list[dict],
    img_array: np.ndarray,
    body_min_confidence: float = 0.20,
) -> list[dict]:
    """
    Filter face detections by requiring a nearby body detection.

    Runs the COCO body model on the full image and keeps only faces
    whose center falls inside (or near) a detected animal body.  This
    eliminates false positives on sticks, trees, and other background
    elements while preserving all genuine cow faces.

    Returns:
        Validated face detections (subset of *face_dets*).
    """
    if not face_dets:
        return []

    body_dets = detect_animal_boxes(
        img_array,
        min_confidence=body_min_confidence,
        prefer_cow=True,
        padding_factor=0.25,  # generous padding for validation area
    )

    if not body_dets:
        return []  # No animal bodies found → reject all faces

    validated = [
        f for f in face_dets
        if _face_inside_any_body(f, body_dets, margin=0.30)
    ]
    return validated


def detect_faces_multi_animal(
    image: Image.Image | np.ndarray,
    face_min_confidence: float = 0.30,
    face_padding: float = 0.30,
    body_min_confidence: float = 0.20,
) -> list[dict]:
    """
    Face-first detection with strict body validation for multi-animal tracking.

    Strategy:
      1. Run face detection on the full image → finds all individual faces
         (faces don't overlap even when bodies do).
      2. Run body detection (COCO) on the full image → confirms where
         actual animals are.
      3. Keep **only** faces whose center is inside/near a body detection.
         No body = no accepted faces.  This filters false positives on
         sticks, trees, bushes.

    Body detection doesn't need to separate individual cows — it just
    validates "there are cows in this general area".  Even if COCO sees
    3 overlapping cows as 1 blob, all 3 faces inside that blob pass
    validation.

    No tiling — it causes excessive false positives on background
    elements at tile scale.

    Returns:
        List of detection dicts, same structure as ``detect_cow_face_boxes``.
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        img_array = image

    # ── Step 1: Face detection on full image ─────────────────────────
    face_dets = detect_cow_face_boxes(
        img_array,
        min_confidence=face_min_confidence,
        padding_factor=face_padding,
    )
    if not face_dets:
        return []

    # ── Step 2: Body detection for validation ────────────────────────
    body_dets = detect_animal_boxes(
        img_array,
        min_confidence=body_min_confidence,
        prefer_cow=True,
        padding_factor=0.25,  # generous padding for validation area
    )

    if not body_dets:
        # No animal bodies found → reject all faces (likely FPs)
        return []

    # ── Step 3: Strict validation — only keep body-validated faces ───
    validated = [
        f for f in face_dets
        if _face_inside_any_body(f, body_dets, margin=0.30)
    ]

    if validated:
        validated = _nms_detections(validated, iou_threshold=0.4)
        validated.sort(key=lambda d: -d["confidence"])

    return validated


# ---------------------------------------------------------------------------
# Multi-scale (tiled) detection for small / distant animals
# ---------------------------------------------------------------------------

def _face_inside_any_body(
    face: dict,
    body_dets: list[dict],
    margin: float = 0.15,
) -> bool:
    """Check if a face detection's center is inside (or near) any body bbox.

    Uses the padded body bbox expanded by *margin* so that faces near the
    edge of a body are still accepted.
    """
    cx, cy = face["center"]
    for body in body_dets:
        bx1, by1, bx2, by2 = body["bbox_padded"]
        bw = bx2 - bx1
        bh = by2 - by1
        # Expand body region by margin
        ex1 = bx1 - bw * margin
        ey1 = by1 - bh * margin
        ex2 = bx2 + bw * margin
        ey2 = by2 + bh * margin
        if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
            return True
    return False


def _compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _nms_detections(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Simple non-maximum suppression to remove duplicate detections from tiling."""
    if not detections:
        return detections

    # Sort by: face detections first, then by confidence
    detections.sort(key=lambda d: (-int(d.get("is_face", False)), -d["confidence"]))

    keep: list[dict] = []
    for det in detections:
        box = det["bbox"]
        is_duplicate = False
        for kept in keep:
            if _compute_iou(box, kept["bbox"]) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(det)

    return keep


def _find_face_in_body(
    img_array: np.ndarray,
    body_det: dict,
    face_min_confidence: float = 0.25,
    face_padding: float = 0.10,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> dict | None:
    """
    Given a body detection, crop that region and try to find a cow face
    inside it.  Returns the **best** face detection with coordinates mapped
    back to the full image, or *None* if no face was found.
    """
    results = _find_all_faces_in_body(
        img_array, body_det,
        face_min_confidence=face_min_confidence,
        face_padding=face_padding,
        offset_x=offset_x, offset_y=offset_y,
    )
    return results[0] if results else None


def _find_all_faces_in_body(
    img_array: np.ndarray,
    body_det: dict,
    face_min_confidence: float = 0.25,
    face_padding: float = 0.10,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[dict]:
    """
    Given a body detection, crop that region and find **all** cow faces
    inside it.  Returns face detections with coordinates mapped back to
    the full image.  Important for multi-animal: a large body box
    covering 2-3 adjacent cows can contain multiple faces.
    """
    px1, py1, px2, py2 = body_det["bbox_padded"]
    bx1, by1 = int(px1), int(py1)
    bx2, by2 = int(px2), int(py2)
    body_crop = img_array[by1:by2, bx1:bx2]

    if body_crop.size == 0 or body_crop.shape[0] < 20 or body_crop.shape[1] < 20:
        return []

    faces = detect_cow_face_boxes(
        body_crop,
        min_confidence=face_min_confidence,
        padding_factor=face_padding,
    )
    if not faces:
        return []

    # Map ALL face coordinates: body-crop → full image
    for face in faces:
        face["bbox"] = [
            face["bbox"][0] + bx1 + offset_x,
            face["bbox"][1] + by1 + offset_y,
            face["bbox"][2] + bx1 + offset_x,
            face["bbox"][3] + by1 + offset_y,
        ]
        face["bbox_padded"] = [
            face["bbox_padded"][0] + bx1 + offset_x,
            face["bbox_padded"][1] + by1 + offset_y,
            face["bbox_padded"][2] + bx1 + offset_x,
            face["bbox_padded"][3] + by1 + offset_y,
        ]
        face["center"] = [
            face["center"][0] + bx1 + offset_x,
            face["center"][1] + by1 + offset_y,
        ]
    return faces


def _map_det_coords(det: dict, offset_x: float, offset_y: float) -> dict:
    """Offset a detection's coordinates (for tiling)."""
    det["bbox"] = [
        det["bbox"][0] + offset_x, det["bbox"][1] + offset_y,
        det["bbox"][2] + offset_x, det["bbox"][3] + offset_y,
    ]
    det["bbox_padded"] = [
        det["bbox_padded"][0] + offset_x, det["bbox_padded"][1] + offset_y,
        det["bbox_padded"][2] + offset_x, det["bbox_padded"][3] + offset_y,
    ]
    det["center"] = [
        det["center"][0] + offset_x, det["center"][1] + offset_y,
    ]
    return det


def detect_best_boxes_multiscale(
    image: Image.Image | np.ndarray,
    face_min_confidence: float = 0.25,
    body_min_confidence: float = 0.30,
    face_padding: float = 0.10,
    body_padding: float = 0.15,
    require_body: bool = False,
) -> list[dict]:
    """
    Robust two-stage detection for inference on field / distant videos.

    The cow-face YOLO model can produce false positives on background
    elements (trees, bushes) when run on a full landscape frame.  To
    prevent this, the function uses a **body-first cascade**:

    When ``require_body`` is True, face detections are only accepted if
    a cow body was also detected (Stages 1 & 3).  Stage 2 (face-only
    fallback) is skipped entirely.  This eliminates false positives on
    trees/bushes at the cost of missing very-close-up faces with no
    visible body.  Recommended for multi-animal video.

      1. Detect cow **bodies** with the generic COCO model (very reliable,
         almost no false positives on background).
      2. Within each confirmed body region, run the specialised **face**
         detector.  This gives the tightest, most useful crop.
      3. If no body was found at full-image scale, try high-confidence
         face-only detection (for close-up images where the cow fills the
         frame and COCO may not fire).
      4. If still nothing, split the image into 2x2 overlapping tiles and
         repeat the body→face cascade on each tile (catches small /
         distant animals).

    The result format is identical to ``detect_best_boxes``.
    """
    # Normalise to numpy once
    if isinstance(image, Image.Image):
        img_array = np.array(image.convert("RGB"))
    else:
        img_array = image

    h, w = img_array.shape[:2]

    # ── Stage 1: body detection on full image (COCO — reliable) ──────
    body_dets = detect_animal_boxes(
        img_array,
        min_confidence=body_min_confidence,
        prefer_cow=True,
        padding_factor=body_padding,
    )

    if body_dets:
        # Strategy: run face detection on the FULL image (not cropped
        # per body).  This avoids the problem where a body crop cuts
        # through an adjacent cow's face making it undetectable.  Then
        # validate each face: its center must fall inside (or near) a
        # detected body.  This prevents tree/bush false positives while
        # maximising face recall.
        full_image_faces = detect_cow_face_boxes(
            img_array,
            min_confidence=face_min_confidence,
            padding_factor=face_padding,
        )

        # Validate faces against body detections
        validated_faces: list[dict] = []
        if full_image_faces:
            for face in full_image_faces:
                if _face_inside_any_body(face, body_dets):
                    validated_faces.append(face)

        # NMS on validated faces
        if validated_faces:
            validated_faces = _nms_detections(validated_faces, iou_threshold=0.4)

        # Body-only fallbacks: for bodies that have no validated face
        # nearby, keep the body detection so we don't lose any animal.
        body_only: list[dict] = []
        for body in body_dets:
            has_face = False
            for face in validated_faces:
                if _compute_iou(face["bbox"], body["bbox"]) > 0.05:
                    has_face = True
                    break
                # Also check if face center is inside body
                cx, cy = face["center"]
                bx1, by1, bx2, by2 = body["bbox_padded"]
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    has_face = True
                    break
            if not has_face:
                body_only.append(body)
        body_only = _nms_detections(body_only, iou_threshold=0.5)

        combined = validated_faces + body_only
        if combined:
            combined.sort(key=lambda d: (-int(d.get("is_face", False)), -d["confidence"]))
            return combined

        return body_dets

    # ── Stage 2: direct face detection with HIGH confidence ──────────
    # For close-up images where the cow fills the frame and the COCO
    # body detector may not fire.  High threshold avoids false positives.
    # Skipped when require_body=True to prevent false positives on
    # trees/bushes in multi-animal video.
    if not require_body:
        high_conf = max(face_min_confidence, 0.45)
        face_dets = detect_cow_face_boxes(
            img_array,
            min_confidence=high_conf,
            padding_factor=face_padding,
        )
        if face_dets:
            return face_dets

    # ── Stage 3: tiled detection for small / distant animals ─────────
    if h < 200 or w < 200:
        return []

    overlap_frac = 0.30
    rows, cols = 2, 2
    tile_h = int(h * (1 + overlap_frac) / rows)
    tile_w = int(w * (1 + overlap_frac) / cols)
    step_h = (h - tile_h) // max(rows - 1, 1)
    step_w = (w - tile_w) // max(cols - 1, 1)

    tile_body_conf = body_min_confidence * 0.8
    tile_face_conf = face_min_confidence * 0.8

    all_dets: list[dict] = []

    for r in range(rows):
        for c in range(cols):
            ty1 = r * step_h
            tx1 = c * step_w
            ty2 = min(h, ty1 + tile_h)
            tx2 = min(w, tx1 + tile_w)

            tile = img_array[ty1:ty2, tx1:tx2]

            # Body detection on tile
            tile_bodies = detect_animal_boxes(
                tile,
                min_confidence=tile_body_conf,
                prefer_cow=True,
                padding_factor=body_padding,
            )

            tile_has_face = False
            for body in tile_bodies:
                # Try ALL faces within this body
                faces = _find_all_faces_in_body(
                    tile, body,
                    face_min_confidence=tile_face_conf,
                    face_padding=face_padding,
                    offset_x=tx1, offset_y=ty1,
                )
                if faces:
                    all_dets.extend(faces)
                    tile_has_face = True
                else:
                    all_dets.append(_map_det_coords(body, tx1, ty1))

    # Remove duplicates from overlapping tiles
    if len(all_dets) > 1:
        all_dets = _nms_detections(all_dets, iou_threshold=0.5)

    # Sort: faces first, then by confidence desc
    all_dets.sort(key=lambda d: (-int(d.get("is_face", False)), -d["confidence"]))

    return all_dets
