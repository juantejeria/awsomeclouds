from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Configurar variables de entorno ANTES de importar PyTorch (previene crashes en macOS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .io_utils import read_json
from .model import build_model, extract_embeddings, predict_proba
from .video_utils import iter_video_frames


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    decision: Literal["known", "unknown"]
    cosine_similarity: float = 0.0   # Similarity to nearest centroid (1.0 = identical)
    nearest_centroid: str = ""       # Which centroid was closest


@dataclass
class FrameDetection:
    """Resultado de detección + reconocimiento en un frame individual."""
    frame_idx: int
    frame_rgb: np.ndarray          # Frame original completo (HxWx3)
    bbox: list[float]              # [x1, y1, x2, y2] bbox original de YOLO
    bbox_padded: list[float]       # [x1, y1, x2, y2] bbox con padding
    crop_rgb: np.ndarray           # Recorte del animal (HxWx3)
    yolo_confidence: float         # Confianza de detección YOLO
    animal_type: str               # Tipo de animal detectado ("cow", "cow_face", etc.)
    label: str                     # Predicción del modelo de reconocimiento
    recognition_confidence: float  # Confianza del modelo de reconocimiento
    all_probs: np.ndarray          # Probabilidades de todas las clases
    cosine_similarity: float = 0.0 # Similitud coseno al centroide más cercano
    is_face_detection: bool = False  # True si la detección fue de rostro (no cuerpo)
    track_id: int = -1              # -1 = untracked (backward compat)


@dataclass
class VideoResult:
    """Resultado completo del análisis de video con detección por frame."""
    prediction: Prediction                    # Predicción final (mayoría + top-N confianza)
    frame_detections: list[FrameDetection]    # TODAS las detecciones por frame
    winning_detections: list[FrameDetection]  # Solo frames del individuo ganador
    top_detections: list[FrameDetection]      # Top N mejores frames del individuo ganador
    total_frames_extracted: int               # Total de frames extraídos del video
    frames_with_detections: int               # Frames donde se detectó animal
    frames_without_detections: int            # Frames sin detección
    classes: list[str] = field(default_factory=list)  # Lista de clases del modelo
    agreement_ratio: float = 0.0             # % de frames que votaron por la clase ganadora
    winning_class_avg_conf: float = 0.0      # Confianza promedio de los top N frames ganadores
    winning_label: str = ""                  # Nombre del individuo ganador
    winning_count: int = 0                   # Cuántos frames votaron por el ganador
    avg_cosine_similarity: float = 0.0       # Similitud coseno promedio de los top frames
    frames_with_face: int = 0                # Frames donde se detectó rostro de vaca
    frames_with_body_only: int = 0           # Frames con detección de cuerpo (sin rostro)


@dataclass
class TrackResult:
    """Per-track aggregated recognition result (one tracked animal)."""
    track_id: int
    prediction: Prediction
    frame_detections: list[FrameDetection]   # All detections for this track
    top_detections: list[FrameDetection]     # Top N best frames
    winning_label: str = ""
    winning_count: int = 0
    agreement_ratio: float = 0.0
    winning_class_avg_conf: float = 0.0
    avg_cosine_similarity: float = 0.0
    frames_with_face: int = 0
    frames_with_body_only: int = 0


@dataclass
class MultiVideoResult:
    """Container for multi-animal tracking results across all tracks."""
    tracks: list[TrackResult]                # Sorted by track length descending
    total_frames_extracted: int
    frames_with_detections: int
    frames_without_detections: int
    all_frame_detections: list[FrameDetection]
    classes: list[str] = field(default_factory=list)
    noise_tracks_discarded: int = 0


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Centroid I/O for open-set recognition
# ---------------------------------------------------------------------------

def compute_centroids(
    model,
    tfm,
    dev: torch.device,
    data_dir: str | Path,
    classes: list[str],
) -> dict[str, np.ndarray]:
    """
    Compute the embedding centroid (average L2-normalised feature vector)
    for each class by running every training image through the backbone.

    Args:
        model: Trained ResNet18 model (eval mode)
        tfm: Inference transforms
        dev: Device
        data_dir: Root ImageFolder directory (each subfolder = class)
        classes: Ordered list of class names

    Returns:
        Dictionary mapping class name -> (512,) numpy centroid vector
    """
    from collections import defaultdict

    data_dir = Path(data_dir)
    embeddings_by_class: dict[str, list[np.ndarray]] = defaultdict(list)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for cls_name in classes:
        cls_dir = data_dir / cls_name
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in img_extensions:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                x = tfm(img).unsqueeze(0).to(dev)
                emb = extract_embeddings(model, x)[0].detach().cpu().numpy()
                embeddings_by_class[cls_name].append(emb)
            except Exception:
                continue  # skip corrupt images

    centroids: dict[str, np.ndarray] = {}
    for cls_name in classes:
        vecs = embeddings_by_class.get(cls_name, [])
        if vecs:
            centroid = np.mean(np.stack(vecs, axis=0), axis=0)
            # Re-normalise centroid to unit length
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids[cls_name] = centroid

    return centroids


def save_centroids(centroids: dict[str, np.ndarray], artifacts_dir: str | Path) -> Path:
    """Save centroids dict as a .npz file in the artifacts directory."""
    artifacts_dir = Path(artifacts_dir)
    path = artifacts_dir / "centroids.npz"
    np.savez(str(path), **centroids)
    return path


def load_centroids(artifacts_dir: str | Path) -> dict[str, np.ndarray] | None:
    """Load centroids from artifacts directory. Returns None if not found."""
    path = Path(artifacts_dir) / "centroids.npz"
    if not path.exists():
        return None
    data = np.load(str(path))
    return {key: data[key] for key in data.files}


def cosine_similarity_to_centroids(
    embedding: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Compute cosine similarity between an embedding and all centroids.
    Both embedding and centroids are assumed to be L2-normalised,
    so cosine similarity = dot product.

    Returns:
        Dictionary mapping class name -> similarity (0..1)
    """
    sims: dict[str, float] = {}
    for cls_name, centroid in centroids.items():
        sims[cls_name] = float(np.dot(embedding, centroid))
    return sims


def load_artifacts(artifacts_dir: str | Path):
    """
    Load model, classes, transforms, device, and centroids.

    Returns:
        (model, classes, tfm, dev, centroids)
        centroids may be None if centroids.npz has not been generated yet.
    """
    artifacts_dir = Path(artifacts_dir)
    classes = read_json(artifacts_dir / "classes.json")
    config = read_json(artifacts_dir / "config.json")

    # Match the dropout used during training (default 0.4 for new models, 0 for legacy)
    dropout = float(config.get("dropout", 0.0))
    model = build_model(num_classes=len(classes), pretrained=False, dropout=dropout)
    state = torch.load(artifacts_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dev = _device()
    model.to(dev)

    img_size = int(config.get("img_size", 224))
    # Inference transforms: deterministic resize/center-crop + ImageNet normalization
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    centroids = load_centroids(artifacts_dir)

    return model, classes, tfm, dev, centroids


def predict_image(
    model,
    classes: list[str],
    tfm,
    dev: torch.device,
    image: Image.Image,
    threshold: float = 0.70,
    use_tta: bool = True,
    centroids: dict[str, np.ndarray] | None = None,
    similarity_threshold: float = 0.50,
    use_face_detection: bool = True,
) -> Prediction:
    """
    Predict class for a single image with open-set recognition.

    When ``use_face_detection`` is True (default), the function first tries
    to detect a cow face using the specialized YOLO model. If a face is
    found the crop is used for identification; otherwise it falls back to
    body detection; and if that also fails, the full image is used.

    Decision logic (when centroids are available):
      - softmax_conf >= threshold AND cosine_sim >= similarity_threshold -> "known"
      - Otherwise -> "unknown"
    This prevents unknown individuals from being misclassified with
    high softmax confidence.

    Args:
        use_tta: If True, applies Test-Time Augmentation (horizontal flip +
                 small crops) and averages predictions for more robust results.
        centroids: Pre-computed class centroids for open-set recognition.
                   If None, falls back to softmax-only decision.
        similarity_threshold: Minimum cosine similarity to nearest centroid
                              to consider the prediction "known".
        use_face_detection: If True, tries cow-face detection (then body
                            fallback) before running the recognition model.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # ------------------------------------------------------------------
    # Try to crop the cow face / body before running recognition
    # ------------------------------------------------------------------
    recognition_image = image  # default: full image

    if use_face_detection:
        from .face_detection import detect_best_boxes_multiscale

        img_array = np.array(image)
        detections = detect_best_boxes_multiscale(img_array)

        if detections:
            best = detections[0]
            px1, py1, px2, py2 = best["bbox_padded"]
            crop = img_array[int(py1):int(py2), int(px1):int(px2)]

            if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                # Always use the crop — the model was trained on
                # face/body crops so the crop is the best input.
                recognition_image = Image.fromarray(crop).convert("RGB")

    # ------------------------------------------------------------------
    # Run recognition on the chosen image
    # ------------------------------------------------------------------
    if not use_tta:
        x = tfm(recognition_image).unsqueeze(0).to(dev)
        probs = predict_proba(model, x)[0].detach().cpu().numpy()
    else:
        probs = _predict_image_tta(model, tfm, dev, recognition_image)

    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = classes[idx]

    # --- Embedding-based open-set check ---
    cos_sim = 0.0
    nearest = label
    if centroids is not None:
        x_emb = tfm(recognition_image).unsqueeze(0).to(dev)
        emb = extract_embeddings(model, x_emb)[0].detach().cpu().numpy()
        sims = cosine_similarity_to_centroids(emb, centroids)
        if sims:
            nearest = max(sims, key=sims.get)
            cos_sim = sims[nearest]
            # Dual decision: both softmax AND embedding must agree
            decision: Literal["known", "unknown"] = (
                "known" if conf >= threshold and cos_sim >= similarity_threshold
                else "unknown"
            )
        else:
            decision = "known" if conf >= threshold else "unknown"
    else:
        decision = "known" if conf >= threshold else "unknown"

    return Prediction(
        label=label,
        confidence=conf,
        decision=decision,
        cosine_similarity=cos_sim,
        nearest_centroid=nearest,
    )


def _predict_image_tta(
    model,
    tfm,
    dev: torch.device,
    image: Image.Image,
    n_crops: int = 4,
) -> np.ndarray:
    """
    Test-Time Augmentation: run model on original + augmented versions and
    average the probability distributions.

    Augmentations:
      - Original
      - Horizontal flip
      - n_crops random center-ish crops (85-95% of image)
    """
    from torchvision.transforms import functional as F

    all_probs: list[np.ndarray] = []

    # 1. Original
    x = tfm(image).unsqueeze(0).to(dev)
    all_probs.append(predict_proba(model, x)[0].detach().cpu().numpy())

    # 2. Horizontal flip
    flipped = F.hflip(image)
    x = tfm(flipped).unsqueeze(0).to(dev)
    all_probs.append(predict_proba(model, x)[0].detach().cpu().numpy())

    # 3. Small random crops around center
    w, h = image.size
    for i in range(n_crops):
        # Crop between 85% and 95% of the image, slightly offset
        scale = 0.85 + (i / max(n_crops - 1, 1)) * 0.10
        cw, ch = int(w * scale), int(h * scale)
        # Offset from center slightly
        left = max(0, (w - cw) // 2 + (i % 2) * int(w * 0.02))
        top = max(0, (h - ch) // 2 + ((i + 1) % 2) * int(h * 0.02))
        right = min(w, left + cw)
        bottom = min(h, top + ch)
        cropped = image.crop((left, top, right, bottom))
        x = tfm(cropped).unsqueeze(0).to(dev)
        all_probs.append(predict_proba(model, x)[0].detach().cpu().numpy())

    return np.mean(np.stack(all_probs, axis=0), axis=0)


def _predict_frame_tta(
    model,
    tfm,
    dev: torch.device,
    image: Image.Image,
) -> np.ndarray:
    """
    Lightweight TTA for video frames: original + horizontal flip.
    Returns averaged probability distribution.
    Only 2x cost (vs 6x for full TTA), suitable for per-frame use in video.
    """
    from torchvision.transforms import functional as F

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Original
    x = tfm(image).unsqueeze(0).to(dev)
    p1 = predict_proba(model, x)[0].detach().cpu().numpy()

    # Horizontal flip
    flipped = F.hflip(image)
    x = tfm(flipped).unsqueeze(0).to(dev)
    p2 = predict_proba(model, x)[0].detach().cpu().numpy()

    return (p1 + p2) / 2.0


def predict_video(
    model,
    classes: list[str],
    tfm,
    dev: torch.device,
    video_path: str,
    threshold: float = 0.70,
    stride: int = 10,
    max_frames: int = 150,
) -> Prediction:
    """Predicción de video original (sin detección YOLO, analiza frames completos)."""
    all_probs = []
    for fr in iter_video_frames(video_path, stride=stride, max_frames=max_frames):
        img = Image.fromarray(fr.rgb)
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = tfm(img).unsqueeze(0).to(dev)
        probs = predict_proba(model, x)[0].detach().cpu().numpy()
        all_probs.append(probs)

    if not all_probs:
        return Prediction(label="(sin frames)", confidence=0.0, decision="unknown")

    mean_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    idx = int(np.argmax(mean_probs))
    conf = float(mean_probs[idx])
    label = classes[idx]
    decision: Literal["known", "unknown"] = "known" if conf >= threshold else "unknown"
    return Prediction(label=label, confidence=conf, decision=decision)


def predict_video_with_detection(
    model,
    classes: list[str],
    tfm,
    dev: torch.device,
    video_path: str,
    threshold: float = 0.70,
    stride: int = 10,
    max_frames: int = 150,
    yolo_min_confidence: float = 0.20,
    progress_callback=None,
    centroids: dict[str, np.ndarray] | None = None,
    similarity_threshold: float = 0.50,
) -> VideoResult:
    """
    Predicción de video mejorada: usa YOLO para detectar el animal en cada frame,
    recorta la región detectada y corre el reconocimiento solo en esa sección.

    Usa detección multi-escala (tiling) para detectar animales pequeños o
    lejanos. Frames sin detección son descartados (el modelo fue entrenado
    con crops de rostro/cuerpo, no con frames completos de paisaje).

    Open-set recognition: when centroids are provided, the final decision
    requires BOTH softmax confidence >= threshold AND average cosine
    similarity >= similarity_threshold.

    Args:
        model: Modelo de reconocimiento entrenado
        classes: Lista de nombres de clases
        tfm: Transformaciones de imagen
        dev: Dispositivo (CPU/CUDA)
        video_path: Ruta al archivo de video
        threshold: Umbral de confianza para decisión known/unknown
        stride: Cada N frames extraer uno
        max_frames: Máximo de frames a procesar
        yolo_min_confidence: Confianza mínima de YOLO para considerar una detección
        progress_callback: Función callback(current, total) para reportar progreso
        centroids: Pre-computed class centroids for open-set recognition
        similarity_threshold: Minimum cosine similarity to consider "known"

    Returns:
        VideoResult con la predicción final y detalles por frame
    """
    from .face_detection import detect_best_boxes_multiscale

    frame_detections: list[FrameDetection] = []
    all_probs: list[np.ndarray] = []
    total_frames = 0
    frames_with_det = 0
    frames_without_det = 0

    # Recolectar frames primero para saber el total
    frames_list = list(iter_video_frames(video_path, stride=stride, max_frames=max_frames))
    total_frames = len(frames_list)

    for frame_num, fr in enumerate(frames_list):
        if progress_callback:
            progress_callback(frame_num + 1, total_frames)

        # ------------------------------------------------------------------
        # Detect cow face/body first (with multi-scale tiling fallback
        # for small / distant animals)
        # ------------------------------------------------------------------
        detections = detect_best_boxes_multiscale(
            fr.rgb,
            face_min_confidence=yolo_min_confidence,
            body_min_confidence=yolo_min_confidence,
        )

        if not detections:
            # No animal detected — skip this frame entirely.
            # The model was trained on face/body crops so running it on a
            # full landscape frame (trees, sky, etc.) produces noise.
            frames_without_det += 1
            continue

        best_det = detections[0]
        px1, py1, px2, py2 = best_det["bbox_padded"]

        # Crop the detected region
        raw_crop = fr.rgb[int(py1):int(py2), int(px1):int(px2)]

        if raw_crop.size == 0 or raw_crop.shape[0] < 10 or raw_crop.shape[1] < 10:
            frames_without_det += 1
            continue

        crop = raw_crop
        crop_img = Image.fromarray(crop).convert("RGB")

        # ------------------------------------------------------------------
        # Run recognition ONLY on the crop (not on the full frame).
        # The model was trained on face/body crops so using the crop
        # directly gives the most consistent results.
        # ------------------------------------------------------------------
        best_probs = _predict_frame_tta(model, tfm, dev, crop_img)

        idx = int(np.argmax(best_probs))
        conf = float(best_probs[idx])
        label = classes[idx]

        # Embedding similarity (if centroids available)
        frame_cos_sim = 0.0
        if centroids is not None:
            x_emb = tfm(crop_img).unsqueeze(0).to(dev)
            emb = extract_embeddings(model, x_emb)[0].detach().cpu().numpy()
            sims = cosine_similarity_to_centroids(emb, centroids)
            if sims:
                frame_cos_sim = max(sims.values())

        frames_with_det += 1

        frame_detections.append(FrameDetection(
            frame_idx=fr.idx,
            frame_rgb=fr.rgb,
            bbox=best_det["bbox"],
            bbox_padded=best_det["bbox_padded"],
            crop_rgb=crop,
            yolo_confidence=best_det["confidence"],
            animal_type=best_det["animal"],
            label=label,
            recognition_confidence=conf,
            all_probs=best_probs,
            cosine_similarity=frame_cos_sim,
            is_face_detection=best_det.get("is_face", False),
        ))

    # Calcular predicción final:
    # 1. Votación por mayoría → individuo ganador
    # 2. Filtrar solo frames del ganador, ordenados por confianza desc
    # 3. Tomar los TOP N mejores frames (default 10)
    # 4. Confianza final = promedio de esos top N
    TOP_N = 10

    if not frame_detections:
        prediction = Prediction(
            label="(sin detecciones)",
            confidence=0.0,
            decision="unknown",
        )
        return VideoResult(
            prediction=prediction,
            frame_detections=frame_detections,
            winning_detections=[],
            top_detections=[],
            total_frames_extracted=total_frames,
            frames_with_detections=frames_with_det,
            frames_without_detections=frames_without_det,
            classes=classes,
        )

    # 1. Contar votos por clase
    from collections import Counter
    votes = Counter(det.label for det in frame_detections)
    w_label, w_count = votes.most_common(1)[0]

    # 2. Filtrar frames del individuo ganador, ordenados por confianza desc
    winning_dets = sorted(
        [det for det in frame_detections if det.label == w_label],
        key=lambda d: d.recognition_confidence,
        reverse=True,
    )

    # 3. Top N mejores frames del ganador
    top_dets = winning_dets[:TOP_N]

    # 4. Confianza final = promedio de los top N
    top_confs = [det.recognition_confidence for det in top_dets]
    winning_avg = float(np.mean(top_confs))

    # Ratio de concordancia
    agreement = w_count / len(frame_detections)

    # Average cosine similarity of top frames
    top_cos_sims = [det.cosine_similarity for det in top_dets]
    avg_cos_sim = float(np.mean(top_cos_sims)) if top_cos_sims else 0.0

    # Dual decision: softmax + embedding distance
    if centroids is not None and avg_cos_sim > 0:
        decision: Literal["known", "unknown"] = (
            "known" if winning_avg >= threshold and avg_cos_sim >= similarity_threshold
            else "unknown"
        )
    else:
        decision = "known" if winning_avg >= threshold else "unknown"

    prediction = Prediction(
        label=w_label,
        confidence=winning_avg,
        decision=decision,
        cosine_similarity=avg_cos_sim,
        nearest_centroid=w_label,
    )

    # Face vs body stats
    n_face = sum(1 for d in frame_detections if d.is_face_detection)
    n_body_only = frames_with_det - n_face

    return VideoResult(
        prediction=prediction,
        frame_detections=frame_detections,
        winning_detections=winning_dets,
        top_detections=top_dets,
        total_frames_extracted=total_frames,
        frames_with_detections=frames_with_det,
        frames_without_detections=frames_without_det,
        classes=classes,
        agreement_ratio=agreement,
        winning_class_avg_conf=winning_avg,
        winning_label=w_label,
        winning_count=w_count,
        avg_cosine_similarity=avg_cos_sim,
        frames_with_face=n_face,
        frames_with_body_only=n_body_only,
    )


def predict_video_multi_animal(
    model,
    classes: list[str],
    tfm,
    dev: torch.device,
    video_path: str,
    threshold: float = 0.70,
    stride: int = 10,
    max_frames: int | None = None,
    yolo_min_confidence: float = 0.20,
    progress_callback=None,
    centroids: dict[str, np.ndarray] | None = None,
    similarity_threshold: float = 0.50,
    min_track_length: int = 5,
) -> MultiVideoResult:
    """
    Multi-animal video prediction using YOLO's built-in ByteTrack tracker.

    Instead of per-frame detection + manual IoU tracker, this uses
    ``face_model.track(frame, persist=True)`` which internally runs
    BoTSORT/ByteTrack — Kalman-filter motion prediction plus
    appearance-based re-identification.  This produces far more stable
    track IDs across frames, even with high stride.

    Body detection (COCO) is still used as a validator: faces whose
    center does not fall inside a detected animal body are rejected to
    filter false positives on sticks, trees, etc.

    After tracking, tracks sharing the same recognition label are merged
    (handles the case where an animal briefly leaves and re-enters the
    frame with a new ByteTrack ID).

    Returns:
        MultiVideoResult with one TrackResult per tracked animal.
    """
    from collections import Counter, defaultdict

    from .face_detection import get_face_model, validate_faces_against_bodies

    face_model = get_face_model()
    if face_model is None:
        return MultiVideoResult(
            tracks=[],
            total_frames_extracted=0,
            frames_with_detections=0,
            frames_without_detections=0,
            all_frame_detections=[],
            classes=classes,
        )

    # Reset tracker state so each video starts fresh.
    # Setting predictor to None forces ultralytics to create a new
    # predictor (and thus a new BoTSORT/ByteTrack instance) on the
    # next .track() call.
    face_model.predictor = None

    all_frame_detections: list[FrameDetection] = []
    total_frames = 0
    frames_with_det = 0
    frames_without_det = 0

    frames_list = list(iter_video_frames(video_path, stride=stride, max_frames=max_frames))
    total_frames = len(frames_list)

    face_conf = max(yolo_min_confidence, 0.30)

    for frame_num, fr in enumerate(frames_list):
        if progress_callback:
            progress_callback(frame_num + 1, total_frames)

        img_array = fr.rgb
        h, w = img_array.shape[:2]

        # --- ByteTrack face tracking ---
        # persist=True keeps the internal tracker state between calls
        # so track IDs are consistent across frames.
        results = face_model.track(
            img_array,
            persist=True,
            conf=face_conf,
            verbose=False,
        )

        # Extract face detections with ByteTrack track IDs
        face_dets: list[dict] = []
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id  # None when tracker couldn't assign

            if track_ids is not None:
                track_ids = track_ids.cpu().numpy().astype(int)

            pad_factor = 0.30
            for i, (box, conf_val) in enumerate(zip(boxes, confs)):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                bw, bh = x2 - x1, y2 - y1

                px1 = max(0, x1 - bw * pad_factor)
                py1 = max(0, y1 - bh * pad_factor)
                px2 = min(w, x2 + bw * pad_factor)
                py2 = min(h, y2 + bh * pad_factor)

                tid = int(track_ids[i]) if track_ids is not None else -1

                face_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "bbox_padded": [px1, py1, px2, py2],
                    "animal": "cow_face",
                    "confidence": float(conf_val),
                    "is_cow": True,
                    "is_face": True,
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    "track_id": tid,
                })

        # --- Body validation to filter false positives ---
        validated = validate_faces_against_bodies(face_dets, img_array)

        if not validated:
            frames_without_det += 1
            continue

        frames_with_det += 1

        # --- Recognition on each validated face ---
        for det in validated:
            px1, py1, px2, py2 = det["bbox_padded"]
            raw_crop = img_array[int(py1):int(py2), int(px1):int(px2)]

            if raw_crop.size == 0 or raw_crop.shape[0] < 10 or raw_crop.shape[1] < 10:
                continue

            crop_img = Image.fromarray(raw_crop).convert("RGB")

            # Lightweight TTA recognition on the crop
            best_probs = _predict_frame_tta(model, tfm, dev, crop_img)
            idx = int(np.argmax(best_probs))
            conf = float(best_probs[idx])
            label = classes[idx]

            # Embedding similarity
            frame_cos_sim = 0.0
            if centroids is not None:
                x_emb = tfm(crop_img).unsqueeze(0).to(dev)
                emb = extract_embeddings(model, x_emb)[0].detach().cpu().numpy()
                sims = cosine_similarity_to_centroids(emb, centroids)
                if sims:
                    frame_cos_sim = max(sims.values())

            all_frame_detections.append(FrameDetection(
                frame_idx=fr.idx,
                frame_rgb=fr.rgb,
                bbox=det["bbox"],
                bbox_padded=det["bbox_padded"],
                crop_rgb=raw_crop,
                yolo_confidence=det["confidence"],
                animal_type=det["animal"],
                label=label,
                recognition_confidence=conf,
                all_probs=best_probs,
                cosine_similarity=frame_cos_sim,
                is_face_detection=True,
                track_id=det["track_id"],
            ))

    # ------------------------------------------------------------------
    # Aggregate per ByteTrack ID, then merge tracks with same identity
    # ------------------------------------------------------------------
    TOP_N = 10

    # Group detections by track_id
    dets_by_track: dict[int, list[FrameDetection]] = defaultdict(list)
    for det in all_frame_detections:
        if det.track_id >= 0:
            dets_by_track[det.track_id].append(det)

    # Filter tracks by minimum detection count
    valid_track_ids = {
        tid for tid, dets in dets_by_track.items()
        if len(dets) >= min_track_length
    }
    noise_discarded = len(dets_by_track) - len(valid_track_ids)

    # Step 1: Compute winning label per raw track
    raw_track_labels: dict[int, str] = {}
    raw_track_dets: dict[int, list[FrameDetection]] = {}
    for tid in valid_track_ids:
        dets = dets_by_track[tid]
        raw_track_dets[tid] = dets
        votes = Counter(d.label for d in dets)
        raw_track_labels[tid] = votes.most_common(1)[0][0]

    # Step 2: Merge tracks that share the same winning label.
    # ByteTrack is much more stable than manual IoU tracking, but an
    # animal that briefly leaves the frame and re-enters still gets a
    # new track ID.  Merging by identity collapses those fragments.
    label_to_tids: dict[str, list[int]] = defaultdict(list)
    for tid, label in raw_track_labels.items():
        label_to_tids[label].append(tid)

    track_results: list[TrackResult] = []
    merged_count = 0

    for label, tids in label_to_tids.items():
        merged_dets: list[FrameDetection] = []
        for tid in tids:
            merged_dets.extend(raw_track_dets[tid])
        if len(tids) > 1:
            merged_count += len(tids) - 1

        representative_tid = tids[0]

        votes = Counter(d.label for d in merged_dets)
        w_label, w_count = votes.most_common(1)[0]

        winning_dets = sorted(
            [d for d in merged_dets if d.label == w_label],
            key=lambda d: d.recognition_confidence,
            reverse=True,
        )
        top_dets = winning_dets[:TOP_N]

        top_confs = [d.recognition_confidence for d in top_dets]
        winning_avg = float(np.mean(top_confs))
        agreement = w_count / len(merged_dets)

        top_cos_sims = [d.cosine_similarity for d in top_dets]
        avg_cos_sim = float(np.mean(top_cos_sims)) if top_cos_sims else 0.0

        if centroids is not None and avg_cos_sim > 0:
            decision: Literal["known", "unknown"] = (
                "known" if winning_avg >= threshold and avg_cos_sim >= similarity_threshold
                else "unknown"
            )
        else:
            decision = "known" if winning_avg >= threshold else "unknown"

        prediction = Prediction(
            label=w_label,
            confidence=winning_avg,
            decision=decision,
            cosine_similarity=avg_cos_sim,
            nearest_centroid=w_label,
        )

        n_face = sum(1 for d in merged_dets if d.is_face_detection)
        n_body_only = len(merged_dets) - n_face

        track_results.append(TrackResult(
            track_id=representative_tid,
            prediction=prediction,
            frame_detections=merged_dets,
            top_detections=top_dets,
            winning_label=w_label,
            winning_count=w_count,
            agreement_ratio=agreement,
            winning_class_avg_conf=winning_avg,
            avg_cosine_similarity=avg_cos_sim,
            frames_with_face=n_face,
            frames_with_body_only=n_body_only,
        ))

    # Sort tracks by number of detections descending (most prominent first)
    track_results.sort(key=lambda tr: len(tr.frame_detections), reverse=True)

    return MultiVideoResult(
        tracks=track_results,
        total_frames_extracted=total_frames,
        frames_with_detections=frames_with_det,
        frames_without_detections=frames_without_det,
        all_frame_detections=all_frame_detections,
        classes=classes,
        noise_tracks_discarded=noise_discarded + merged_count,
    )


