"""
Gestión de datasets: establecimientos y animales.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json, write_json
from .video_utils import iter_video_frames


def get_data_base_dir() -> Path:
    """Retorna el directorio base de datos."""
    return Path("data")


def get_artifacts_base_dir() -> Path:
    """Retorna el directorio base de artifacts."""
    return Path("artifacts")


def list_establecimientos() -> list[str]:
    """Lista todos los establecimientos disponibles (carpetas en data/)."""
    base = get_data_base_dir()
    if not base.exists():
        return []
    
    establecimientos = []
    for item in base.iterdir():
        if item.is_dir() and (item / "cows").exists():
            establecimientos.append(item.name)
    
    return sorted(establecimientos)


def create_establecimiento(nombre: str) -> Path:
    """Crea un nuevo establecimiento."""
    nombre_sanitizado = _sanitize_name(nombre)
    base = get_data_base_dir()
    estab_dir = ensure_dir(base / nombre_sanitizado / "cows")
    write_json(
        base / nombre_sanitizado / "meta.json",
        {
            "display_name": nombre.strip(),
            "created_at": datetime.utcnow().isoformat(),
        },
    )
    return estab_dir.parent


def get_establecimiento_display_name(establecimiento: str) -> str:
    """Obtiene el nombre visible de un establecimiento."""
    base = get_data_base_dir()
    meta_path = base / establecimiento / "meta.json"
    if meta_path.exists():
        try:
            meta = read_json(meta_path)
            display_name = meta.get("display_name")
            if isinstance(display_name, str) and display_name.strip():
                return display_name.strip()
        except Exception:
            pass
    return establecimiento.replace("_", " ")


def update_establecimiento_display_name(establecimiento: str, display_name: str) -> None:
    """Actualiza el nombre visible de un establecimiento."""
    base = get_data_base_dir()
    meta_path = base / establecimiento / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = read_json(meta_path)
        except Exception:
            meta = {}
    meta["display_name"] = display_name.strip()
    meta.setdefault("created_at", datetime.utcnow().isoformat())
    meta["updated_at"] = datetime.utcnow().isoformat()
    write_json(meta_path, meta)


def get_establecimiento_dir(nombre: str) -> Path:
    """Retorna el directorio de un establecimiento."""
    return get_data_base_dir() / nombre / "cows"


def list_animales(establecimiento: str) -> list[str]:
    """Lista los animales de un establecimiento."""
    estab_dir = get_establecimiento_dir(establecimiento)
    if not estab_dir.exists():
        return []
    
    animales = []
    for item in estab_dir.iterdir():
        if item.is_dir():
            animales.append(item.name)
    
    return sorted(animales)


def create_animal(establecimiento: str, nombre_animal: str) -> Path:
    """Crea un nuevo animal en un establecimiento."""
    nombre_sanitizado = _sanitize_name(nombre_animal)
    estab_dir = get_establecimiento_dir(establecimiento)
    animal_dir = ensure_dir(estab_dir / nombre_sanitizado)
    write_json(
        animal_dir / "meta.json",
        {
            "display_name": nombre_animal.strip(),
            "created_at": datetime.utcnow().isoformat(),
        },
    )
    return animal_dir


def get_animal_dir(establecimiento: str, animal: str) -> Path:
    """Retorna el directorio de un animal."""
    return get_establecimiento_dir(establecimiento) / animal


def get_animal_display_name(establecimiento: str, animal: str) -> str:
    """Obtiene el nombre visible de un animal."""
    animal_dir = get_animal_dir(establecimiento, animal)
    meta_path = animal_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = read_json(meta_path)
            display_name = meta.get("display_name")
            if isinstance(display_name, str) and display_name.strip():
                return display_name.strip()
        except Exception:
            pass
    return animal.replace("_", " ")


def count_images_in_dir(dir_path: Path) -> int:
    """Cuenta imágenes en un directorio."""
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    count = 0
    if dir_path.exists():
        for file in dir_path.iterdir():
            if file.is_file() and file.suffix in image_extensions:
                count += 1
    return count


def count_frames_in_videos(dir_path: Path, stride: int = 30, max_frames_per_video: int = 50) -> int:
    """Cuenta frames que se extraerían de videos en un directorio."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
    total_frames = 0
    
    if not dir_path.exists():
        return 0
    
    for file in dir_path.iterdir():
        if file.is_file() and file.suffix in video_extensions:
            try:
                frame_count = sum(1 for _ in iter_video_frames(str(file), stride=stride, max_frames=max_frames_per_video))
                total_frames += frame_count
            except Exception:
                pass  # Ignorar videos corruptos
    
    return total_frames


def get_dataset_stats(establecimiento: str) -> dict[str, Any]:
    """Obtiene estadísticas del dataset de un establecimiento."""
    estab_dir = get_establecimiento_dir(establecimiento)
    animales = list_animales(establecimiento)
    
    stats = {
        "total_animales": len(animales),
        "animales": {},
        "balanceado": False,
        "min_samples": 0,
        "max_samples": 0,
        "promedio_samples": 0.0,
    }
    
    if not animales:
        return stats
    
    samples_per_animal = []
    
    for animal in animales:
        animal_dir = get_animal_dir(establecimiento, animal)
        image_count = count_images_in_dir(animal_dir)
        video_frames = count_frames_in_videos(animal_dir)
        total_samples = image_count + video_frames
        
        stats["animales"][animal] = {
            "imagenes": image_count,
            "frames_video": video_frames,
            "total": total_samples,
        }
        
        samples_per_animal.append(total_samples)
    
    if samples_per_animal:
        stats["min_samples"] = min(samples_per_animal)
        stats["max_samples"] = max(samples_per_animal)
        stats["promedio_samples"] = sum(samples_per_animal) / len(samples_per_animal)
        
        # Considerar balanceado si la diferencia entre min y max es <= 20% del promedio
        diff = stats["max_samples"] - stats["min_samples"]
        threshold = stats["promedio_samples"] * 0.2
        stats["balanceado"] = diff <= threshold and stats["min_samples"] > 0
    
    return stats


def validate_dataset_for_training(establecimiento: str, min_samples_per_animal: int = 2) -> tuple[bool, list[str]]:
    """Valida que el dataset esté listo para entrenar."""
    errors = []
    stats = get_dataset_stats(establecimiento)
    
    if stats["total_animales"] < 2:
        errors.append(f"Se necesitan al menos 2 animales. Actualmente hay {stats['total_animales']}.")
    
    # Solo validar mínimo de muestras si se especifica un valor mayor a 0
    if min_samples_per_animal > 0:
        for animal, animal_stats in stats["animales"].items():
            if animal_stats["total"] < min_samples_per_animal:
                errors.append(
                    f"Animal '{animal}': tiene {animal_stats['total']} muestras. "
                    f"Mínimo requerido: {min_samples_per_animal}."
                )
    
    # No agregar error de desbalanceo como bloqueante, solo como advertencia
    # El balanceo automático puede solucionarlo
    # if not stats["balanceado"]:
    #     errors.append(
    #         f"Dataset desbalanceado: mínimo {stats['min_samples']} muestras, "
    #         f"máximo {stats['max_samples']} muestras. "
    #         f"Diferencia: {stats['max_samples'] - stats['min_samples']}."
    #     )
    
    return len(errors) == 0, errors


def save_uploaded_file(uploaded_file, destino: Path) -> Path:
    """Guarda un archivo subido en el destino especificado."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destino, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return destino


def extract_frames_from_uploaded_video(
    video_path: Path, 
    output_dir: Path, 
    stride: int = 30, 
    max_frames: int = 50,
    filter_faces: bool = False,
    require_features: list[str] | None = None,
    crop_to_face: bool = True,
) -> int:
    """
    Extrae frames de un video y los guarda como imágenes.

    When ``crop_to_face`` is True (default), each frame is run through the
    cattle-face detector.  If a face is found the saved image is the face
    crop; otherwise it falls back to a body crop; and if neither is
    detected the full frame is saved (or skipped if ``filter_faces`` is
    also True).

    Args:
        video_path: Ruta al video
        output_dir: Directorio de salida
        stride: Cada cuántos frames extraer
        max_frames: Máximo frames a extraer
        filter_faces: Si True, solo guarda frames con animal detectado
        require_features: Lista de características requeridas (ej: ["face", "eyes"])
        crop_to_face: Si True, recorta cada frame al rostro / cuerpo
                      detectado antes de guardar (recomendado para
                      datasets de reconocimiento facial).

    Returns:
        Número de frames guardados
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    skipped_count = 0
    face_crop_count = 0
    body_crop_count = 0
    
    for frame in iter_video_frames(str(video_path), stride=stride, max_frames=max_frames):
        from PIL import Image
        import numpy as np
        img = Image.fromarray(frame.rgb)

        save_img = img  # default: full frame

        # ------------------------------------------------------------------
        # Face / body crop
        # ------------------------------------------------------------------
        if crop_to_face:
            try:
                from .face_detection import detect_best_boxes
                detections = detect_best_boxes(frame.rgb)
                if detections:
                    best = detections[0]
                    px1, py1, px2, py2 = best["bbox_padded"]
                    crop = frame.rgb[int(py1):int(py2), int(px1):int(px2)]
                    if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                        save_img = Image.fromarray(crop)
                        if best.get("is_face", False):
                            face_crop_count += 1
                        else:
                            body_crop_count += 1
                    elif filter_faces:
                        skipped_count += 1
                        continue
                elif filter_faces:
                    # No detection at all and filtering is on → skip
                    skipped_count += 1
                    continue
            except Exception:
                pass  # keep full frame on error

        # ------------------------------------------------------------------
        # Legacy filter (non-crop mode): just check presence
        # ------------------------------------------------------------------
        elif filter_faces:
            try:
                if require_features is None:
                    from .face_detection import detect_cow_specifically
                    has_valid = detect_cow_specifically(img, min_confidence=0.3)
                else:
                    from .face_detection import has_valid_animal_frame
                    has_valid = has_valid_animal_frame(img, require_features=require_features)

                if not has_valid:
                    skipped_count += 1
                    continue
            except Exception:
                pass

        frame_filename = f"vid_frame_{frame.idx:06d}.jpg"
        frame_path = output_dir / frame_filename
        save_img.save(frame_path, quality=95)
        saved_count += 1
    
    if skipped_count > 0:
        print(f"  (Se omitieron {skipped_count} frames sin detección)")
    if crop_to_face:
        print(f"  Crops: {face_crop_count} rostros, {body_crop_count} cuerpos, "
              f"{saved_count - face_crop_count - body_crop_count} sin crop")
    
    return saved_count


def crop_image_to_face(
    image_path: Path,
    output_path: Path | None = None,
    min_confidence: float = 0.25,
) -> bool:
    """
    Detect the cow face (or body fallback) in an image and overwrite/save
    the file with only the cropped region.

    Args:
        image_path: Path to the source image.
        output_path: Where to save the crop. If *None*, overwrites the
                     original file.
        min_confidence: Minimum YOLO confidence for face/body detection.

    Returns:
        True if a crop was applied, False if no detection was found
        (the original image is kept as-is in that case).
    """
    from PIL import Image as PILImage
    import numpy as np
    from .face_detection import detect_best_boxes

    try:
        img = PILImage.open(image_path).convert("RGB")
        img_array = np.array(img)

        detections = detect_best_boxes(
            img_array,
            face_min_confidence=min_confidence,
            body_min_confidence=min_confidence + 0.05,
        )

        if not detections:
            # No detection – keep the full image
            if output_path and output_path != image_path:
                img.save(output_path, quality=95)
            return False

        best = detections[0]
        px1, py1, px2, py2 = best["bbox_padded"]
        crop = img_array[int(py1):int(py2), int(px1):int(px2)]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            if output_path and output_path != image_path:
                img.save(output_path, quality=95)
            return False

        crop_img = PILImage.fromarray(crop)
        dest = output_path or image_path
        crop_img.save(dest, quality=95)
        return True

    except Exception:
        # On any error, keep the original image untouched
        if output_path and output_path != image_path:
            try:
                shutil.copy2(str(image_path), str(output_path))
            except Exception:
                pass
        return False


def crop_dataset_to_faces(
    data_dir: Path,
    output_dir: Path,
    min_confidence: float = 0.25,
) -> dict:
    """
    Create a face-cropped copy of an entire ImageFolder dataset.

    For each image in *data_dir* (organised as class-sub-folders), the
    function detects the cow face (with body fallback) and writes the
    cropped version to the mirror location under *output_dir*.  If no
    detection is found the full image is copied unchanged.

    Args:
        data_dir: Root of the source ImageFolder.
        output_dir: Root of the destination (created if it doesn't exist).
        min_confidence: Minimum YOLO confidence.

    Returns:
        Dictionary with stats:
        ``{"total", "face_cropped", "body_cropped", "no_detection"}``
    """
    from PIL import Image as PILImage
    import numpy as np
    from .face_detection import detect_best_boxes

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    stats = {"total": 0, "face_cropped": 0, "body_cropped": 0, "no_detection": 0}

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in img_exts:
                # Copy non-image files (meta.json, etc.) as-is
                dest = out_class_dir / img_path.name
                if img_path.is_file():
                    shutil.copy2(str(img_path), str(dest))
                continue

            stats["total"] += 1
            dest = out_class_dir / img_path.name

            try:
                img = PILImage.open(img_path).convert("RGB")
                img_array = np.array(img)

                detections = detect_best_boxes(
                    img_array,
                    face_min_confidence=min_confidence,
                    body_min_confidence=min_confidence + 0.05,
                )

                if not detections:
                    img.save(dest, quality=95)
                    stats["no_detection"] += 1
                    continue

                best = detections[0]
                px1, py1, px2, py2 = best["bbox_padded"]
                crop = img_array[int(py1):int(py2), int(px1):int(px2)]

                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    img.save(dest, quality=95)
                    stats["no_detection"] += 1
                    continue

                PILImage.fromarray(crop).save(dest, quality=95)

                if best.get("is_face", False):
                    stats["face_cropped"] += 1
                else:
                    stats["body_cropped"] += 1

            except Exception:
                # On error, copy original
                try:
                    shutil.copy2(str(img_path), str(dest))
                except Exception:
                    pass
                stats["no_detection"] += 1

            if stats["total"] % 20 == 0:
                print(f"   Pre-procesando imágenes... {stats['total']} procesadas", end="\r")

    print(f"   Pre-procesamiento completado: {stats['total']} imágenes                ")
    return stats


def _sanitize_name(name: str) -> str:
    """Sanitiza un nombre para usarlo como nombre de carpeta."""
    # Reemplazar espacios y caracteres especiales
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name.strip('_').lower()

