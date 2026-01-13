"""
Gestión de datasets: establecimientos y animales.
"""

from __future__ import annotations

import json
import re
import shutil
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
    return estab_dir.parent


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
    return animal_dir


def get_animal_dir(establecimiento: str, animal: str) -> Path:
    """Retorna el directorio de un animal."""
    return get_establecimiento_dir(establecimiento) / animal


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
) -> int:
    """
    Extrae frames de un video y los guarda como imágenes.
    
    Args:
        video_path: Ruta al video
        output_dir: Directorio de salida
        stride: Cada cuántos frames extraer
        max_frames: Máximo frames a extraer
        filter_faces: Si True, solo guarda frames con rostro detectado
        require_features: Lista de características requeridas (ej: ["face", "eyes"])
    
    Returns:
        Número de frames guardados
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    skipped_count = 0
    
    for frame in iter_video_frames(str(video_path), stride=stride, max_frames=max_frames):
        from PIL import Image
        img = Image.fromarray(frame.rgb)
        
        # Filtrar por detección de rostro si está habilitado
        if filter_faces:
            try:
                # Por defecto, requerir específicamente vacas para entrenamiento de individuos
                if require_features is None:
                    from .face_detection import detect_cow_specifically
                    has_valid = detect_cow_specifically(img, min_confidence=0.3)
                else:
                    from .face_detection import has_valid_animal_frame
                    has_valid = has_valid_animal_frame(img, require_features=require_features)
                
                if not has_valid:
                    skipped_count += 1
                    continue
            except Exception as e:
                # Si falla la detección, guardar el frame de todas formas
                # (mejor tener datos que perderlos por un error)
                pass
        
        frame_filename = f"frame_{frame.idx:06d}.jpg"
        frame_path = output_dir / frame_filename
        img.save(frame_path, quality=95)
        saved_count += 1
    
    if filter_faces and skipped_count > 0:
        print(f"  (Se omitieron {skipped_count} frames sin rostro detectado)")
    
    return saved_count


def _sanitize_name(name: str) -> str:
    """Sanitiza un nombre para usarlo como nombre de carpeta."""
    # Reemplazar espacios y caracteres especiales
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name.strip('_').lower()

