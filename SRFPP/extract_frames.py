#!/usr/bin/env python3
"""
Script para extraer frames de videos y guardarlos en las carpetas del dataset.

Uso:
    # Opción 1: Videos en carpetas separadas (data/videos/cow_01/video1.mp4, etc.)
    python extract_frames.py --videos_dir data/videos --output_dir data/cows --stride 30 --max_frames_per_video 50

    # Opción 2: Videos directamente en las carpetas de vacas (data/cows/cow_01/video1.mp4, etc.)
    python extract_frames.py --videos_dir data/cows --output_dir data/cows --stride 30 --max_frames_per_video 50 --extract_from_same_dir

    # Opción 3: Un solo video para una vaca específica
    python extract_frames.py --video_path video.mp4 --cow_class cow_01 --output_dir data/cows --stride 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.video_utils import iter_video_frames


def extract_frames_from_video(
    video_path: str | Path,
    output_dir: Path,
    stride: int = 30,
    max_frames: int | None = 50,
    prefix: str = "frame",
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
        prefix: Prefijo para nombres de archivo
        filter_faces: Si True, solo guarda frames con rostro detectado
        require_features: Lista de características requeridas (ej: ["face", "eyes"])
    
    Returns:
        Número de frames guardados
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")
    
    saved_count = 0
    skipped_count = 0
    
    for frame in iter_video_frames(str(video_path), stride=stride, max_frames=max_frames):
        # Convertir numpy array a PIL Image
        img = Image.fromarray(frame.rgb)
        
        # Filtrar por detección de rostro si está habilitado
        if filter_faces:
            try:
                from src.face_detection import has_valid_animal_frame
                if not has_valid_animal_frame(img, require_features=require_features):
                    skipped_count += 1
                    continue
            except Exception as e:
                # Si falla la detección, guardar el frame de todas formas
                # (mejor tener datos que perderlos por un error)
                pass
        
        # Generar nombre único para cada frame
        frame_filename = f"{prefix}_f{frame.idx:06d}.jpg"
        frame_path = output_dir / frame_filename
        img.save(frame_path, quality=95)
        saved_count += 1
    
    if filter_faces and skipped_count > 0:
        print(f"  (Se omitieron {skipped_count} frames sin rostro detectado)")
    
    return saved_count


def main():
    p = argparse.ArgumentParser(
        description="Extrae frames de videos para crear dataset de entrenamiento"
    )
    
    # Dos modos: carpeta de videos o video individual
    p.add_argument(
        "--videos_dir",
        type=str,
        help="Carpeta con videos organizados por clase (ej: data/videos/cow_01/video1.mp4)",
    )
    p.add_argument(
        "--video_path",
        type=str,
        help="Ruta a un video individual (requiere --cow_class)",
    )
    p.add_argument(
        "--cow_class",
        type=str,
        help="Clase de vaca para --video_path (ej: cow_01)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Carpeta de salida donde guardar frames (ej: data/cows)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Extraer cada N frames (default: 30)",
    )
    p.add_argument(
        "--max_frames_per_video",
        type=int,
        default=50,
        help="Máximo frames a extraer por video (default: 50, None=sin límite)",
    )
    p.add_argument(
        "--extract_from_same_dir",
        action="store_true",
        help="Si videos_dir == output_dir, extraer frames en la misma carpeta donde está el video",
    )
    p.add_argument(
        "--filter_faces",
        action="store_true",
        help="Solo guardar frames con rostro detectado (mejora la calidad del dataset)",
    )
    p.add_argument(
        "--require_features",
        nargs="+",
        choices=["cow", "animal"],
        help="Tipo de detección requerida en los frames (ej: --require_features cow). Opciones: 'cow' (solo vacas), 'animal' (cualquier animal)",
    )
    
    args = p.parse_args()
    
    if args.video_path:
        # Modo: video individual
        if not args.cow_class:
            p.error("--cow_class es requerido cuando usas --video_path")
        
        output_cow_dir = Path(args.output_dir) / args.cow_class
        video_name = Path(args.video_path).stem
        prefix = f"{video_name}_"
        
        print(f"Extrayendo frames de: {args.video_path}")
        print(f"Guardando en: {output_cow_dir}")
        
        count = extract_frames_from_video(
            args.video_path,
            output_cow_dir,
            stride=args.stride,
            max_frames=args.max_frames_per_video,
            prefix=prefix,
            filter_faces=args.filter_faces,
            require_features=args.require_features,
        )
        print(f"✓ Guardados {count} frames en {output_cow_dir}")
        return
    
    if not args.videos_dir:
        p.error("Debes especificar --videos_dir o --video_path")
    
    # Modo: carpeta de videos
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    
    if not videos_dir.exists():
        raise FileNotFoundError(f"Carpeta de videos no encontrada: {videos_dir}")
    
    # Buscar carpetas de clases (cow_01, cow_02, etc.)
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
    cow_dirs = sorted([d for d in videos_dir.iterdir() if d.is_dir() and d.name.startswith("cow_")])
    
    if not cow_dirs:
        print(f"⚠ No se encontraron carpetas 'cow_XX' en {videos_dir}")
        print("Estructura esperada:")
        print("  videos_dir/")
        print("    cow_01/")
        print("      video1.mp4")
        print("      video2.mp4")
        print("    cow_02/")
        print("      ...")
        return
    
    total_frames = 0
    for cow_dir in cow_dirs:
        cow_class = cow_dir.name
        videos = [f for f in cow_dir.iterdir() if f.suffix in video_extensions]
        
        if not videos:
            print(f"⚠ No se encontraron videos en {cow_dir}")
            continue
        
        # Determinar carpeta de salida
        if args.extract_from_same_dir and videos_dir == output_dir:
            output_cow_dir = cow_dir  # Guardar en la misma carpeta
        else:
            output_cow_dir = output_dir / cow_class
        
        print(f"\n📁 Procesando {cow_class} ({len(videos)} videos)...")
        
        for video_path in videos:
            video_name = video_path.stem
            prefix = f"{video_name}_"
            
            try:
                count = extract_frames_from_video(
                    video_path,
                    output_cow_dir,
                    stride=args.stride,
                    max_frames=args.max_frames_per_video,
                    prefix=prefix,
                    filter_faces=args.filter_faces,
                    require_features=args.require_features,
                )
                total_frames += count
                print(f"  ✓ {video_path.name}: {count} frames → {output_cow_dir}")
            except Exception as e:
                print(f"  ✗ Error procesando {video_path.name}: {e}")
    
    print(f"\n✅ Total: {total_frames} frames extraídos")


if __name__ == "__main__":
    main()

