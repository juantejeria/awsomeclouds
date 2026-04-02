"""
Extrae rostros de vacas desde los videos de entrega_reconocimiento.

Para cada video en data/Entrga_Reconocimiento/vaca_N/vacaN.MOV:
  - Lee el video frame a frame
  - Detecta rostros de vaca usando el modelo especializado + validacion cuerpo
  - Recorta el rostro y lo guarda como PNG en la misma carpeta
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Agregar src al path para importar face_detection
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from face_detection import detect_cow_face_boxes, detect_animal_boxes

BASE_DIR = Path(__file__).resolve().parent / "data" / "Entrga_Reconocimiento"
FRAME_INTERVAL = 5
FACE_MIN_CONFIDENCE = 0.45
BODY_MIN_CONFIDENCE = 0.30
MIN_CROP_SIZE = 50
# Proporcion frontal del cuerpo a recortar como "rostro" (40% desde la izq o der)
HEAD_RATIO = 0.40


def process_video(video_path: Path, output_dir: Path) -> int:
    """Procesa un video y guarda recortes de rostros. Retorna cantidad guardada."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: No se pudo abrir {video_path.name}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {video_path.name} | {total_frames} frames | {fps:.1f} FPS")

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL != 0:
            frame_idx += 1
            continue

        # Convertir BGR -> RGB para el modelo de deteccion
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1) Intentar deteccion de rostro directa
        face_dets = detect_cow_face_boxes(
            rgb,
            min_confidence=FACE_MIN_CONFIDENCE,
            padding_factor=0.15,
        )

        for det_idx, det in enumerate(face_dets):
            x1, y1, x2, y2 = [int(v) for v in det["bbox_padded"]]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                continue
            crop = frame[y1:y2, x1:x2]
            conf = det["confidence"]
            fname = f"frame{frame_idx:05d}_face{det_idx}_conf{conf:.2f}.png"
            cv2.imwrite(str(output_dir / fname), crop)
            saved += 1

        # 2) Si no hubo rostros, usar cuerpo y recortar porcion frontal
        if not face_dets:
            body_dets = detect_animal_boxes(
                rgb,
                min_confidence=BODY_MIN_CONFIDENCE,
                prefer_cow=True,
                padding_factor=0.10,
            )
            for det_idx, det in enumerate(body_dets):
                if not det["is_cow"]:
                    continue
                x1, y1, x2, y2 = [int(v) for v in det["bbox_padded"]]
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                body_w = x2 - x1
                body_h = y2 - y1
                if body_w < MIN_CROP_SIZE or body_h < MIN_CROP_SIZE:
                    continue
                # Recortar la porcion frontal (cabeza)
                # La cabeza puede estar a la izq o der segun orientacion
                # Tomamos el lado mas cuadrado (la cabeza tiende a ser cuadrada)
                head_w = int(body_w * HEAD_RATIO)
                # Crop izquierdo
                crop_left = frame[y1:y2, x1:x1 + head_w]
                # Crop derecho
                crop_right = frame[y1:y2, x2 - head_w:x2]
                # Guardar ambos, el usuario puede filtrar luego
                conf = det["confidence"]
                if head_w >= MIN_CROP_SIZE:
                    fname_l = f"frame{frame_idx:05d}_body{det_idx}_left_conf{conf:.2f}.png"
                    fname_r = f"frame{frame_idx:05d}_body{det_idx}_right_conf{conf:.2f}.png"
                    cv2.imwrite(str(output_dir / fname_l), crop_left)
                    cv2.imwrite(str(output_dir / fname_r), crop_right)
                    saved += 2

        frame_idx += 1

    cap.release()
    return saved


def main():
    print(f"Directorio base: {BASE_DIR}")
    if not BASE_DIR.exists():
        print("ERROR: No existe el directorio de entrega")
        return

    total_saved = 0

    for i in range(3, 4):  # Solo vaca_3
        folder = BASE_DIR / f"vaca_{i}"
        video_file = folder / f"vaca{i}.MOV"

        print(f"\n=== Vaca {i} ===")
        if not video_file.exists():
            print(f"  SKIP: {video_file} no existe")
            continue

        count = process_video(video_file, folder)
        total_saved += count
        print(f"  -> {count} rostros guardados en {folder.name}/")

    print(f"\n=== TOTAL: {total_saved} rostros extraidos ===")


if __name__ == "__main__":
    main()
