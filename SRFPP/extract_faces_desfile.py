"""
Extrae rostros de vacas del dataset desfile26marz usando el pipeline original
(ByteTrack + face model + body validation).

Para cada video: detecta rostros frame a frame, recorta y guarda como PNG
en la misma carpeta del video.
"""

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np

from src.face_detection import (
    get_face_model,
    detect_animal_boxes,
    detect_cow_face_boxes,
)

BASE_DIR = Path(__file__).resolve().parent / "data" / "desfile26marz"
STRIDE = 3            # cada 3 frames (~10 fps efectivos)
MIN_CONFIDENCE = 0.30 # confianza face model
MIN_CROP_SIZE = 40    # px minimo de recorte
PADDING = 0.15        # padding alrededor del rostro


def extract_faces_from_video(video_path: Path, output_dir: Path) -> int:
    """Extrae rostros de un video usando face detection + body validation."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: No se pudo abrir {video_path.name}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {video_path.name} | {total_frames} frames | {fps:.0f} FPS")

    # Usar YOLO track con ByteTrack para IDs estables
    face_model = get_face_model()
    if face_model is None:
        print("  ERROR: No se encontro el modelo de face detection")
        return 0

    # Reset tracker
    if hasattr(face_model, 'predictor') and face_model.predictor is not None:
        face_model.predictor = None

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % STRIDE != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Face detection con ByteTrack tracking
        try:
            results = face_model.track(
                rgb, persist=True, conf=MIN_CONFIDENCE, verbose=False,
            )
        except Exception:
            results = face_model(rgb, conf=MIN_CONFIDENCE, verbose=False)

        face_dets = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = (
                results[0].boxes.id.cpu().numpy().astype(int)
                if results[0].boxes.id is not None
                else list(range(len(boxes)))
            )

            for box, conf, tid in zip(boxes, confs, track_ids):
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                bw, bh = x2 - x1, y2 - y1
                # Aplicar padding
                px1 = max(0, int(x1 - bw * PADDING))
                py1 = max(0, int(y1 - bh * PADDING))
                px2 = min(w, int(x2 + bw * PADDING))
                py2 = min(h, int(y2 + bh * PADDING))

                face_dets.append({
                    "bbox_padded": [px1, py1, px2, py2],
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    "confidence": float(conf),
                    "track_id": int(tid),
                })

        if not face_dets:
            frame_idx += 1
            continue

        # Body validation: verificar que hay una vaca real
        body_dets = detect_animal_boxes(
            rgb, min_confidence=0.25, prefer_cow=True, padding_factor=0.25,
        )
        cow_bodies = [b for b in body_dets if b.get("is_cow", False)]

        # Validar cada rostro contra cuerpos de vaca
        for det in face_dets:
            cx, cy = det["center"]
            valid = False

            if cow_bodies:
                for b in cow_bodies:
                    bx1, by1, bx2, by2 = b["bbox_padded"]
                    bw, bh = bx2 - bx1, by2 - by1
                    margin = 0.30
                    if (bx1 - bw * margin <= cx <= bx2 + bw * margin and
                        by1 - bh * margin <= cy <= by2 + bh * margin):
                        valid = True
                        break
            else:
                # Si no hay body dets, aceptar con alta confianza
                valid = det["confidence"] >= 0.50

            if not valid:
                continue

            px1, py1, px2, py2 = det["bbox_padded"]
            crop_w = px2 - px1
            crop_h = py2 - py1
            if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
                continue

            crop = frame[py1:py2, px1:px2]  # BGR para cv2.imwrite
            conf = det["confidence"]
            tid = det["track_id"]
            fname = f"frame{frame_idx:05d}_t{tid}_conf{conf:.2f}.png"
            cv2.imwrite(str(output_dir / fname), crop)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def extract_face_from_image(img_path: Path, output_dir: Path) -> int:
    """Extrae rostros de una imagen estática."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ERROR: No se pudo leer {img_path.name}")
        return 0

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    print(f"  Imagen: {img_path.name} | {w}x{h}")

    face_dets = detect_cow_face_boxes(rgb, min_confidence=MIN_CONFIDENCE, padding_factor=PADDING)
    saved = 0

    for i, det in enumerate(face_dets):
        x1, y1, x2, y2 = [int(v) for v in det["bbox_padded"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
            continue
        crop = img[y1:y2, x1:x2]
        conf = det["confidence"]
        fname = f"img_det{i}_conf{conf:.2f}.png"
        cv2.imwrite(str(output_dir / fname), crop)
        saved += 1

    return saved


def main():
    print(f"Dataset: {BASE_DIR}")
    total_saved = 0

    for folder in sorted(BASE_DIR.iterdir()):
        if not folder.is_dir():
            continue

        print(f"\n=== {folder.name} ===")

        # Buscar videos
        videos = list(folder.glob("*.mp4")) + list(folder.glob("*.MOV")) + list(folder.glob("*.mov"))
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        # Excluir PNGs que ya son recortes previos (empiezan con frame o img)
        images = [im for im in images if not im.stem.startswith(("frame", "img_det"))]

        if not videos and not images:
            print("  SKIP: sin archivos")
            continue

        count = 0
        for video in videos:
            count += extract_faces_from_video(video, folder)

        for img in images:
            count += extract_face_from_image(img, folder)

        total_saved += count
        print(f"  -> {count} rostros guardados en {folder.name}/")

    print(f"\n=== TOTAL: {total_saved} rostros extraidos ===")


if __name__ == "__main__":
    main()
