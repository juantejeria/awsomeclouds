"""Genera V2Recorte26marz: 21 frames PNG (10 antes + centro + 10 despues)
centrados en el frame donde la vaca esta mas al centro de la imagen (detectado por YOLO).

Para vacas 1-13 escanea el clip completo en checkpoints/Recorte26marz/<vaca>/<vaca>_clip.mp4.
Para vaca250/253 escanea +-3s alrededor del timestamp dado en el video completo.

Resolucion original conservada (PNG sin reescalar).
"""
import cv2
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(__file__).parent
SRC_BASE = PROJECT / 'checkpoints' / 'Recorte26marz'
DST_BASE = PROJECT / 'checkpoints' / 'V2Recorte26marz'

FRAMES_BEFORE = 10
FRAMES_AFTER = 10
SCAN_WINDOW_S = 3  # para videos completos (vaca250/253): +-3s alrededor del timestamp
YOLO_CONF = 0.15

CLIPS_VACAS = [f'vaca{i}' for i in range(1, 14)]

FULL_VIDEOS = {
    'vaca250': {
        'video': SRC_BASE / 'vaca250' / 'Vaca_250_VisualC.mp4',
        'hint_s': 1 * 60 + 32,
    },
    'vaca253': {
        'video': SRC_BASE / 'vaca253' / 'Vaca_253_VisualC.mp4',
        'hint_s': 1 * 60 + 42,
    },
}

# clase COCO 19 = 'cow'
COW_CLASS = 19


def find_centered_frame(model, video_path: Path, start_frame: int, end_frame: int):
    """Escanea [start_frame, end_frame] y devuelve el frame donde el centroide
    de la deteccion 'cow' mas grande esta mas cerca del centro de la imagen."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx_img, cy_img = w / 2, h / 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    best_frame = None
    best_dist = float('inf')
    best_bbox = None
    scanned = 0
    detected = 0
    for fnum in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        scanned += 1
        res = model.predict(frame, conf=YOLO_CONF, classes=[COW_CLASS], verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        detected += 1
        # bbox con mayor area (vaca dominante)
        xyxy = res.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        i = int(areas.argmax())
        x1, y1, x2, y2 = xyxy[i]
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        dist = ((bx - cx_img) ** 2 + (by - cy_img) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_frame = fnum
            best_bbox = (float(x1), float(y1), float(x2), float(y2))
    cap.release()
    return best_frame, best_dist, best_bbox, scanned, detected, (w, h)


def extract_around(video_path: Path, center_frame: int, out_dir: Path, name: str):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, center_frame - FRAMES_BEFORE)
    end = min(total - 1, center_frame + FRAMES_AFTER)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    saved = 0
    for fnum in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        saved += 1
        offset = fnum - center_frame
        sign = 'p' if offset >= 0 else 'm'
        tag = f'c{sign}{abs(offset):02d}'
        fname = out_dir / f'{name}_f{saved:03d}_{tag}.png'
        cv2.imwrite(str(fname), frame)
    cap.release()
    print(f"  guardados: {saved} PNG en {out_dir}")


def main():
    DST_BASE.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(PROJECT / 'yolov8n.pt'))

    # Vacas 1-13: escanear el clip completo
    for name in CLIPS_VACAS:
        clip = SRC_BASE / name / f'{name}_clip.mp4'
        if not clip.exists():
            print(f"[skip] {name}: no existe {clip}")
            continue
        print(f"\n== {name} ==")
        cap = cv2.VideoCapture(str(clip))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        best, dist, bbox, scanned, detected, (w, h) = find_centered_frame(
            model, clip, 0, total - 1
        )
        if best is None:
            print(f"  WARN: sin detecciones en {scanned} frames. Uso midpoint.")
            best = total // 2
        print(f"  scan=[0,{total-1}] detectados={detected}/{scanned} "
              f"mejor_frame={best} dist_centro={dist:.1f}px")
        extract_around(clip, best, DST_BASE / name, name)

    # Vaca250/253: escanear +-3s alrededor del hint
    for name, cfg in FULL_VIDEOS.items():
        video = cfg['video']
        if not video.exists():
            print(f"[skip] {name}: no existe {video}")
            continue
        print(f"\n== {name} ==")
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        hint_frame = int(round(cfg['hint_s'] * fps))
        scan_half = int(round(SCAN_WINDOW_S * fps))
        s = max(0, hint_frame - scan_half)
        e = min(total - 1, hint_frame + scan_half)
        best, dist, bbox, scanned, detected, (w, h) = find_centered_frame(
            model, video, s, e
        )
        if best is None:
            print(f"  WARN: sin detecciones. Uso hint={hint_frame}.")
            best = hint_frame
        print(f"  hint={hint_frame} scan=[{s},{e}] detectados={detected}/{scanned} "
              f"mejor_frame={best} dist_centro={dist:.1f}px")
        extract_around(video, best, DST_BASE / name, name)

    print("\nListo.")


if __name__ == '__main__':
    main()
