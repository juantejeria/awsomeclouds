"""Corta clips de 6s (±3s) del video Desfile_VisualC.mp4 para cada vaca y extrae frames."""
import cv2
import shutil
from pathlib import Path

PROJECT = Path(__file__).parent
VIDEO = PROJECT / 'checkpoints' / 'Recorte26marz' / 'Desfile_VisualC.mp4'
BASE = PROJECT / 'checkpoints' / 'Recorte26marz'

TIMESTAMPS = {
    'vaca1':  (1, 25),
    'vaca2':  (3, 45),
    'vaca3':  (5, 7),
    'vaca4':  (5, 51),
    'vaca5':  (7, 45),
    'vaca6':  (11, 19),
    'vaca7':  (12, 40),
    'vaca8':  (16, 9),
    'vaca9':  (17, 19),
    'vaca10': (18, 14),
    'vaca11': (19, 11),
    'vaca12': (20, 1),
    'vaca13': (22, 11),
}

WINDOW_S = 3
FRAME_EVERY_N = 6  # cada 6 frames originales → ~5 fps si el video es 30fps

def main():
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total_frames / fps
    print(f"Video: {w}x{h} @ {fps:.2f} fps, {total_frames} frames, {dur:.1f}s ({dur/60:.1f} min)")

    for name, (mm, ss) in TIMESTAMPS.items():
        center = mm * 60 + ss
        start = max(0, center - WINDOW_S)
        end = center + WINDOW_S
        print(f"\n{name}: centro={mm}:{ss:02d}  ventana=[{start}s, {end}s]")

        vaca_dir = BASE / name
        vaca_dir.mkdir(exist_ok=True)

        # Respaldar pngs/jpgs viejos
        old_dir = vaca_dir / '_old_screenshots'
        moved = 0
        for f in list(vaca_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg') and not f.name.startswith(name):
                old_dir.mkdir(exist_ok=True)
                shutil.move(str(f), str(old_dir / f.name))
                moved += 1
        if moved:
            print(f"  movidos {moved} screenshots viejos a _old_screenshots/")

        # Seek al inicio
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # VideoWriter para el clip
        clip_path = vaca_dir / f'{name}_clip.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))

        frames_written = 0
        frames_extracted = 0
        current = start_frame
        while current < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frames_written += 1
            # Extraer cada N frames
            if (current - start_frame) % FRAME_EVERY_N == 0:
                frames_extracted += 1
                fname = vaca_dir / f'{name}_frame{frames_extracted:03d}.png'
                cv2.imwrite(str(fname), frame)
            current += 1

        writer.release()
        print(f"  clip: {frames_written} frames en {clip_path.name}")
        print(f"  extracción: {frames_extracted} jpgs")

    cap.release()
    print("\nListo.")

if __name__ == '__main__':
    main()
