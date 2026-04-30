"""Regenera los frames de una carpeta extrayendo directo del video original
con OpenCV (calidad 95). Reemplaza los JPGs degradados por canvas+toBlob.

Uso:
    python regenerar_frames_desde_video.py <carpeta_frames> <video_path>

Ejemplo:
    python regenerar_frames_desde_video.py \\
        checkpoints/22abril/central160_20260429_110236 \\
        checkpoints/22abril/v1-22abril-165.MOV
"""
import os
import sys
import cv2
from pathlib import Path

if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)
folder = Path(sys.argv[1])
video = Path(sys.argv[2])

if not folder.is_dir():
    print(f"[error] no existe carpeta: {folder}"); sys.exit(1)
if not video.exists():
    print(f"[error] no existe video: {video}"); sys.exit(1)

# Para cada frame_*_f<n>.jpg, extraer el frame n del video y reemplazar
files = sorted(folder.glob('frame_*.jpg'))
cap = cv2.VideoCapture(str(video))
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[video] {video.name}: {w}x{h}, {total_video_frames} frames")
print(f"[folder] {len(files)} jpgs a regenerar\n")

regenerados = 0
for fp in files:
    # Parsear el frame number del nombre: frame_<sign>_f<n>.jpg
    parts = fp.name.replace('.jpg', '').split('_')
    if len(parts) < 3 or not parts[2].startswith('f'):
        print(f"  {fp.name}: nombre no parseable, saltando")
        continue
    try:
        fn = int(parts[2][1:])
    except ValueError:
        print(f"  {fp.name}: frame_num no parseable")
        continue
    if fn < 0 or fn >= total_video_frames:
        print(f"  {fp.name}: frame {fn} fuera del rango [0,{total_video_frames-1}]")
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"  {fp.name}: no se pudo leer frame {fn}")
        continue
    cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    regenerados += 1
    print(f"  {fp.name}: regenerado desde video frame {fn}")
cap.release()
print(f"\n[ok] {regenerados}/{len(files)} frames regenerados (calidad 95)")
