"""
Agrega los 21 frames de cada individuo de checkpoints/14mayo y checkpoints/20mayo
al dataset de barril training (_barril_training).

Para cada frame_*.jpg:
1. Detecta la vaca (YOLO COCO clase 19)
2. Segmenta el barril con barril_seg.pt (mismo modelo que procesar_21_frames.py)
3. Recorta al bbox con padding
4. Guarda img + mask en _barril_training
5. Agrega al indice como 'pending'

Idempotente: salta frames cuyo id ya existe en el indice.
"""
import cv2
import numpy as np
import json
import re
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(__file__).parent
BARRIL_DIR = PROJECT / 'output_modelos3d_grandes' / '_barril_training'
INDEX_FILE = BARRIL_DIR / 'frames_index.json'

SOURCES = {
    '14mayo': PROJECT / 'checkpoints' / '14mayo',
    '20mayo': PROJECT / 'checkpoints' / '20mayo',
}

PAD = 50          # padding alrededor del bbox para el recorte
MIN_W, MIN_H = 200, 150


def load_index():
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            return json.load(f)
    return []


def save_index(data):
    with open(INDEX_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sanitize(s):
    """Hace un string seguro para usar en id / nombre de archivo."""
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def detectar_y_segmentar(img, coco_model, barril_model):
    """Devuelve (bbox_xyxy, barril_binmask 0/1 tamano full) o (None, None).
    Replica la logica de procesar_21_frames.py._detectar_y_segmentar."""
    h_orig, w_orig = img.shape[:2]
    r_cow = coco_model(img, classes=[19], conf=0.2, verbose=False)
    if not r_cow or len(r_cow[0].boxes) == 0:
        return None, None
    boxes = r_cow[0].boxes.xyxy.cpu().numpy()
    scores = r_cow[0].boxes.conf.cpu().numpy()
    bi = int(np.argmax(scores))
    bx1, by1, bx2, by2 = [int(v) for v in boxes[bi]]
    pad = max(20, int(0.08 * max(bx2 - bx1, by2 - by1)))
    cx1, cy1 = max(0, bx1 - pad), max(0, by1 - pad)
    cx2, cy2 = min(w_orig, bx2 + pad), min(h_orig, by2 + pad)
    crop = img[cy1:cy2, cx1:cx2]
    r_bar = barril_model(crop, conf=0.25, verbose=False)
    if not r_bar or r_bar[0].masks is None or len(r_bar[0].masks.data) == 0:
        return (bx1, by1, bx2, by2), None
    masks = r_bar[0].masks.data.cpu().numpy()
    areas = np.array([float(np.sum(m)) for m in masks])
    keep = areas >= 0.05 * areas.max() if areas.max() > 0 else np.ones(len(masks), bool)
    sil = np.max(masks[keep], axis=0)
    if sil.shape != (crop.shape[0], crop.shape[1]):
        sil = cv2.resize(sil, (crop.shape[1], crop.shape[0]))
    binmask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    binmask[cy1:cy2, cx1:cx2] = (sil > 0.5).astype(np.uint8)
    return (bx1, by1, bx2, by2), binmask


def parse_frame_idx(stem):
    """frame_000_f32 -> 32 (numero de frame del video). Fallback: None."""
    m = re.search(r'_f(\d+)$', stem)
    return int(m.group(1)) if m else None


def main():
    print("Cargando modelos YOLO (COCO + barril_seg.pt)...")
    coco_model = YOLO(str(PROJECT / 'yolov8n.pt'))
    barril_model = YOLO(str(PROJECT / 'barril_seg.pt'))

    frames = load_index()
    existing_ids = {f['id'] for f in frames}

    added = 0
    no_vaca = 0
    no_barril = 0
    chico = 0
    skipped = 0

    for dataset, source_dir in SOURCES.items():
        if not source_dir.exists():
            print(f"  SKIP: {source_dir} no existe")
            continue
        print(f"\n=== {dataset} ({source_dir}) ===")

        for ind_dir in sorted(source_dir.iterdir()):
            if not ind_dir.is_dir():
                continue
            fotos = sorted(ind_dir.glob('frame_*.jpg'))
            if not fotos:
                continue

            individuo = f"{dataset}_{ind_dir.name}"      # p.ej. 14mayo_100_137.5
            ind_safe = sanitize(individuo)
            print(f"  {individuo}: {len(fotos)} frames")

            for i, foto_path in enumerate(fotos):
                fid = f"{ind_safe}_{sanitize(foto_path.stem)}"
                if fid in existing_ids:
                    skipped += 1
                    continue

                img = cv2.imread(str(foto_path))
                if img is None:
                    print(f"    ERROR leyendo {foto_path.name}")
                    continue

                bbox, binmask = detectar_y_segmentar(img, coco_model, barril_model)
                if bbox is None:
                    print(f"    {foto_path.name}: no se detecto vaca")
                    no_vaca += 1
                    continue
                if binmask is None:
                    print(f"    {foto_path.name}: no se detecto barril")
                    no_barril += 1
                    continue

                x1, y1, x2, y2 = bbox
                ry1 = max(0, y1 - PAD)
                ry2 = min(img.shape[0], y2 + PAD)
                rx1 = max(0, x1 - PAD)
                rx2 = min(img.shape[1], x2 + PAD)

                img_crop = img[ry1:ry2, rx1:rx2]
                mask_crop = (binmask[ry1:ry2, rx1:rx2] * 255).astype(np.uint8)

                if img_crop.shape[1] < MIN_W or img_crop.shape[0] < MIN_H:
                    print(f"    {foto_path.name}: crop muy chico "
                          f"({img_crop.shape[1]}x{img_crop.shape[0]}), skip")
                    chico += 1
                    continue

                # Limpieza morfologica ligera de la mascara
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, k)
                mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, k)

                img_filename = f"{fid}_img.png"
                mask_filename = f"{fid}_mask.png"
                cv2.imwrite(str(BARRIL_DIR / img_filename), img_crop)
                cv2.imwrite(str(BARRIL_DIR / mask_filename), mask_crop)

                frames.append({
                    'id': fid,
                    'individuo': individuo,
                    'video': '',
                    'frame_idx': parse_frame_idx(foto_path.stem) or i,
                    'bbox': [int(rx1), int(ry1), int(rx2), int(ry2)],
                    'crop_w': int(rx2 - rx1),
                    'crop_h': int(ry2 - ry1),
                    'img': img_filename,
                    'mask': mask_filename,
                    'status': 'pending',
                    'cuts': [],
                    'brush_rle': [],
                    'source': dataset,
                    'foto_original': foto_path.name,
                })
                existing_ids.add(fid)
                added += 1

    save_index(frames)
    print(f"\nResultado:")
    print(f"  Frames agregados:   {added}")
    print(f"  Ya existian (skip): {skipped}")
    print(f"  Sin vaca:           {no_vaca}")
    print(f"  Sin barril:         {no_barril}")
    print(f"  Crop muy chico:     {chico}")
    print(f"  Total en dataset:   {len(frames)}")


if __name__ == '__main__':
    main()
