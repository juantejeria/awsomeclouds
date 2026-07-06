"""
Agrega frames de checkpoints/desfile26marz y checkpoints/26marz
al dataset de silueta completa (_silueta_training).

Para cada screenshot:
1. Detecta la vaca (YOLO cow.pt / COCO fallback)
2. Recorta al bbox
3. Genera mascara GrabCut como base
4. Guarda img + mask en _silueta_training
5. Agrega al indice
"""
import cv2
import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from ultralytics import YOLO
from core.generar_modelos3d_grandes import detectar_vaca, segmentar

PROJECT = Path(__file__).resolve().parents[1]
SILUETA_DIR = PROJECT / 'output_modelos3d_grandes' / '_silueta_training'
INDEX_FILE = SILUETA_DIR / 'frames_index.json'

SOURCES = [
    PROJECT / 'checkpoints' / 'desfile26marz',
    PROJECT / 'checkpoints' / '26marz',
    PROJECT / 'checkpoints' / 'V2Recorte26marz',
]


def load_index():
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            return json.load(f)
    return []


def save_index(data):
    with open(INDEX_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_id(individuo, foto_name, idx):
    """Genera un ID unico para el frame."""
    safe = foto_name.replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '')
    return f"{individuo}_{safe}_{idx:04d}"


def main():
    print("Cargando modelos YOLO...")
    cow_model = YOLO(str(PROJECT / 'models' / 'cow.pt'))
    coco_model = YOLO(str(PROJECT / 'yolov8n.pt'))

    frames = load_index()
    existing_ids = {f['id'] for f in frames}

    added = 0
    failed = 0

    for source_dir in SOURCES:
        if not source_dir.exists():
            print(f"  SKIP: {source_dir} no existe")
            continue
        source_name = source_dir.name
        print(f"\nProcesando {source_name}...")

        for ind_dir in sorted(source_dir.iterdir()):
            if not ind_dir.is_dir():
                continue
            individuo = ind_dir.name
            fotos = sorted([f for f in ind_dir.iterdir()
                          if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
            if not fotos:
                continue

            print(f"  {individuo}: {len(fotos)} fotos")

            for idx, foto_path in enumerate(fotos):
                fid = make_id(individuo, foto_path.stem, idx)

                if fid in existing_ids:
                    continue

                img = cv2.imread(str(foto_path))
                if img is None:
                    print(f"    ERROR leyendo {foto_path.name}")
                    failed += 1
                    continue

                # Detectar vaca
                bbox = detectar_vaca(img, cow_model, coco_model)
                if bbox is None:
                    print(f"    {foto_path.name}: no se detecto vaca")
                    failed += 1
                    continue

                x1, y1, x2, y2 = bbox
                crop_w = int(x2 - x1)
                crop_h = int(y2 - y1)

                # Generar mascara GrabCut
                mask_full, contorno = segmentar(img, bbox, nombre_foto=foto_path.name)
                if mask_full is None:
                    print(f"    {foto_path.name}: segmentacion fallo")
                    failed += 1
                    continue

                # Recortar al bbox con padding generoso para no cortar el animal
                pad = 50
                ry1 = max(0, y1 - pad)
                ry2 = min(img.shape[0], y2 + pad)
                rx1 = max(0, x1 - pad)
                rx2 = min(img.shape[1], x2 + pad)

                img_crop = img[ry1:ry2, rx1:rx2]
                mask_crop = mask_full[ry1:ry2, rx1:rx2]

                # Filtro de resolucion minima: si el crop es muy chico, no sirve
                if img_crop.shape[1] < 200 or img_crop.shape[0] < 150:
                    print(f"    {foto_path.name}: crop muy chico ({img_crop.shape[1]}x{img_crop.shape[0]}), skip")
                    failed += 1
                    continue

                # Guardar como PNG para no perder calidad
                img_filename = f"{fid}_img.png"
                mask_filename = f"{fid}_mask.png"

                cv2.imwrite(str(SILUETA_DIR / img_filename), img_crop)
                cv2.imwrite(str(SILUETA_DIR / mask_filename), mask_crop)

                frames.append({
                    'id': fid,
                    'individuo': individuo,
                    'video': '',
                    'frame_idx': idx,
                    'bbox': [int(rx1), int(ry1), int(rx2), int(ry2)],
                    'crop_w': int(rx2 - rx1),
                    'crop_h': int(ry2 - ry1),
                    'img': img_filename,  # ahora PNG
                    'mask': mask_filename,
                    'status': 'pending',
                    'cuts': [],
                    'brush_rle': [],
                    'source': source_name,
                    'foto_original': foto_path.name,
                })
                existing_ids.add(fid)
                added += 1

    save_index(frames)
    print(f"\nResultado:")
    print(f"  Frames agregados: {added}")
    print(f"  Frames fallidos: {failed}")
    print(f"  Total en dataset: {len(frames)}")


if __name__ == '__main__':
    main()
