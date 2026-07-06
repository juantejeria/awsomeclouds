"""
Reemplaza las _mask.png de los frames pending del barril training
con predicciones de silueta_seg.pt (nuestro modelo entrenado).
Así el editor muestra la silueta buena en verde como base para recortar el barril.
"""
import cv2
import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / 'output_modelos3d_grandes' / '_barril_training'
INDEX_FILE = DATA_DIR / 'frames_index.json'


def main():
    with open(INDEX_FILE) as f:
        frames = json.load(f)

    # Solo frames pending (no tocar los ya validados)
    pending = [fr for fr in frames if fr.get('status') == 'pending']
    print(f"{len(pending)} frames pending para actualizar mask")

    if not pending:
        print("No hay frames pending.")
        return

    silueta_path = PROJECT / 'silueta_seg.pt'
    if not silueta_path.exists():
        print("ERROR: silueta_seg.pt no encontrado")
        sys.exit(1)

    model = YOLO(str(silueta_path))
    updated = 0
    failed = 0

    for fr in pending:
        img_path = DATA_DIR / fr['img']
        mask_path = DATA_DIR / fr['mask']
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        results = model(img, conf=0.25, verbose=False)
        if results and len(results[0].boxes) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            best_mask = None
            best_area = 0
            for m in masks:
                m_resized = cv2.resize(m, (w, h))
                m_bin = (m_resized > 0.5).astype(np.uint8)
                area = np.sum(m_bin)
                if area > best_area:
                    best_area = area
                    best_mask = m_bin

            if best_mask is not None:
                mask_out = (best_mask * 255).astype(np.uint8)
                # Limpieza morfológica
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_OPEN, kernel, iterations=1)
                cv2.imwrite(str(mask_path), mask_out)
                updated += 1
            else:
                failed += 1
        else:
            failed += 1

    print(f"Masks actualizadas: {updated}")
    print(f"Fallidas: {failed}")


if __name__ == '__main__':
    main()
