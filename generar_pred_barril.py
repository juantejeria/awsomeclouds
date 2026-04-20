"""
Genera predicciones de silueta (_pred.png) para los frames del barril training
que no las tengan. Usa silueta_seg.pt como guía visual en el editor.
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

    # Solo frames pending que no tengan _pred.png
    pending = []
    for fr in frames:
        pred_path = DATA_DIR / f"{fr['id']}_pred.png"
        if not pred_path.exists():
            pending.append(fr)

    if not pending:
        print("Todos los frames ya tienen predicción.")
        return

    print(f"{len(pending)} frames sin predicción. Generando...")

    # Cargar modelo de silueta
    silueta_path = PROJECT / 'silueta_seg.pt'
    if not silueta_path.exists():
        print("ERROR: silueta_seg.pt no encontrado")
        sys.exit(1)

    model = YOLO(str(silueta_path))
    generated = 0

    for fr in pending:
        img_path = DATA_DIR / fr['img']
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        results = model(img, conf=0.25, verbose=False)
        if results and len(results[0].boxes) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            # Tomar la máscara más grande
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
                pred = (best_mask * 255).astype(np.uint8)
                cv2.imwrite(str(DATA_DIR / f"{fr['id']}_pred.png"), pred)
                generated += 1

    print(f"Predicciones generadas: {generated}/{len(pending)}")


if __name__ == '__main__':
    main()
