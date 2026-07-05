"""Largo PROMEDIO del bbox de la vaca (COCO) por individuo, sobre los 21 frames.

largo_bbox_cm = (bx2-bx1) * cm_per_px,  cm_per_px = altura_calc / (by2-by1)
Solo usa el modelo COCO → es independiente de v6/v7 (un solo largo).

Uso: python largo_bbox_promedio.py
"""
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

PROJ = Path(__file__).parent
coco_model = YOLO(str(PROJ / 'yolov8n.pt'))

ALTURAS = {
    '14mayo': {
        '100_137.5': 85.6, '101_156': 105.3, '102_150': 108.2, '102_168.5': 108.2,
        '102_171.5': 90.2, '102_177': 96.1, '103_169': 105.4, '103.5_173.5': 102.1,
        '105_182.5': 100.9, '110_221': 105.1, '110_228': 112.0, '112_203': 102.6,
        '113_214': 109.1, '114_166': 108.9,
    },
    '20mayo': {
        '118_462': 115.1, '118_510': 125.5, '124_478': 119.6, '124_498': 124.4,
        '126_463': 129.5, '127_435': 120.7, '129_472': 123.7, '129_477': 129.0,
        '129_487': 124.4, '129_504': 126.6, '133_544': 122.8, '134_532': 128.9,
        '134_556': 134.0, '135_630': 135.0,
    },
}


def largo_bbox_prom(folder, altura_cm):
    largos = []
    for fp in sorted(folder.glob('frame_*.jpg')):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        r = coco_model(img, classes=[19], conf=0.2, verbose=False)
        if not r or len(r[0].boxes) == 0:
            continue
        boxes = r[0].boxes.xyxy.cpu().numpy()
        scores = r[0].boxes.conf.cpu().numpy()
        bx1, by1, bx2, by2 = boxes[int(np.argmax(scores))]
        h_px = max(1.0, by2 - by1)
        w_px = max(0.0, bx2 - bx1)
        cm_per_px = altura_cm / h_px
        largos.append(w_px * cm_per_px)
    if not largos:
        return None
    a = np.array(largos)
    return {'prom': a.mean(), 'std': a.std(), 'min': a.min(), 'max': a.max(), 'n': len(a)}


for ds in ('14mayo', '20mayo'):
    print(f"\n==== {ds.upper()} — largo PROMEDIO del bbox vaca (21 frames) ====")
    print(f"{'individuo':<16}{'altura':>8}{'largo_prom':>12}{'±std':>8}{'min':>8}{'max':>8}{'n':>5}")
    base = PROJ / 'checkpoints' / ds
    for name, alt in ALTURAS[ds].items():
        r = largo_bbox_prom(base / name, alt)
        if r is None:
            print(f"{name:<16}{alt:>8}  (sin deteccion)")
            continue
        print(f"{name:<16}{alt:>8}{r['prom']:>12.1f}{r['std']:>8.1f}"
              f"{r['min']:>8.1f}{r['max']:>8.1f}{r['n']:>5}")
print("\nDONE")
