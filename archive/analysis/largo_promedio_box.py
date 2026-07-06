"""Largo PROMEDIO del box (barril) por individuo, sobre los 21 frames.

Replica la detección + reparación de máscara + escala por-frame de
procesar_21_frames.py (cm_per_px_i = altura_calc / bbox_h_px), pero en vez
del envelope (max) reporta el PROMEDIO de width_cm entre los frames válidos.

Uso: python largo_promedio_box.py
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

PROJ = Path(__file__).parent
_MODEL = sys.argv[1] if len(sys.argv) > 1 else 'barril_seg.pt'
print(f"[init] barril model = {_MODEL}")
barril_model = YOLO(str(PROJ / _MODEL))
coco_model = YOLO(str(PROJ / 'yolov8n.pt'))

# altura_calc (cm) por individuo, de los batch scripts
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


def _reparar_mascara(binmask, frac_alto=0.45):
    if binmask is None or binmask.size == 0:
        return
    bh, bw = binmask.shape
    cols_sum = binmask.sum(axis=0)
    cols_valid = np.where(cols_sum > 0)[0]
    if cols_valid.size < 2:
        return
    heights = np.zeros(bw, dtype=np.int32)
    for _c in cols_valid:
        rs = np.where(binmask[:, _c] > 0)[0]
        heights[_c] = int(rs[-1] - rs[0] + 1)
    h_med = int(np.median(heights[cols_valid]))
    umbral = max(2, int(frac_alto * h_med))
    for _c in cols_valid:
        if heights[_c] < umbral:
            binmask[:, _c] = 0
    cols_sum = binmask.sum(axis=0)
    cols_valid = np.where(cols_sum > 0)[0]
    if cols_valid.size < 2:
        return
    c0, c1 = int(cols_valid[0]), int(cols_valid[-1])
    top = np.full(bw, -1, dtype=np.int32)
    bot = np.full(bw, -1, dtype=np.int32)
    for _c in cols_valid:
        rs = np.where(binmask[:, _c] > 0)[0]
        top[_c], bot[_c] = int(rs[0]), int(rs[-1])
    c = c0 + 1
    while c < c1:
        if top[c] < 0:
            gs, ge = c, c
            while ge + 1 < c1 and top[ge + 1] < 0:
                ge += 1
            tL, bL = int(top[gs - 1]), int(bot[gs - 1])
            tR, bR = int(top[ge + 1]), int(bot[ge + 1])
            tk, bk = min(tL, tR), max(bL, bR)
            for ck in range(gs, ge + 1):
                if bk >= tk:
                    binmask[tk:bk + 1, ck] = 1
                    top[ck], bot[ck] = tk, bk
            c = ge + 1
        else:
            c += 1


def _detectar_y_segmentar(img):
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


def largo_promedio(folder, altura_cm):
    widths = []
    for fp in sorted(folder.glob('frame_*.jpg')):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        bbox, binmask = _detectar_y_segmentar(img)
        if bbox is None or binmask is None:
            continue
        bx1, by1, bx2, by2 = bbox
        bbox_h_px = max(1, by2 - by1)
        cm_per_px = altura_cm / bbox_h_px
        _reparar_mascara(binmask)
        cols_any = np.where(binmask.sum(axis=0) > 0)[0]
        if cols_any.size < 2:
            continue
        width_cm = (int(cols_any[-1]) - int(cols_any[0]) + 1) * cm_per_px
        widths.append(width_cm)
    if not widths:
        return None
    w = np.array(widths)
    return {'prom': w.mean(), 'std': w.std(), 'min': w.min(), 'max': w.max(), 'n': len(w)}


for ds in ('14mayo', '20mayo'):
    print(f"\n============ {ds.upper()} — largo promedio del box (21 frames) ============")
    print(f"{'individuo':<16}{'altura(cm)':>11}{'largo_prom':>12}{'±std':>8}{'min':>8}{'max':>8}{'n':>5}")
    base = PROJ / 'checkpoints' / ds
    for name, alt in ALTURAS[ds].items():
        folder = base / name
        if not folder.is_dir():
            print(f"{name:<16}  (carpeta no encontrada)")
            continue
        r = largo_promedio(folder, alt)
        if r is None:
            print(f"{name:<16}{alt:>11}   (sin barril en ningún frame)")
            continue
        print(f"{name:<16}{alt:>11}{r['prom']:>12.1f}{r['std']:>8.1f}"
              f"{r['min']:>8.1f}{r['max']:>8.1f}{r['n']:>5}")
print("\nDONE")
