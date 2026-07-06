"""Grid diagnóstico del LARGO por frame, para cada individuo de 14mayo/20mayo.

Por frame dibuja sobre el recorte de la vaca:
  · VERDE   = largo del bbox de la vaca (COCO)      → (bx2-bx1)*cm_per_px
  · NARANJA = largo de la máscara del barril (torso) → (xmax-xmin)*cm_per_px
con cm_per_px = altura_calc / (by2-by1) (la escala derivada de la altura).

Salida: grids_largo_<ds>/<individuo>_largo_grid.png

Uso: python diagnostico_largo.py [barril_seg.pt]
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

PROJ = Path(__file__).parent
_MODEL = sys.argv[1] if len(sys.argv) > 1 else 'barril_seg.pt'
barril_model = YOLO(str(PROJ / _MODEL))
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

GREEN = (80, 220, 80)
ORANGE = (0, 140, 240)


def _reparar_mascara(binmask, frac_alto=0.45):
    if binmask is None or binmask.size == 0:
        return
    bh, bw = binmask.shape
    cols_valid = np.where(binmask.sum(axis=0) > 0)[0]
    if cols_valid.size < 2:
        return
    heights = np.zeros(bw, dtype=np.int32)
    for _c in cols_valid:
        rs = np.where(binmask[:, _c] > 0)[0]
        heights[_c] = int(rs[-1] - rs[0] + 1)
    umbral = max(2, int(frac_alto * int(np.median(heights[cols_valid]))))
    for _c in cols_valid:
        if heights[_c] < umbral:
            binmask[:, _c] = 0
    cols_valid = np.where(binmask.sum(axis=0) > 0)[0]
    if cols_valid.size < 2:
        return
    c0, c1 = int(cols_valid[0]), int(cols_valid[-1])
    top = np.full(bw, -1, dtype=np.int32); bot = np.full(bw, -1, dtype=np.int32)
    for _c in cols_valid:
        rs = np.where(binmask[:, _c] > 0)[0]
        top[_c], bot[_c] = int(rs[0]), int(rs[-1])
    c = c0 + 1
    while c < c1:
        if top[c] < 0:
            gs, ge = c, c
            while ge + 1 < c1 and top[ge + 1] < 0:
                ge += 1
            tk = min(int(top[gs - 1]), int(top[ge + 1]))
            bk = max(int(bot[gs - 1]), int(bot[ge + 1]))
            for ck in range(gs, ge + 1):
                if bk >= tk:
                    binmask[tk:bk + 1, ck] = 1; top[ck], bot[ck] = tk, bk
            c = ge + 1
        else:
            c += 1


def procesar_frame(img, altura_cm):
    H, W = img.shape[:2]
    rc = coco_model(img, classes=[19], conf=0.2, verbose=False)
    if not rc or len(rc[0].boxes) == 0:
        return None
    boxes = rc[0].boxes.xyxy.cpu().numpy()
    scores = rc[0].boxes.conf.cpu().numpy()
    bx1, by1, bx2, by2 = [int(v) for v in boxes[int(np.argmax(scores))]]
    pad = max(20, int(0.08 * max(bx2 - bx1, by2 - by1)))
    cx1, cy1 = max(0, bx1 - pad), max(0, by1 - pad)
    cx2, cy2 = min(W, bx2 + pad), min(H, by2 + pad)
    crop = img[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop.shape[:2]
    if ch < 30 or cw < 30:
        return None
    cm_per_px = altura_cm / max(1, (by2 - by1))
    bbox_len = (bx2 - bx1) * cm_per_px

    # Barril
    mask = None; barril_len = None; bxmin = bxmax = None
    rb = barril_model(crop, conf=0.25, verbose=False)
    if rb and rb[0].masks is not None and len(rb[0].masks.data):
        masks = rb[0].masks.data.cpu().numpy()
        areas = np.array([float(m.sum()) for m in masks])
        keep = areas >= 0.05 * areas.max() if areas.max() > 0 else np.ones(len(masks), bool)
        union = np.max(masks[keep], axis=0)
        if union.shape != (ch, cw):
            union = cv2.resize(union, (cw, ch))
        mask = (union > 0.5).astype(np.uint8)
        _reparar_mascara(mask)
        cols_any = np.where(mask.sum(axis=0) > 0)[0]
        if cols_any.size >= 2:
            bxmin, bxmax = int(cols_any[0]), int(cols_any[-1])
            barril_len = (bxmax - bxmin + 1) * cm_per_px

    # Dibujar overlay barril + líneas de largo
    if mask is not None:
        cl = np.zeros_like(crop); cl[mask > 0] = ORANGE
        crop = cv2.addWeighted(crop, 1.0, cl, 0.35, 0)
        cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, cont, -1, ORANGE, 1)

    # bbox de la vaca (en coords del crop)
    bxc1, byc1, bxc2, byc2 = bx1 - cx1, by1 - cy1, bx2 - cx1, by2 - cy1
    cv2.rectangle(crop, (bxc1, byc1), (bxc2, byc2), GREEN, 1)
    # línea largo bbox (verde) cerca del top
    yb = max(8, byc1 + 10)
    cv2.line(crop, (bxc1, yb), (bxc2, yb), GREEN, 2)
    cv2.line(crop, (bxc1, yb - 5), (bxc1, yb + 5), GREEN, 2)
    cv2.line(crop, (bxc2, yb - 5), (bxc2, yb + 5), GREEN, 2)
    # línea largo barril (naranja) en el centro vertical del barril
    if barril_len is not None:
        rows_any = np.where(mask.sum(axis=1) > 0)[0]
        yo = int((rows_any[0] + rows_any[-1]) / 2) if rows_any.size else byc2
        cv2.line(crop, (bxmin, yo), (bxmax, yo), ORANGE, 2)
        cv2.line(crop, (bxmin, yo - 5), (bxmin, yo + 5), ORANGE, 2)
        cv2.line(crop, (bxmax, yo - 5), (bxmax, yo + 5), ORANGE, 2)
    return dict(crop=crop, cm_per_px=cm_per_px, bbox_len=bbox_len, barril_len=barril_len)


def make_tile(crop, name, cm_per_px, bbox_len, barril_len, tile_w=340):
    h, w = crop.shape[:2]
    if w == 0:
        return None
    img = cv2.resize(crop, (tile_w, int(h * tile_w / w)))
    bar = np.zeros((46, tile_w, 3), dtype=np.uint8)
    cv2.putText(bar, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    bl = f"bbox={bbox_len:.0f}cm" if bbox_len is not None else "bbox=?"
    rl = f"barril={barril_len:.0f}cm" if barril_len is not None else "barril=NA"
    cv2.putText(bar, bl, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.46, GREEN, 1, cv2.LINE_AA)
    cv2.putText(bar, rl, (160, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.46, ORANGE, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def grid_individuo(folder, altura_cm, out_path, cols=7):
    tiles = []
    for fp in sorted(folder.glob('frame_*.jpg')):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        res = procesar_frame(img, altura_cm)
        if res is None:
            continue
        t = make_tile(res['crop'], fp.name.replace('.jpg', ''), res['cm_per_px'],
                      res['bbox_len'], res['barril_len'])
        if t is not None:
            tiles.append(t)
    if not tiles:
        return False
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    rows_img = []
    for r in range(rows):
        row = tiles[r * cols:(r + 1) * cols]
        row = [cv2.resize(t, (tw, th)) for t in row]
        while len(row) < cols:
            row.append(np.zeros((th, tw, 3), dtype=np.uint8))
        rows_img.append(np.hstack(row))
    cv2.imwrite(str(out_path), np.vstack(rows_img))
    return True


for ds in ('14mayo', '20mayo'):
    out_dir = PROJ / f'grids_largo_{ds}'
    out_dir.mkdir(exist_ok=True)
    base = PROJ / 'checkpoints' / ds
    for name, alt in ALTURAS[ds].items():
        ok = grid_individuo(base / name, alt, out_dir / f'{name}_largo_grid.png')
        print(f"[{ds}] {name}: {'OK' if ok else 'FALLO'}")
print("DONE")
