"""Detector de VERIJA por frame: ventana + 'donde empiezan las patas' (silueta
por debajo de la panza del barril). Mediana sobre los 21 frames. Valida contra
los labels guardados (verija_frac_manual, referencia = barril).
"""
import json
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

PROJ = Path(__file__).parent
DATASETS = ['output_modelos3d_live_14mayo', 'output_modelos3d_live_20mayo']
W0, W1 = 0.62, 0.82          # ventana desde el FRENTE del barril
LEG_THR = 0.12               # pata si silueta baja > LEG_THR*alto_barril bajo la panza
RUN = 3                      # columnas sostenidas para confirmar pata

print("[init] modelos...")
coco = YOLO(str(PROJ / 'yolov8n.pt'))
barril = YOLO(str(PROJ / 'barril_seg.pt'))
sil = YOLO(str(PROJ / 'silueta_seg.pt'))


def seg(model, crop):
    r = model(crop, conf=0.25, verbose=False)
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        return None
    m = r[0].masks.data.cpu().numpy(); a = np.array([float(x.sum()) for x in m])
    if a.max() <= 0:
        return None
    s = np.max(m[a >= 0.05 * a.max()], axis=0)
    if s.shape != crop.shape[:2]:
        s = cv2.resize(s, (crop.shape[1], crop.shape[0]))
    return (s > 0.5).astype(np.uint8)


def bottom_top(mask, x):
    rr = np.where(mask[:, x] > 0)[0]
    if len(rr):
        return int(rr[0]), int(rr[-1])
    return None, None


def verija_frame(img, facing_right):
    H, W = img.shape[:2]
    r = coco(img, classes=[19], conf=0.2, verbose=False)
    if not r or len(r[0].boxes) == 0:
        return None
    b = r[0].boxes.xyxy.cpu().numpy(); s = r[0].boxes.conf.cpu().numpy()
    i = int(np.argmax(s)); x1, y1, x2, y2 = [int(v) for v in b[i]]
    pad = max(20, int(0.08 * max(x2 - x1, y2 - y1)))
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
    crop = img[cy1:cy2, cx1:cx2]
    mb = seg(barril, crop); ms = seg(sil, crop)
    if mb is None or ms is None:
        return None
    cols = np.where(mb.sum(0) > 0)[0]
    if len(cols) < 20:
        return None
    bxmin, bxmax = int(cols[0]), int(cols[-1]); Wb = bxmax - bxmin
    rows = np.where(mb.sum(1) > 0)[0]
    altoB = int(rows[-1] - rows[0]) if len(rows) else crop.shape[0]
    thr = LEG_THR * altoB
    # ventana en coords de barril, desde el frente
    if facing_right:   # frente = bxmax
        xa = bxmax - W1 * Wb; xb = bxmax - W0 * Wb
    else:              # frente = bxmin
        xa = bxmin + W0 * Wb; xb = bxmin + W1 * Wb
    lo, hi = int(min(xa, xb)), int(max(xa, xb))
    # leg_depth = cuanto baja la silueta bajo la panza del barril
    xs_win = list(range(lo, hi + 1)) if facing_right else list(range(lo, hi + 1))
    # recorrer de FRENTE hacia FONDO dentro de la ventana
    seq = list(range(hi, lo - 1, -1)) if facing_right else list(range(lo, hi + 1))
    run = 0; vx = None
    for x in seq:
        bt, bb = bottom_top(mb, x)          # panza = bb (barril bottom)
        st, sb = bottom_top(ms, x)          # silueta bottom
        if bb is None or sb is None:
            run = 0; continue
        leg = sb - bb                        # >0 si silueta baja mas que la panza
        if leg > thr:
            run += 1
            if run >= RUN:
                vx = x - (RUN - 1) * (1 if facing_right else -1)  # borde frontal de la pata
                break
        else:
            run = 0
    if vx is None:
        return None
    xrear = bxmin if facing_right else bxmax
    return abs(vx - xrear) / Wb             # frac desde el fondo del barril


items = []
for ds in DATASETS:
    base = PROJ / ds
    if not base.is_dir():
        continue
    for d in sorted(base.iterdir()):
        rj = next(d.glob('*_resumen.json'), None)
        if not rj:
            continue
        meta = json.loads(rj.read_text())
        if meta.get('verija_frac_manual') is None:
            continue
        origen = meta.get('carpeta_origen')
        if not origen:
            continue
        items.append((d.name, float(meta['verija_frac_manual']),
                      meta.get('barril_dir') != 'left', PROJ / origen))

print(f"[run] {len(items)} individuos con label\n")
errs = []
print(f"{'ind':14}{'label':>7}{'pred':>7}{'err_cm':>8}{'nfr':>5}")
for ind, lbl, fr, origen in items:
    fracs = []
    for fp in sorted(origen.glob('frame_*.jpg')):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        v = verija_frame(img, fr)
        if v is not None and 0.0 <= v <= 0.6:
            fracs.append(v)
    if len(fracs) < 5:
        print(f"{ind:14}{lbl:>7.2f}{'--':>7}{'--':>8}{len(fracs):>5}")
        continue
    pred = float(np.median(fracs))
    # err en cm: necesitamos L del barril (cm); usar largo del resumen si esta, si no aprox
    err_frac = abs(pred - lbl)
    errs.append((ind, lbl, pred, err_frac))
    print(f"{ind:14}{lbl:>7.2f}{pred:>7.2f}{err_frac*100:>7.0f}%{len(fracs):>5}")

if errs:
    ef = np.array([e[3] for e in errs])
    print(f"\n[resultado] n={len(errs)}  error frac: media={ef.mean()*100:.1f}%  "
          f"max={ef.max()*100:.1f}%   (constante 29% daba ~3.5% medio)")
