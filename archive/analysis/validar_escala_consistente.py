"""VALIDACIÓN (no destructivo): recomputa el volumen del barril con UNA escala
consistente por individuo (cm_per_px mediana del resumen) en vez de per-frame,
y compara el ajuste de peso contra el volumen actual. No toca modelos ni código
de producción (PLYs throwaway en /tmp).
"""
import json, sys, tempfile, os
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from scipy.spatial import Delaunay

PROJ = Path(__file__).parent
sys.path.insert(0, str(PROJ))
from generar_modelos3d_grandes import guardar_ply, volumen_malla_cerrada

DATASETS = ['output_modelos3d_live_14mayo', 'output_modelos3d_live_20mayo']
MIN_COV_X = 55.0
N = 60

print("[init] modelos...")
coco_model = YOLO(str(PROJ / 'yolov8n.pt'))
barril_model = YOLO(str(PROJ / 'barril_seg.pt'))


def _reparar_mascara(binmask, frac_alto=0.45):
    if binmask is None or binmask.size == 0:
        return set()
    bh, bw = binmask.shape
    cols_valid = np.where(binmask.sum(0) > 0)[0]
    if cols_valid.size < 2:
        return set()
    heights = np.zeros(bw, np.int32)
    for c in cols_valid:
        rs = np.where(binmask[:, c] > 0)[0]; heights[c] = int(rs[-1] - rs[0] + 1)
    umbral = max(2, int(frac_alto * int(np.median(heights[cols_valid]))))
    rep = set()
    for c in cols_valid:
        if heights[c] < umbral:
            binmask[:, c] = 0; rep.add(int(c))
    cols_valid = np.where(binmask.sum(0) > 0)[0]
    if cols_valid.size < 2:
        return rep
    c0, c1 = int(cols_valid[0]), int(cols_valid[-1])
    top = np.full(bw, -1, np.int32); bot = np.full(bw, -1, np.int32)
    for c in cols_valid:
        rs = np.where(binmask[:, c] > 0)[0]; top[c], bot[c] = int(rs[0]), int(rs[-1])
    c = c0 + 1
    while c < c1:
        if top[c] < 0:
            gs = ge = c
            while ge + 1 < c1 and top[ge + 1] < 0:
                ge += 1
            tk = min(int(top[gs - 1]), int(top[ge + 1])); bk = max(int(bot[gs - 1]), int(bot[ge + 1]))
            for ck in range(gs, ge + 1):
                if bk >= tk:
                    binmask[tk:bk + 1, ck] = 1; top[ck], bot[ck] = tk, bk; rep.add(int(ck))
            c = ge + 1
        else:
            c += 1
    return rep


def detectar(img):
    h, w = img.shape[:2]
    r = coco_model(img, classes=[19], conf=0.2, verbose=False)
    if not r or len(r[0].boxes) == 0:
        return None
    b = r[0].boxes.xyxy.cpu().numpy(); s = r[0].boxes.conf.cpu().numpy()
    i = int(np.argmax(s)); bx1, by1, bx2, by2 = [int(v) for v in b[i]]
    pad = max(20, int(0.08 * max(bx2 - bx1, by2 - by1)))
    cx1, cy1, cx2, cy2 = max(0, bx1 - pad), max(0, by1 - pad), min(w, bx2 + pad), min(h, by2 + pad)
    crop = img[cy1:cy2, cx1:cx2]
    rb = barril_model(crop, conf=0.25, verbose=False)
    if not rb or rb[0].masks is None or len(rb[0].masks.data) == 0:
        return None
    m = rb[0].masks.data.cpu().numpy(); a = np.array([float(x.sum()) for x in m])
    if a.max() <= 0:
        return None
    sil = np.max(m[a >= 0.05 * a.max()], axis=0)
    if sil.shape != (crop.shape[0], crop.shape[1]):
        sil = cv2.resize(sil, (crop.shape[1], crop.shape[0]))
    bin_full = np.zeros((h, w), np.uint8); bin_full[cy1:cy2, cx1:cx2] = (sil > 0.5).astype(np.uint8)
    return (bx1, by1, bx2, by2), bin_full, (cx2 - cx1)


def volumen_const(folder, s0):
    contornos = []
    for fp in sorted(Path(folder).glob('frame_*.jpg')):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        d = detectar(img)
        if d is None:
            continue
        (bx1, by1, bx2, by2), binmask, crop_w = d
        _reparar_mascara(binmask)
        cols = np.where(binmask.sum(0) > 0)[0]
        if cols.size < 2:
            continue
        if 100.0 * cols.size / max(1, crop_w) < MIN_COV_X:
            continue
        x_min, x_max = int(cols[0]), int(cols[-1])
        width_cm = (x_max - x_min + 1) * s0          # <<< escala CONSTANTE
        tops, bots = [], []
        for i in range(N):
            xp = int(x_min + (x_max - x_min) * i / (N - 1))
            rr = np.where(binmask[:, xp] > 0)[0]
            if rr.size == 0:
                tops.append(0.0); bots.append(0.0)
            else:
                tops.append((by2 - int(rr[0])) * s0); bots.append((by2 - int(rr[-1])) * s0)
        contornos.append((width_cm, tops, bots))
    if len(contornos) < 3:
        return None
    tops_env, bots_env = [], []
    for i in range(N):
        ts = [c[1][i] for c in contornos if c[1][i] > 0]
        bs = [c[2][i] for c in contornos if c[2][i] >= 0]
        if not ts or not bs:
            tops_env.append(0.0); bots_env.append(0.0)
        else:
            tops_env.append(max(ts)); bots_env.append(min(bs))
    width_env = max(c[0] for c in contornos)
    xs = np.linspace(0, width_env, N)
    contorno = np.vstack([np.column_stack([xs, tops_env]),
                          np.column_stack([xs[::-1], np.array(bots_env)[::-1]])])
    SCALE = 5; margin = 2.0
    xmn, ymn = contorno[:, 0].min() - margin, contorno[:, 1].min() - margin
    xmx, ymx = contorno[:, 0].max() + margin, contorno[:, 1].max() + margin
    Wp, Hp = int((xmx - xmn) * SCALE) + 1, int((ymx - ymn) * SCALE) + 1
    poly = ((contorno - [xmn, ymn]) * SCALE).astype(np.int32)
    mask_poly = np.zeros((Hp, Wp), np.uint8); cv2.fillPoly(mask_poly, [poly], 255)
    h_env = [t - b for t, b in zip(tops_env, bots_env)]
    step = max(2.5, max(h_env) / 12)
    pts_i = []
    for gx in np.arange(0, width_env, step):
        i = int(gx / width_env * (N - 1)); t, b = tops_env[i], bots_env[i]
        for gy in np.arange(b + step / 2, t, step):
            pts_i.append([gx, gy])
    all_px = np.vstack([contorno, np.array(pts_i)]) if pts_i else contorno
    all_px = np.unique(all_px, axis=0)
    if len(all_px) < 3:
        return None
    tri = Delaunay(all_px); tv = []
    for s in tri.simplices:
        cx, cy = all_px[s].mean(0)
        px, py = int((cx - xmn) * SCALE), int((cy - ymn) * SCALE)
        if 0 <= px < Wp and 0 <= py < Hp and mask_poly[py, px] > 0:
            tv.append(s)
    if not tv:
        return None
    tmp = os.path.join(tempfile.gettempdir(), 'vchk.ply')
    cols_c = np.array([[139, 90, 43]] * len(all_px), np.uint8)
    pts3d, tris3d = guardar_ply(tmp, all_px, np.array(tv), cols_c, simetrico=True, escala_info='chk')
    return volumen_malla_cerrada(pts3d, tris3d)


rows = []
for ds in DATASETS:
    base = PROJ / ds
    if not base.is_dir():
        continue
    for d in sorted(base.iterdir()):
        rj = next(d.glob('*_resumen.json'), None)
        if not rj:
            continue
        meta = json.loads(rj.read_text())
        s0 = meta.get('cm_per_px_median'); vcur = meta.get('vol_barril_litros')
        origen = meta.get('carpeta_origen')
        if not (s0 and vcur and origen):
            continue
        try:
            peso = float(d.name.split('_')[1])
        except Exception:
            continue
        vconst = volumen_const(PROJ / origen, s0)
        if vconst is None:
            print(f"  {d.name}: no se pudo recomputar"); continue
        rows.append((d.name, peso, vcur, vconst))
        print(f"  {d.name:14} peso={peso:>5.0f}  V_actual={vcur:>6.1f}L  V_const={vconst:>6.1f}L", flush=True)

if rows:
    peso = np.array([r[1] for r in rows])
    for label, idx in [('V_actual (per-frame)', 2), ('V_const (escala única)', 3)]:
        v = np.array([r[idx] for r in rows])
        k = peso / v; cv = 100 * k.std() / k.mean()
        pred = k.mean() * v; mape = 100 * np.mean(np.abs(pred - peso) / peso)
        ss = 1 - np.sum((peso - pred)**2) / np.sum((peso - peso.mean())**2)
        print(f"\n{label}: k={k.mean():.4f}  CV={cv:.1f}%  R2={ss:.2f}  MAPE={mape:.1f}%")
