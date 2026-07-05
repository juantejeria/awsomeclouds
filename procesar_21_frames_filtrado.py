"""Igual que procesar_21_frames.py, pero filtra frames cuya cobertura X de la
máscara del barril dentro del cow_crop sea menor a `--min-cov-x` (default 55%).

Cobertura X = (columnas con mask>0) / (ancho del cow_crop).

Salida: output_modelos3d_live_<tag>/<cow_name>/  (default tag = "filtrado")

Uso:
    python procesar_21_frames_filtrado.py <carpeta_frames> <altura_cm> <cow_name> \
        [--min-cov-x 55] [--out-tag filtrado]
"""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO


_ap = argparse.ArgumentParser()
_ap.add_argument('folder')
_ap.add_argument('altura_cm', type=float)
_ap.add_argument('cow_name')
_ap.add_argument('--min-cov-x', type=float, default=55.0,
                 help='Cobertura X mínima en %% para incluir el frame en el consenso.')
_ap.add_argument('--out-tag', default='filtrado',
                 help='Subdirectorio base: output_modelos3d_live_<tag>/')
_ap.add_argument('--barril-model', default='barril_seg.pt',
                 help='.pt del modelo de segmentación de barril (relativo al proyecto o ruta absoluta)')
_args = _ap.parse_args()
folder = Path(_args.folder)
altura_cm = _args.altura_cm
cow_name = _args.cow_name
min_cov_x = _args.min_cov_x
out_tag = _args.out_tag

if not folder.is_dir():
    print(f"[error] no existe: {folder}"); sys.exit(1)

proj_dir = Path(__file__).parent
_barril_arg = Path(_args.barril_model)
barril_path = _barril_arg if _barril_arg.is_absolute() else (proj_dir / _barril_arg)
if not barril_path.exists():
    print(f"[error] modelo barril no existe: {barril_path}"); sys.exit(1)
print(f"[init] cargando modelos... (barril={barril_path.name})")
barril_model = YOLO(str(barril_path))
coco_model = YOLO(str(proj_dir / 'yolov8n.pt'))


def _reparar_mascara(binmask, frac_alto=0.45):
    """Idéntico a app.py:_reparar_mascara_oclusion (envelope local)."""
    if binmask is None or binmask.size == 0:
        return set()
    bh, bw = binmask.shape
    cols_sum = binmask.sum(axis=0)
    cols_valid = np.where(cols_sum > 0)[0]
    if cols_valid.size < 2:
        return set()
    heights = np.zeros(bw, dtype=np.int32)
    for _c in cols_valid:
        rs = np.where(binmask[:, _c] > 0)[0]
        heights[_c] = int(rs[-1] - rs[0] + 1)
    h_med = int(np.median(heights[cols_valid]))
    umbral = max(2, int(frac_alto * h_med))
    cols_rep = set()
    for _c in cols_valid:
        if heights[_c] < umbral:
            binmask[:, _c] = 0
            cols_rep.add(int(_c))
    cols_sum = binmask.sum(axis=0)
    cols_valid = np.where(cols_sum > 0)[0]
    if cols_valid.size < 2:
        return cols_rep
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
                    cols_rep.add(int(ck))
            c = ge + 1
        else:
            c += 1
    return cols_rep


def _detectar_y_segmentar(img):
    """Devuelve (bbox_yxyx, barril_binmask, crop_xy) o (None, None, None) si falla.
    crop_xy = (cx1, cy1, cx2, cy2) del cow_crop usado por barril_seg."""
    h_orig, w_orig = img.shape[:2]
    r_cow = coco_model(img, classes=[19], conf=0.2, verbose=False)
    if not r_cow or len(r_cow[0].boxes) == 0:
        return None, None, None
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
        return (bx1, by1, bx2, by2), None, (cx1, cy1, cx2, cy2)
    masks = r_bar[0].masks.data.cpu().numpy()
    areas = np.array([float(np.sum(m)) for m in masks])
    keep = areas >= 0.05 * areas.max() if areas.max() > 0 else np.ones(len(masks), bool)
    sil = np.max(masks[keep], axis=0)
    if sil.shape != (crop.shape[0], crop.shape[1]):
        sil = cv2.resize(sil, (crop.shape[1], crop.shape[0]))
    binmask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    binmask[cy1:cy2, cx1:cx2] = (sil > 0.5).astype(np.uint8)
    return (bx1, by1, bx2, by2), binmask, (cx1, cy1, cx2, cy2)


# Procesar cada frame con su PROPIO cm_per_px (la vaca cambia de tamaño en
# pixels por perspectiva al desplazarse → cada frame tiene su escala).
# cm_per_px[i] = altura_real / bbox_h_px[i]  (asumiendo que la vaca mide
# siempre lo mismo físicamente, 92.5 cm en este caso).
frame_files = sorted([p for p in folder.glob('frame_*.jpg')])
print(f"[init] {len(frame_files)} frames\n")
print(f"{'frame':<28} {'bbox_h(px)':>10} {'cm/px':>10} {'width(cm)':>10} {'altmax(cm)':>11} {'cols_rep':>9}")
print("-" * 80)

N = 60
contornos = []
for fp in frame_files:
    img = cv2.imread(str(fp))
    if img is None: continue
    bbox, binmask, crop_xy = _detectar_y_segmentar(img)
    if bbox is None:
        print(f"  {fp.name:<28}  SIN VACA"); continue
    bx1, by1, bx2, by2 = bbox
    bbox_h_px = max(1, by2 - by1)
    cm_per_px_i = altura_cm / bbox_h_px  # PER-FRAME
    if binmask is None:
        print(f"  {fp.name:<28} {bbox_h_px:>10} {cm_per_px_i:>10.5f}   SIN BARRIL"); continue
    cols_rep = _reparar_mascara(binmask)
    cols_any = np.where(binmask.sum(axis=0) > 0)[0]
    if cols_any.size < 2:
        print(f"  {fp.name:<28} {bbox_h_px:>10} {cm_per_px_i:>10.5f}   VACÍO"); continue
    x_min, x_max = int(cols_any[0]), int(cols_any[-1])
    width_cm = (x_max - x_min + 1) * cm_per_px_i
    # Cobertura X relativa al cow_crop (mismo criterio que diagnostico_21frames_barril.py)
    cx1, cy1, cx2, cy2 = crop_xy
    crop_w = max(1, cx2 - cx1)
    cov_x_pct = 100.0 * cols_any.size / crop_w
    if cov_x_pct < min_cov_x:
        print(f"  {fp.name:<28} {bbox_h_px:>10} {cm_per_px_i:>10.5f}   FILTRADO (covX={cov_x_pct:.0f}% < {min_cov_x:.0f}%)")
        continue
    tops, bots, hs = [], [], []
    for i in range(N):
        x_px = int(x_min + (x_max - x_min) * i / (N - 1))
        rows = np.where(binmask[:, x_px] > 0)[0]
        if rows.size == 0:
            tops.append(0.0); bots.append(0.0); hs.append(0.0)
        else:
            t_above = (by2 - int(rows[0])) * cm_per_px_i
            b_above = (by2 - int(rows[-1])) * cm_per_px_i
            tops.append(round(t_above, 2))
            bots.append(round(b_above, 2))
            hs.append(round(t_above - b_above, 2))
    contornos.append({
        'frame': fp.name, 'width_cm': width_cm, 'cols_rep': len(cols_rep),
        'cm_per_px': cm_per_px_i, 'bbox_h_px': bbox_h_px,
        'tops_cm': tops, 'bottoms_cm': bots, 'heights_cm': hs,
    })
    print(f"  {fp.name:<28} {bbox_h_px:>10} {cm_per_px_i:>10.5f} {width_cm:>10.2f} {max(hs):>11.2f} {len(cols_rep):>9}")

if not contornos:
    print("[error] no hay contornos válidos"); sys.exit(1)
print(f"\n[consenso] {len(contornos)} frames procesados")


# 3. Consenso ENVELOPE
tops_env, bots_env, h_env = [], [], []
for i in range(N):
    ts = [c['tops_cm'][i] for c in contornos if c['tops_cm'][i] > 0]
    bs = [c['bottoms_cm'][i] for c in contornos if c['bottoms_cm'][i] >= 0]
    if not ts or not bs:
        tops_env.append(0.0); bots_env.append(0.0); h_env.append(0.0); continue
    tops_env.append(max(ts))
    bots_env.append(min(bs))
    h_env.append(tops_env[-1] - bots_env[-1])
widths = [c['width_cm'] for c in contornos]
width_env = max(widths)


print(f"\n  width env:    {width_env:.1f} cm")
print(f"  alto max:     {max(h_env):.1f} cm")


# 4. Construir contorno cerrado del barril (cm) y triangular
xs = np.linspace(0, width_env, N)
contorno_top = np.column_stack([xs, tops_env])
contorno_bot = np.column_stack([xs[::-1], np.array(bots_env)[::-1]])
contorno_cm = np.vstack([contorno_top, contorno_bot])

# Rasterizar el polígono → máscara para filtrar triángulos por punto-en-polígono
SCALE = 5
margin = 2.0
xmin = contorno_cm[:, 0].min() - margin
ymin = contorno_cm[:, 1].min() - margin
xmax = contorno_cm[:, 0].max() + margin
ymax = contorno_cm[:, 1].max() + margin
W = int((xmax - xmin) * SCALE) + 1
H = int((ymax - ymin) * SCALE) + 1
poly_px = ((contorno_cm - [xmin, ymin]) * SCALE).astype(np.int32)
mask_poly = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(mask_poly, [poly_px], 255)

# Grid interior cada step cm
step = max(2.5, max(h_env) / 12)
pts_i = []
for gx in np.arange(0, width_env, step):
    i = int(gx / width_env * (N - 1))
    t, b = tops_env[i], bots_env[i]
    for gy in np.arange(b + step/2, t, step):
        pts_i.append([gx, gy])
pts_i = np.array(pts_i) if pts_i else np.empty((0, 2))
all_px = np.vstack([contorno_cm, pts_i]) if len(pts_i) else contorno_cm
all_px = np.unique(all_px, axis=0)

from scipy.spatial import Delaunay
tri = Delaunay(all_px)
tris_validos = []
for s in tri.simplices:
    cx, cy = all_px[s].mean(axis=0)
    px, py = int((cx - xmin) * SCALE), int((cy - ymin) * SCALE)
    if 0 <= px < W and 0 <= py < H and mask_poly[py, px] > 0:
        tris_validos.append(s)
tris_arr = np.array(tris_validos)
print(f"[malla] {len(all_px)} verts, {len(tris_arr)} triángulos")


# 5. Generar PLYs
sys.path.insert(0, str(proj_dir))
from generar_modelos3d_grandes import guardar_ply, volumen_malla_cerrada

out_dir = proj_dir / f'output_modelos3d_live_{out_tag}' / cow_name
out_dir.mkdir(parents=True, exist_ok=True)
colores = np.array([[139, 90, 43]] * len(all_px), dtype=np.uint8)

ply_lat = out_dir / f'{cow_name}_lateral.ply'
guardar_ply(str(ply_lat), all_px, tris_arr, colores, simetrico=False,
            escala_info=f'Consenso E envelope | n={len(contornos)} frames | alto={altura_cm:.1f}cm')
print(f"[ply] lateral → {ply_lat}")

# 3D: malla cerrada (silueta espejada). Su volumen encerrado es el volumen
# reportado — única fuente de volumen (sin rebanadas/cilindros).
ply_3d = out_dir / f'{cow_name}_3d.ply'
pts_3d, tris_3d = guardar_ply(str(ply_3d), all_px, tris_arr, colores, simetrico=True,
                              escala_info=f'Consenso E envelope | alto={altura_cm:.1f}cm')
vol_barril = volumen_malla_cerrada(pts_3d, tris_3d)
print(f"[ply] 3d      → {ply_3d}")
print(f"[volumen] malla cerrada _3d.ply: {vol_barril} L")


# 6. Resumen
resumen = {
    'individuo': cow_name,
    'altura_real_cm': altura_cm,
    'metodo': f'envelope_21_frames_filtrado(min_covX={min_cov_x:.0f}%)',
    'frames_usados': len(contornos),
    'min_cov_x_pct': min_cov_x,
    'frames_descartados': len(frame_files) - len(contornos),
    'carpeta_origen': str(folder),
    'cm_per_px_min': round(min(c['cm_per_px'] for c in contornos), 5),
    'cm_per_px_max': round(max(c['cm_per_px'] for c in contornos), 5),
    'cm_per_px_median': round(float(np.median([c['cm_per_px'] for c in contornos])), 5),
    'vol_barril_litros': vol_barril,
    'width_consenso_cm': round(width_env, 1),
    'alto_max_consenso_cm': round(max(h_env), 1),
}
with open(out_dir / f'{cow_name}_resumen.json', 'w') as f:
    json.dump(resumen, f, indent=2)
print(f"[resumen] → {out_dir}/{cow_name}_resumen.json")
print(f"\nDone: {cow_name}")
