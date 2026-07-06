"""Regenera las mallas 3D usando A_mediana (no envelope).
Para cada vN en output_modelos3d_live/:
  - Backup de los PLYs originales (envelope) → renombrados a *_envelope_orig.ply
    (el viewer los ignora por el filtro '_orig.ply')
  - Recalcula tops/bottoms/widths como MEDIANA por columna entre los 21 frames
  - Genera PLYs nuevos (lateral, 3d) sobre la silueta consenso mediana
  - Actualiza el resumen.json con vol_barril_litros = volumen encerrado del _3d.ply

Uso:
    python regenerar_mallas_mediana.py [--solo v1,v2,...] [--carpeta-frames checkpoints/22abril]
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO


PROJ = Path(__file__).parent
N = 60


def _reparar_mascara(binmask, frac_alto=0.45):
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
            for cc in range(gs, ge + 1):
                binmask[tk:bk + 1, cc] = 1
                cols_rep.add(int(cc))
                top[cc], bot[cc] = tk, bk
            c = ge + 1
        c += 1
    return cols_rep


def _detectar_y_segmentar(img, coco_model, barril_model):
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


def procesar(cow_id, altura_cm, frames_dir, coco_model, barril_model, out_dir, peso_real=None):
    contornos = []
    frame_files = sorted(frames_dir.glob('frame_*.jpg'))
    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        bbox, binmask = _detectar_y_segmentar(img, coco_model, barril_model)
        if bbox is None or binmask is None:
            continue
        bx1, by1, bx2, by2 = bbox
        bbox_h_px = max(1, by2 - by1)
        cm_per_px_i = altura_cm / bbox_h_px
        _reparar_mascara(binmask)
        cols_any = np.where(binmask.sum(axis=0) > 0)[0]
        if cols_any.size < 2:
            continue
        x_min, x_max = int(cols_any[0]), int(cols_any[-1])
        width_cm = (x_max - x_min + 1) * cm_per_px_i
        tops, bots, hs = [], [], []
        for i in range(N):
            x_px = int(x_min + (x_max - x_min) * i / (N - 1))
            rows = np.where(binmask[:, x_px] > 0)[0]
            if rows.size == 0:
                tops.append(0.0); bots.append(0.0); hs.append(0.0)
            else:
                t = (by2 - int(rows[0])) * cm_per_px_i
                b = (by2 - int(rows[-1])) * cm_per_px_i
                tops.append(t); bots.append(b); hs.append(t - b)
        contornos.append({
            'frame': fp.name, 'tops': tops, 'bots': bots, 'hs': hs,
            'width': width_cm, 'cm_per_px': cm_per_px_i,
        })
    if not contornos:
        print(f"  [{cow_id}] sin contornos válidos"); return None

    # Consenso MEDIANA por columna
    tops_med = []
    bots_med = []
    hs_med = []
    for i in range(N):
        ts = sorted([c['tops'][i] for c in contornos if c['tops'][i] > 0])
        bs = sorted([c['bots'][i] for c in contornos if c['bots'][i] >= 0])
        hsv = sorted([c['hs'][i] for c in contornos if c['hs'][i] > 0])
        tops_med.append(ts[len(ts) // 2] if ts else 0.0)
        bots_med.append(bs[len(bs) // 2] if bs else 0.0)
        hs_med.append(hsv[len(hsv) // 2] if hsv else 0.0)
    widths = sorted([c['width'] for c in contornos])
    width_med = widths[len(widths) // 2]

    # Construir contorno cerrado y triangular
    xs = np.linspace(0, width_med, N)
    contorno_top = np.column_stack([xs, tops_med])
    contorno_bot = np.column_stack([xs[::-1], np.array(bots_med)[::-1]])
    contorno_cm = np.vstack([contorno_top, contorno_bot])

    SCALE = 5
    margin = 2.0
    xmin = contorno_cm[:, 0].min() - margin
    ymin = contorno_cm[:, 1].min() - margin
    xmax = contorno_cm[:, 0].max() + margin
    ymax = contorno_cm[:, 1].max() + margin
    Wpx = int((xmax - xmin) * SCALE) + 1
    Hpx = int((ymax - ymin) * SCALE) + 1
    poly_px = ((contorno_cm - [xmin, ymin]) * SCALE).astype(np.int32)
    mask_poly = np.zeros((Hpx, Wpx), dtype=np.uint8)
    cv2.fillPoly(mask_poly, [poly_px], 255)

    step = max(2.5, max(hs_med) / 12) if max(hs_med) > 0 else 5.0
    pts_i = []
    for gx in np.arange(0, width_med, step):
        i = int(gx / max(width_med, 1) * (N - 1))
        t, b = tops_med[i], bots_med[i]
        for gy in np.arange(b + step / 2, t, step):
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
        if 0 <= px < Wpx and 0 <= py < Hpx and mask_poly[py, px] > 0:
            tris_validos.append(s)
    tris_arr = np.array(tris_validos)

    # Generar PLYs
    sys.path.insert(0, str(PROJ))
    from generar_modelos3d_grandes import guardar_ply, volumen_malla_cerrada

    out_dir.mkdir(parents=True, exist_ok=True)

    # Backup envelope originals (idempotente)
    for f in out_dir.iterdir():
        if not f.is_file() or not f.suffix == '.ply':
            continue
        if '_orig.ply' in f.name:
            continue  # ya respaldado
        # Solo PLYs principales: lateral, 3d, volumen
        stem = f.stem
        if stem.endswith('_lateral') or stem.endswith('_3d') or stem.endswith('_volumen'):
            backup = f.with_name(f.stem + '_envelope_orig.ply')
            if not backup.exists():
                f.rename(backup)
                print(f"  [{cow_id}] backup → {backup.name}")

    colores = np.array([[139, 90, 43]] * len(all_px), dtype=np.uint8)

    ply_lat = out_dir / f'{cow_id}_lateral.ply'
    guardar_ply(str(ply_lat), all_px, tris_arr, colores, simetrico=False,
                escala_info=f'Consenso A_mediana | n={len(contornos)} frames | alto={altura_cm:.1f}cm')

    # 3D: malla cerrada (silueta espejada). Su volumen encerrado es el volumen
    # reportado — única fuente de volumen (sin rebanadas/cilindros).
    ply_3d = out_dir / f'{cow_id}_3d.ply'
    pts_3d, tris_3d = guardar_ply(str(ply_3d), all_px, tris_arr, colores, simetrico=True,
                                  escala_info=f'Consenso A_mediana | alto={altura_cm:.1f}cm')
    vol_barril = volumen_malla_cerrada(pts_3d, tris_3d)

    # Update resumen.json
    res_path = out_dir / f'{cow_id}_resumen.json'
    if res_path.exists():
        d = json.load(open(res_path))
    else:
        d = {}
    d['individuo'] = cow_id
    d['altura_real_cm'] = altura_cm
    if peso_real is not None:
        d['peso_real_kg'] = peso_real
    d['frames_usados'] = len(contornos)
    d['vol_barril_litros'] = vol_barril
    d['vol_barril_metodo'] = 'malla_cerrada_3d'
    d['width_consenso_cm'] = round(width_med, 1)
    d['alto_max_consenso_cm'] = round(max(hs_med), 1)
    json.dump(d, open(res_path, 'w'), indent=2)

    return {
        'cow_id': cow_id, 'vol_med': vol_barril, 'width_med': width_med,
        'alto_max': max(hs_med), 'verts': len(all_px), 'tris': len(tris_arr),
        'frames': len(contornos),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solo', default='', help='lista de IDs separados por coma (ej. v1,v2)')
    ap.add_argument('--carpeta-frames', default='checkpoints/22abril')
    args = ap.parse_args()

    ALL = ['v1','v2','v3','v4','v5','v7','v8','v9','v10','v12','v13','v14','v15']
    only = set(s.strip() for s in args.solo.split(',')) if args.solo else None
    if only:
        ALL = [v for v in ALL if v in only]

    alturas = json.load(open(PROJ / 'alturas_individuos.json'))
    H = alturas['alturas_22abril_cm']
    W = alturas.get('pesos_22abril_kg', {})

    print('[init] cargando modelos YOLO...')
    barril_model = YOLO(str(PROJ / 'barril_seg.pt'))
    coco_model = YOLO(str(PROJ / 'yolov8n.pt'))

    print(f"\n{'ID':4s}  {'altura':>7s}  {'frames':>6s}  {'vol':>7s}  {'width':>7s}  {'alto':>6s}  verts/tris")
    print('-' * 65)
    for v in ALL:
        if v not in H:
            print(f"  [{v}] sin altura, skip"); continue
        frames_dir = PROJ / args.carpeta_frames / v
        if not frames_dir.is_dir():
            print(f"  [{v}] no existe {frames_dir}, skip"); continue
        out_dir = PROJ / 'output_modelos3d_live' / v
        r = procesar(v, H[v], frames_dir, coco_model, barril_model, out_dir, peso_real=W.get(v))
        if r:
            print(f"{v:4s}  {H[v]:>5.1f}cm  {r['frames']:>6d}  {r['vol_med']:>5.1f}L  {r['width_med']:>5.1f}cm  {r['alto_max']:>4.1f}cm  {r['verts']}/{r['tris']}")
    print('\n[ok] mallas regeneradas con A_mediana')


if __name__ == '__main__':
    main()
