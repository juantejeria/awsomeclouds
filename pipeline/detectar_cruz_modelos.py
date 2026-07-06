"""
Detecta el PUNTO DE LA CRUZ de cada modelo 3D con el modelo entrenado cruz_pose.pt
(YOLO-pose, 1 keypoint = cruz).

Corre sobre los frames fuente de cada individuo (checkpoints/<dataset>/<ind>/frame_*.jpg)
de los datasets live, agrega una cruz robusta por individuo (mediana sobre los frames,
en coords normalizadas dentro del bbox del barril para ser independiente de escala) y
guarda:
  output_cruz_modelos/grids/<dataset>_<ind>.png   -> 21 frames con la cruz marcada
  output_cruz_modelos/overview_<dataset>.png       -> 1 frame por individuo
  output_cruz_modelos/cruz_resultados.csv          -> cruz por individuo (px + normalizada)

La cruz se normaliza respecto al bbox del barril (barril_seg) para que la posición sea
comparable entre frames/individuos:
  cruz_xn = (cruz_x - barril_xmin) / ancho_barril   (0 = borde trasero/delantero según dir)
  cruz_yn = (cruz_y - barril_ymin) / alto_barril

Uso:  python detectar_cruz_modelos.py [--dataset 20mayo]
"""
import argparse
import csv
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / 'output_cruz_modelos'
(OUT / 'grids').mkdir(parents=True, exist_ok=True)

# dataset -> carpeta de frames fuente
SOURCES = {
    '6mayo':  PROJ / 'checkpoints' / '6mayo',
    '14mayo': PROJ / 'checkpoints' / '14mayo',
    '20mayo': PROJ / 'checkpoints' / '20mayo',
    '12junio': PROJ / 'checkpoints' / '12 junio',
}

print("[init] cargando modelos...")
cruz_model = YOLO(str(PROJ / 'models' / 'cruz_pose.pt'))
barril = YOLO(str(PROJ / 'models' / 'barril_seg.pt'))


def seg_bbox(crop):
    """bbox (x0,y0,x1,y1) de la mascara de barril dentro de `crop`, o None."""
    r = barril(crop, conf=0.25, verbose=False)
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        return None
    m = r[0].masks.data.cpu().numpy()
    a = np.array([x.sum() for x in m])
    k = a >= 0.05 * a.max()
    s = np.max(m[k], axis=0)
    if s.shape != crop.shape[:2]:
        s = cv2.resize(s, (crop.shape[1], crop.shape[0]))
    ys, xs = (s > 0.5).nonzero()
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def detectar_cruz(img):
    """Devuelve (cruz_x, cruz_y, conf, box) del keypoint de la mejor deteccion, o None.
    box = (x0,y0,x1,y1) de la caja del animal (silueta+cruz) de cruz_pose."""
    r = cruz_model(img, conf=0.25, imgsz=640, verbose=False)
    if not r or r[0].keypoints is None or len(r[0].boxes) == 0:
        return None
    boxes = r[0].boxes
    kxy = r[0].keypoints.xy.cpu().numpy()   # (n, 1, 2)
    conf = boxes.conf.cpu().numpy()
    i = int(np.argmax(conf))
    if kxy[i].shape[0] == 0:
        return None
    cx, cy = float(kxy[i][0][0]), float(kxy[i][0][1])
    bx = boxes.xyxy.cpu().numpy()[i]
    box = (int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3]))
    return cx, cy, float(conf[i]), box


def barril_bbox_en_crop(img, box):
    """bbox del barril en coords del frame completo. Segmenta barril_seg dentro de
    un crop alrededor de la caja del animal (barril_seg falla sobre el frame entero
    por el fondo: cercas, arboles). Devuelve (x0,y0,x1,y1) o None."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    pad = int(0.15 * max(x1 - x0, y1 - y0)) + 20
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
    bb = seg_bbox(img[cy0:cy1, cx0:cx1])
    if bb is None:
        return None
    return bb[0] + cx0, bb[1] + cy0, bb[2] + cx0, bb[3] + cy0


def medir_frame(img):
    """Cruz + bbox barril + normalizacion. Devuelve dict o None."""
    det = detectar_cruz(img)
    if det is None:
        return None
    cx, cy, conf, box = det
    bb = barril_bbox_en_crop(img, box)
    res = {'cruz': (int(round(cx)), int(round(cy))), 'conf': conf, 'bbox': bb}
    if bb is not None:
        x0, y0, x1, y1 = bb
        w = max(1, x1 - x0); h = max(1, y1 - y0)
        xn = (cx - x0) / w
        # Aceptar solo si la cruz cae dentro/cerca del barril (margen 15%). Si cae
        # muy fuera, la mascara de barril de ese frame es mala -> no normalizamos.
        if -0.15 <= xn <= 1.15:
            res['cruz_xn'] = xn
            res['cruz_yn'] = (cy - y0) / h
    return res


def overlay(img, res):
    """Dibuja la cruz + bbox y recorta (zoom) alrededor del animal para que el
    punto sea claramente visible en el grid."""
    ov = img.copy()
    H, W = ov.shape[:2]
    cx, cy = res['cruz']
    if res.get('bbox') is not None:
        x0, y0, x1, y1 = res['bbox']
        cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 165, 255), 2)
        pad = int(0.18 * max(x1 - x0, y1 - y0)) + 20
    else:
        x0, y0, x1, y1 = cx - 120, cy - 120, cx + 120, cy + 120
        pad = 20
    cv2.drawMarker(ov, (cx, cy), (0, 215, 255), cv2.MARKER_CROSS, 28, 3)
    cv2.circle(ov, (cx, cy), 6, (0, 0, 255), -1)
    rx0 = max(0, min(x0, cx) - pad); ry0 = max(0, min(y0, cy) - pad)
    rx1 = min(W, max(x1, cx) + pad); ry1 = min(H, max(y1, cy) + pad)
    return ov[ry0:ry1, rx0:rx1].copy()


def montar_grid(tiles, ncol, th, tw):
    cells = [cv2.resize(t, (tw, th)) for t in tiles]
    nrow = (len(cells) + ncol - 1) // ncol
    grid = np.zeros((nrow * th, ncol * tw, 3), np.uint8)
    for i, c in enumerate(cells):
        r, cc = divmod(i, ncol)
        grid[r * th:(r + 1) * th, cc * tw:(cc + 1) * tw] = c
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', help='Procesar solo un dataset (p.ej. 20mayo)')
    args = ap.parse_args()

    datasets = SOURCES if not args.dataset else {args.dataset: SOURCES[args.dataset]}
    rows = []
    for dataset, base in datasets.items():
        if not base.exists():
            print(f"[skip] no existe {base}")
            continue
        overview = []
        for ind_dir in sorted(base.iterdir()):
            if not ind_dir.is_dir():
                continue
            frames = sorted(ind_dir.glob('frame_*.jpg'))
            if not frames:
                continue
            tiles = []
            xs_n, ys_n, confs = [], [], []
            central_tile = None
            for fp in frames:
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                res = medir_frame(img)
                if res is None:
                    continue
                confs.append(res['conf'])
                if 'cruz_xn' in res:
                    xs_n.append(res['cruz_xn']); ys_n.append(res['cruz_yn'])
                t = overlay(img, res)
                lbl = f"{fp.stem.split('_')[1]} c={res['conf']:.2f}"
                cv2.putText(t, lbl, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
                cv2.putText(t, lbl, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                tiles.append(t)
                if fp.stem.startswith('frame_000'):
                    central_tile = t.copy()
            if not tiles:
                print(f"[warn] {dataset}/{ind_dir.name}: 0 frames con cruz")
                continue
            grid = montar_grid(tiles, 3, 240, 360)
            cv2.imwrite(str(OUT / 'grids' / f"{dataset}_{ind_dir.name}.png"), grid)

            cxn = float(np.median(xs_n)) if xs_n else None
            cyn = float(np.median(ys_n)) if ys_n else None
            cconf = float(np.median(confs)) if confs else 0.0
            ot = central_tile if central_tile is not None else tiles[len(tiles) // 2]
            ot = cv2.resize(ot, (360, 240))
            cv2.putText(ot, ind_dir.name, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
            overview.append(ot)
            rows.append({
                'dataset': dataset, 'individuo': ind_dir.name,
                'n_frames': len(tiles),
                'cruz_xn': round(cxn, 4) if cxn is not None else '',
                'cruz_yn': round(cyn, 4) if cyn is not None else '',
                'conf_med': round(cconf, 3),
            })
            print(f"  {dataset}/{ind_dir.name}: n={len(tiles)} "
                  f"xn={cxn if cxn is None else round(cxn,3)} "
                  f"yn={cyn if cyn is None else round(cyn,3)} conf={cconf:.2f}")
        if overview:
            cv2.imwrite(str(OUT / f"overview_{dataset}.png"), montar_grid(overview, 3, 240, 360))
            print(f"[overview] {dataset} -> {len(overview)} individuos")

    with open(OUT / 'cruz_resultados.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'individuo', 'n_frames',
                                          'cruz_xn', 'cruz_yn', 'conf_med'])
        w.writeheader(); w.writerows(rows)
    print(f"\n[done] {len(rows)} individuos -> {OUT}")


if __name__ == '__main__':
    main()
