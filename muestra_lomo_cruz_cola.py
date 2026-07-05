"""
Muestra diagnóstica: medida cruz->cola (lomo) en línea recta sobre los 21 frames
de cada individuo de 14mayo y 20mayo.

Método (v5):
- Dirección: masa de silueta (silueta_seg) fuera del barril, en la mitad superior.
  El lado con más masa = cabeza.
- Cruz: inserción de las patas delanteras. Se detectan las patas (silueta por
  debajo de la panza del barril), se toma SOLO la banda superior (inserción) de
  las patas del lado de la cabeza, su centroide x, y se sube a la topline del
  barril. (Las pezuñas se abren al caminar, la inserción se mantiene en el hombro.)
- Cola: extremo trasero de la "meseta" del lomo (continuación de la columna).
- Distancia recta cruz->cola en px y en cm (altura_real / alto_bbox por frame).

Salidas en output_lomo_cruz/:
  grids/<dataset>_<individuo>.png   -> 21 frames anotados por individuo
  overview_<dataset>.png            -> 1 frame por individuo, todos juntos
  resumen.csv                       -> mediana / mediana_recortada / rango por individuo

Uso:  python muestra_lomo_cruz_cola.py
"""
import cv2, numpy as np, csv
from pathlib import Path
from ultralytics import YOLO

PROJ = Path(__file__).parent
OUT = PROJ / 'output_lomo_cruz'
(OUT / 'grids').mkdir(parents=True, exist_ok=True)
SOURCES = {'14mayo': PROJ / 'checkpoints' / '14mayo',
           '20mayo': PROJ / 'checkpoints' / '20mayo'}

print("[init] cargando modelos...")
coco = YOLO(str(PROJ / 'yolov8n.pt'))
barril = YOLO(str(PROJ / 'barril_seg.pt'))
sil = YOLO(str(PROJ / 'silueta_seg.pt'))


def parse_altura(name):
    try:
        return float(name.split('_')[0])
    except Exception:
        return None


def detect(img):
    H, W = img.shape[:2]
    r = coco(img, classes=[19], conf=0.2, verbose=False)
    if not r or len(r[0].boxes) == 0:
        return None
    b = r[0].boxes.xyxy.cpu().numpy(); s = r[0].boxes.conf.cpu().numpy()
    i = int(np.argmax(s)); x1, y1, x2, y2 = [int(v) for v in b[i]]
    pad = max(20, int(0.08 * max(x2 - x1, y2 - y1)))
    return (max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)), (x1, y1, x2, y2)


def seg(model, crop):
    r = model(crop, conf=0.25, verbose=False)
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        return None
    m = r[0].masks.data.cpu().numpy(); a = np.array([x.sum() for x in m])
    k = a >= 0.05 * a.max(); s = np.max(m[k], axis=0)
    if s.shape != crop.shape[:2]:
        s = cv2.resize(s, (crop.shape[1], crop.shape[0]))
    return (s > 0.5).astype(np.uint8)


def contour(f):
    cols = np.where(f.sum(0) > 0)[0]
    xs = np.arange(int(cols[0]), int(cols[-1]) + 1)
    top = np.full(len(xs), -1); bot = np.full(len(xs), -1)
    for k, x in enumerate(xs):
        rr = np.where(f[:, x] > 0)[0]
        if len(rr):
            top[k] = rr[0]; bot[k] = rr[-1]
    return xs, top, bot


def medir_frame(img):
    """Devuelve dict con cruz, cola, dist_px, dist_cm, overlay (recortado) o None."""
    d = detect(img)
    if d is None:
        return None
    (cx1, cy1, cx2, cy2), (bx1, by1, bx2, by2) = d
    crop = img[cy1:cy2, cx1:cx2]
    mb = seg(barril, crop); ms = seg(sil, crop)
    if mb is None or ms is None:
        return None
    H, W = img.shape[:2]
    fb = np.zeros((H, W), np.uint8); fb[cy1:cy2, cx1:cx2] = mb
    fs = np.zeros((H, W), np.uint8); fs[cy1:cy2, cx1:cx2] = ms
    bxs, btop, bbot = contour(fb)
    if len(bxs) < 20:
        return None
    bxmin, bxmax = bxs[0], bxs[-1]; xc = (bxmin + bxmax) / 2
    ker = 11; padk = ker // 2
    btsm = np.convolve(np.pad(btop, padk, mode='edge'), np.ones(ker) / ker, mode='valid')
    btopmin = int(btop.min()); bbotmax = int(bbot.max()); body_h = bbotmax - btopmin
    bmid = (btopmin + bbotmax) // 2
    # dirección
    lm = int(fs[btopmin:bmid, :bxmin].sum()); rm = int(fs[btopmin:bmid, bxmax + 1:].sum())
    head_left = lm >= rm
    belly = {int(bxs[k]): int(bbot[k]) for k in range(len(bxs))}
    # patas + banda de inserción (front)
    band = 0.22 * body_h
    legpx = []; usepx = []
    for x in range(bxmin, bxmax + 1):
        ref = belly.get(x)
        if ref is None:
            continue
        col = np.where(fs[:, x] > 0)[0]; below = col[col > ref]
        if below.size < 3:
            continue
        isfront = (x < xc) if head_left else (x > xc)
        for yy in below:
            legpx.append((x, int(yy)))
            if isfront and yy <= ref + band:
                usepx.append((x, int(yy)))
    cruz = None
    if len(usepx) >= 8:
        cruz_x = int(round(np.mean([p[0] for p in usepx])))
        ci = int(np.clip(cruz_x - bxmin, 0, len(bxs) - 1))
        cruz = (int(bxs[ci]), int(btsm[ci]))
    spine = np.percentile(btop, 12); thr = spine + 0.12 * body_h
    inb = np.where(btsm <= thr)[0]
    if cruz is None:
        ci = inb[0] if head_left else inb[-1]
        cruz = (int(bxs[ci]), int(btop[ci]))
    # Cola = nacimiento de la cola = borde TRASERO del barril a la altura de la
    # columna. SIEMPRE sobre la linea del barril (ni por fuera = punta de la cola,
    # ni por dentro = corto). Tomamos la fila spine_y (percentil 12 de la topline,
    # bien arriba en el lomo, para que el borde quede en el nacimiento de la cola
    # y no se corra hacia atras) y el pixel de barril mas trasero en esa fila.
    spine_y = int(np.clip(round(np.percentile(btop, 12)), btopmin, bbotmax))
    row = np.where(fb[spine_y] > 0)[0]
    if row.size:
        cola_x = int(row.max()) if head_left else int(row.min())
    else:
        cola_x = int(bxmax) if head_left else int(bxmin)
    ci2 = int(np.clip(cola_x - bxmin, 0, len(bxs) - 1))
    cola = (int(bxs[ci2]), int(btop[ci2]))
    dist_px = float(np.hypot(cruz[0] - cola[0], cruz[1] - cola[1]))
    cmpp = parse_altura.current / max(1, (by2 - by1))
    dist_cm = dist_px * cmpp
    # overlay
    ov = img.copy()
    ov[fb > 0] = (0.6 * ov[fb > 0] + 0.4 * np.array([0, 165, 255])).astype(np.uint8)
    for x, y in legpx:
        ov[y, x] = (0, 140, 255)
    for x, y in usepx:
        ov[y, x] = (0, 255, 0)
    cv2.line(ov, cruz, cola, (0, 255, 0), 3)
    cv2.putText(ov, "<=cab" if head_left else "cab=>", (cx1, cy1 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tile = ov[max(0, cy1 - 40):min(H, cy2 + 60), max(0, cx1 - 40):min(W, cx2 + 40)].copy()
    return {'cruz': cruz, 'cola': cola, 'dist_px': dist_px, 'dist_cm': dist_cm, 'tile': tile}


def montar_grid(tiles, ncol, th, tw):
    cells = [cv2.resize(t, (tw, th)) for t in tiles]
    nrow = (len(cells) + ncol - 1) // ncol
    grid = np.zeros((nrow * th, ncol * tw, 3), np.uint8)
    for i, c in enumerate(cells):
        r, cc = divmod(i, ncol); grid[r * th:(r + 1) * th, cc * tw:(cc + 1) * tw] = c
    return grid


def trimmed_median(vals, frac=0.2):
    if not vals:
        return 0.0
    v = sorted(vals); n = len(v); k = int(n * frac)
    core = v[k:n - k] if n - 2 * k >= 1 else v
    return float(np.median(core))


def main():
    rows = []
    for dataset, base in SOURCES.items():
        if not base.exists():
            print(f"[skip] {base}")
            continue
        overview = []
        for ind_dir in sorted(base.iterdir()):
            if not ind_dir.is_dir():
                continue
            frames = sorted(ind_dir.glob('frame_*.jpg'))
            if not frames:
                continue
            altura = parse_altura(ind_dir.name)
            if altura is None:
                print(f"[warn] sin altura: {ind_dir.name}"); continue
            parse_altura.current = altura
            tiles = []; dists = []; central_tile = None
            for fp in frames:
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                res = medir_frame(img)
                if res is None:
                    continue
                lbl = f"{fp.stem.split('_')[1]} {res['dist_px']:.0f}px {res['dist_cm']:.0f}cm"
                t = res['tile']
                cv2.putText(t, lbl, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
                cv2.putText(t, lbl, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                tiles.append(t); dists.append(res['dist_cm'])
                if fp.stem.startswith('frame_000'):
                    central_tile = t.copy()
            if not tiles:
                print(f"[warn] {dataset}/{ind_dir.name}: 0 frames OK"); continue
            med = float(np.median(dists)); tmed = trimmed_median(dists)
            grid = montar_grid(tiles, 3, 240, 360)
            gpath = OUT / 'grids' / f"{dataset}_{ind_dir.name}.png"
            cv2.imwrite(str(gpath), grid)
            # overview tile
            ot = central_tile if central_tile is not None else tiles[len(tiles) // 2]
            ot = cv2.resize(ot, (360, 240))
            cv2.putText(ot, f"{ind_dir.name}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
            cv2.putText(ot, f"med {med:.0f}cm", (8, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            overview.append(ot)
            rows.append({'dataset': dataset, 'individuo': ind_dir.name, 'altura_cm': altura,
                         'n_frames': len(dists), 'mediana_cm': round(med, 1),
                         'mediana_recortada_cm': round(tmed, 1),
                         'min_cm': round(min(dists), 1), 'max_cm': round(max(dists), 1),
                         'rango_cm': round(max(dists) - min(dists), 1)})
            print(f"  {dataset}/{ind_dir.name}: n={len(dists)} med={med:.1f} trim={tmed:.1f} "
                  f"rango={min(dists):.0f}-{max(dists):.0f}")
        if overview:
            ovg = montar_grid(overview, 3, 240, 360)
            cv2.imwrite(str(OUT / f"overview_{dataset}.png"), ovg)
            print(f"[overview] {dataset} -> {len(overview)} individuos")
    # CSV
    with open(OUT / 'resumen.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'individuo', 'altura_cm', 'n_frames',
                                          'mediana_cm', 'mediana_recortada_cm',
                                          'min_cm', 'max_cm', 'rango_cm'])
        w.writeheader(); w.writerows(rows)
    print(f"\n[done] {len(rows)} individuos -> {OUT}")


if __name__ == '__main__':
    main()
