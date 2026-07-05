"""Medida alternativa: AXILA -> VERIJA (largo de panza entre patas).

Hipotesis: el lomo (cruz->cola) tiene mucha desviacion entre los 21 frames del
mismo individuo. Las inserciones de las patas en la panza son puntos mas
estables: aunque la pata se balancee, el borde donde la pata entra a la panza
se mueve poco.

Metodo:
- Barril (barril_seg) = panza; su contorno inferior (bbot) es la linea de panza.
- Patas = silueta (silueta_seg) que cuelga POR DEBAJO de la panza.
- Por columna, leg_mass = px de silueta debajo de la panza. Las patas dan
  leg_mass alto; el hueco de la panza (entre patas) da leg_mass ~0.
- Axila = borde interno (hacia el centro) de la pata DELANTERA (lado cabeza).
- Verija = borde interno de la pata TRASERA (lado cola).
- Linea recta axila->verija; distancia px y cm (altura_real/alto_bbox por frame).

Salidas en output_axila_verija/ (grids, overview, resumen.csv con rango).
Uso:  python muestra_axila_verija.py
"""
import cv2, numpy as np, csv
from pathlib import Path
from ultralytics import YOLO

PROJ = Path(__file__).parent
OUT = PROJ / 'output_axila_verija'
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


def runs(mask):
    """Lista de (ini, fin) inclusive de tramos True consecutivos."""
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out


def medir_frame(img):
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
    btopmin = int(btop.min()); bbotmax = int(bbot.max()); body_h = bbotmax - btopmin
    bmid = (btopmin + bbotmax) // 2
    # direccion: masa de silueta arriba, fuera del barril -> cabeza
    lm = int(fs[btopmin:bmid, :bxmin].sum()); rm = int(fs[btopmin:bmid, bxmax + 1:].sum())
    head_left = lm >= rm
    # leg_mass por columna = silueta por debajo de la panza
    legmass = np.zeros(len(bxs))
    for k, x in enumerate(bxs):
        ref = int(bbot[k])
        col = np.where(fs[:, x] > 0)[0]
        legmass[k] = int((col > ref).sum())
    present = legmass > 0.12 * body_h
    # suavizar: exigir tramos de >=3 columnas
    present = np.array([present[max(0, k - 1):k + 2].sum() >= 2 for k in range(len(present))])
    rr = runs(present)
    if not rr:
        return None
    xc_k = int(round(xc - bxmin))
    if head_left:
        # cabeza izq: delantera = izquierda (k bajo); axila = borde DERECHO (interno)
        front = [(a, b) for (a, b) in rr if (a + b) / 2 < xc_k]
        rear = [(a, b) for (a, b) in rr if (a + b) / 2 >= xc_k]
        if not front or not rear:
            return None
        ax_k = max(b for (a, b) in front)       # borde interno pata delantera
        ve_k = min(a for (a, b) in rear)         # borde interno pata trasera
    else:
        front = [(a, b) for (a, b) in rr if (a + b) / 2 >= xc_k]
        rear = [(a, b) for (a, b) in rr if (a + b) / 2 < xc_k]
        if not front or not rear:
            return None
        ax_k = min(a for (a, b) in front)
        ve_k = max(b for (a, b) in rear)
    axila = (int(bxs[ax_k]), int(bbot[ax_k]))
    verija = (int(bxs[ve_k]), int(bbot[ve_k]))
    dist_px = float(np.hypot(axila[0] - verija[0], axila[1] - verija[1]))
    cmpp = parse_altura.current / max(1, (by2 - by1))
    dist_cm = dist_px * cmpp
    ov = img.copy()
    ov[fb > 0] = (0.6 * ov[fb > 0] + 0.4 * np.array([0, 165, 255])).astype(np.uint8)
    cv2.line(ov, axila, verija, (0, 255, 0), 3)
    cv2.putText(ov, "<=cab" if head_left else "cab=>", (cx1, cy1 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tile = ov[max(0, cy1 - 40):min(H, cy2 + 60), max(0, cx1 - 40):min(W, cx2 + 40)].copy()
    return {'axila': axila, 'verija': verija, 'dist_px': dist_px, 'dist_cm': dist_cm, 'tile': tile}


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
            print(f"[skip] {base}"); continue
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
            cv2.imwrite(str(OUT / 'grids' / f"{dataset}_{ind_dir.name}.png"), grid)
            ot = central_tile if central_tile is not None else tiles[len(tiles) // 2]
            ot = cv2.resize(ot, (360, 240))
            cv2.putText(ot, f"{ind_dir.name}", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 3)
            cv2.putText(ot, f"med {med:.0f}cm", (8, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            overview.append(ot)
            rng = round(max(dists) - min(dists), 1)
            cv = round(100 * float(np.std(dists)) / med, 1) if med else 0.0
            rows.append({'dataset': dataset, 'individuo': ind_dir.name, 'altura_cm': altura,
                         'n_frames': len(dists), 'mediana_cm': round(med, 1),
                         'mediana_recortada_cm': round(tmed, 1),
                         'min_cm': round(min(dists), 1), 'max_cm': round(max(dists), 1),
                         'rango_cm': rng, 'cv_pct': cv})
            print(f"  {dataset}/{ind_dir.name}: n={len(dists)} med={med:.1f} "
                  f"rango={min(dists):.0f}-{max(dists):.0f} (={rng:.0f}) cv={cv:.1f}%")
        if overview:
            cv2.imwrite(str(OUT / f"overview_{dataset}.png"), montar_grid(overview, 3, 240, 360))
            print(f"[overview] {dataset} -> {len(overview)} individuos")
    with open(OUT / 'resumen.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'individuo', 'altura_cm', 'n_frames',
                                          'mediana_cm', 'mediana_recortada_cm',
                                          'min_cm', 'max_cm', 'rango_cm', 'cv_pct'])
        w.writeheader(); w.writerows(rows)
    if rows:
        print(f"\n[done] {len(rows)} individuos | rango medio={np.mean([r['rango_cm'] for r in rows]):.1f}cm "
              f"cv medio={np.mean([r['cv_pct'] for r in rows]):.1f}% -> {OUT}")


if __name__ == '__main__':
    main()
