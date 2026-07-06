"""Diagnostico: dibuja 3 candidatos para el COMIENZO DE LA COLA sobre el frame.
  A (magenta) = borde trasero del BARRIL a la altura de la columna  (lo actual)
  B (verde)   = esquina del anca en la SILUETA (rumbo columna, parte maciza)
  C (rojo)    = extremo trasero de la SILUETA (suele ser la punta de la cola)
Tambien dibuja la cruz (azul) y la linea cruz->B para referencia.
Contornos: barril=azul claro, silueta=amarillo."""
import cv2, numpy as np, sys
from pathlib import Path
from ultralytics import YOLO

PROJ = Path(__file__).parent
coco = YOLO(str(PROJ / 'yolov8n.pt'))
barril = YOLO(str(PROJ / 'barril_seg.pt'))
sil = YOLO(str(PROJ / 'silueta_seg.pt'))


def detect(img):
    H, W = img.shape[:2]
    r = coco(img, classes=[19], conf=0.2, verbose=False)
    if not r or len(r[0].boxes) == 0:
        return None
    b = r[0].boxes.xyxy.cpu().numpy(); s = r[0].boxes.conf.cpu().numpy()
    i = int(np.argmax(s)); x1, y1, x2, y2 = [int(v) for v in b[i]]
    pad = max(20, int(0.08 * max(x2 - x1, y2 - y1)))
    return (max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad))


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


def dbg(path, out):
    img = cv2.imread(str(path))
    d = detect(img)
    cx1, cy1, cx2, cy2 = d
    crop = img[cy1:cy2, cx1:cx2]
    mb = seg(barril, crop); ms = seg(sil, crop)
    H, W = img.shape[:2]
    fb = np.zeros((H, W), np.uint8); fb[cy1:cy2, cx1:cx2] = mb
    fs = np.zeros((H, W), np.uint8); fs[cy1:cy2, cx1:cx2] = ms
    bxs, btop, bbot = contour(fb)
    bxmin, bxmax = bxs[0], bxs[-1]
    btopmin = int(btop.min()); bbotmax = int(bbot.max()); body_h = bbotmax - btopmin
    bmid = (btopmin + bbotmax) // 2
    lm = int(fs[btopmin:bmid, :bxmin].sum()); rm = int(fs[btopmin:bmid, bxmax + 1:].sum())
    head_left = lm >= rm
    spine = np.percentile(btop, 12); thr = spine + 0.12 * body_h
    spine_y = int(np.clip(round(np.percentile(btop, 20)), btopmin, bbotmax))

    sxs, stop, sbot = contour(fs)
    ker = 11; padk = ker // 2
    stsm = np.convolve(np.pad(stop, padk, mode='edge'), np.ones(ker) / ker, mode='valid')
    col_mass = fs[:, sxs].sum(axis=0)

    def barrel_edge(pct):
        sy = int(np.clip(round(np.percentile(btop, pct)), btopmin, bbotmax))
        r = np.where(fb[sy] > 0)[0]
        if not r.size:
            return (int(bxmax) if head_left else int(bxmin), sy)
        xx = int(r.max()) if head_left else int(r.min())
        cc = int(np.clip(xx - bxmin, 0, len(bxs) - 1))
        return (int(bxs[cc]), int(btop[cc]))
    A = barrel_edge(20)          # actual (magenta)
    A8 = barrel_edge(8)          # mas alto / mas adelante (cyan)
    j = len(sxs) - 1 if head_left else 0; C = (int(sxs[j]), int(stop[j]))

    ov = img.copy()
    cb, _ = cv2.findContours(fb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs, _ = cv2.findContours(fs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cs, -1, (0, 255, 255), 1)
    cv2.drawContours(ov, cb, -1, (255, 150, 0), 1)
    for p, col, lbl in [(A, (255, 0, 255), 'A20'), (A8, (255, 255, 0), 'A8'), (C, (0, 0, 255), 'C')]:
        cv2.circle(ov, p, 3, col, -1)
        cv2.putText(ov, lbl, (p[0] + 5, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    tile = ov[max(0, cy1 - 50):min(H, cy2 + 70), max(0, cx1 - 70):min(W, cx2 + 70)]
    tile = cv2.resize(tile, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out, tile)
    print('escrito', out, '| head_left=', head_left)


if __name__ == '__main__':
    ind = sys.argv[1] if len(sys.argv) > 1 else '113_214'
    base = PROJ / 'checkpoints' / '14mayo' / ind
    frames = sorted(base.glob('frame_000*.jpg')) or sorted(base.glob('frame_*.jpg'))
    dbg(frames[0], str(PROJ / f'debug_cola_{ind}.png'))
