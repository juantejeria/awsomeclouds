"""Diagnostico de varianza: para unos individuos, mide cruz->cola y reporta el
CV (desviacion relativa) de dist_px, alto_bbox y dist_cm por separado.
Asi sabemos si la inestabilidad viene de la LINEA (px) o de la ESCALA (bbox)."""
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


def medir(img, altura):
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
    lm = int(fs[btopmin:bmid, :bxmin].sum()); rm = int(fs[btopmin:bmid, bxmax + 1:].sum())
    head_left = lm >= rm
    belly = {int(bxs[k]): int(bbot[k]) for k in range(len(bxs))}
    band = 0.22 * body_h; usepx = []
    for x in range(bxmin, bxmax + 1):
        ref = belly.get(x)
        if ref is None:
            continue
        col = np.where(fs[:, x] > 0)[0]; below = col[col > ref]
        if below.size < 3:
            continue
        isfront = (x < xc) if head_left else (x > xc)
        for yy in below:
            if isfront and yy <= ref + band:
                usepx.append((x, int(yy)))
    spine = np.percentile(btop, 12); thr = spine + 0.12 * body_h
    inb = np.where(btsm <= thr)[0]
    if len(usepx) >= 8:
        cruz_x = int(round(np.mean([p[0] for p in usepx])))
        ci = int(np.clip(cruz_x - bxmin, 0, len(bxs) - 1)); cruz = (int(bxs[ci]), int(btsm[ci]))
    else:
        ci = inb[0] if head_left else inb[-1]; cruz = (int(bxs[ci]), int(btop[ci]))
    spine_y = int(np.clip(round(np.percentile(btop, 12)), btopmin, bbotmax))
    row = np.where(fb[spine_y] > 0)[0]
    cola_x = (int(row.max()) if head_left else int(row.min())) if row.size else (int(bxmax) if head_left else int(bxmin))
    ci2 = int(np.clip(cola_x - bxmin, 0, len(bxs) - 1)); cola = (int(bxs[ci2]), int(btop[ci2]))
    dist_px = float(np.hypot(cruz[0] - cola[0], cruz[1] - cola[1]))
    bbox_h = max(1, by2 - by1)
    barril_h = body_h
    return dist_px, bbox_h, barril_h, dist_px * altura / bbox_h, dist_px * altura / barril_h


def cv(v):
    v = np.array(v, float)
    return 100 * v.std() / v.mean() if v.mean() else 0.0


def main(inds):
    for ind in inds:
        base = PROJ / 'checkpoints' / '14mayo' / ind
        if not base.exists():
            base = PROJ / 'checkpoints' / '20mayo' / ind
        altura = float(ind.split('_')[0])
        px = []; bh = []; rh = []; cm_bbox = []; cm_barril = []
        for fp in sorted(base.glob('frame_*.jpg')):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            r = medir(img, altura)
            if r is None:
                continue
            px.append(r[0]); bh.append(r[1]); rh.append(r[2]); cm_bbox.append(r[3]); cm_barril.append(r[4])
        print(f"\n=== {ind}  (n={len(px)}) ===")
        print(f"  dist_px      CV={cv(px):5.1f}%   [{min(px):.0f}-{max(px):.0f}]")
        print(f"  alto_bbox    CV={cv(bh):5.1f}%   [{min(bh):.0f}-{max(bh):.0f}]  (escala actual)")
        print(f"  alto_barril  CV={cv(rh):5.1f}%   [{min(rh):.0f}-{max(rh):.0f}]  (escala alternativa)")
        print(f"  dist_cm (bbox)    CV={cv(cm_bbox):5.1f}%")
        print(f"  dist_cm (barril)  CV={cv(cm_barril):5.1f}%")


if __name__ == '__main__':
    main(sys.argv[1:] or ['113_214', '100_137.5', '105_182.5', '110_221'])
