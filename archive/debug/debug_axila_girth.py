"""Diagnostico: contorno inferior del BARRIL para ubicar la AXILA (girth).
Dibuja el contorno del barril, su linea inferior (panza) coloreada por pendiente,
marca minimos/maximos locales del fondo y la posicion actual girthFrac=0.20.
Asi vemos si hay un 'cambio de sentido' detectable en la axila (entre pata y panza).
"""
import cv2, numpy as np, sys
from pathlib import Path
from ultralytics import YOLO

P = Path(__file__).parent
coco = YOLO(str(P / 'yolov8n.pt'))
barril = YOLO(str(P / 'barril_seg.pt'))
sil = YOLO(str(P / 'silueta_seg.pt'))


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


def contour_bot_top(f):
    cols = np.where(f.sum(0) > 0)[0]
    xs = np.arange(int(cols[0]), int(cols[-1]) + 1)
    top = np.full(len(xs), -1.0); bot = np.full(len(xs), -1.0)
    for k, x in enumerate(xs):
        rr = np.where(f[:, x] > 0)[0]
        if len(rr):
            top[k] = rr[0]; bot[k] = rr[-1]
    return xs, top, bot


def head_left_of(barmask, silmask):
    cols = np.where(barmask.sum(0) > 0)[0]; rows = np.where(barmask.sum(1) > 0)[0]
    bxmin, bxmax = int(cols[0]), int(cols[-1]); btop, bbot = int(rows[0]), int(rows[-1])
    bmid = (btop + bbot) // 2
    lm = int(silmask[btop:bmid, :bxmin].sum()); rm = int(silmask[btop:bmid, bxmax + 1:].sum())
    return lm >= rm


def run(ind, ds):
    base = P / 'checkpoints' / ds / ind
    fp = sorted(base.glob('frame_000*.jpg'))[0]
    img = cv2.imread(str(fp))
    r = coco(img, classes=[19], conf=0.2, verbose=False)
    b = r[0].boxes.xyxy.cpu().numpy(); s = r[0].boxes.conf.cpu().numpy(); i = int(np.argmax(s))
    x1, y1, x2, y2 = [int(v) for v in b[i]]; pad = max(20, int(0.08 * max(x2 - x1, y2 - y1)))
    H, W = img.shape[:2]; ox1, oy1 = max(0, x1 - pad), max(0, y1 - pad)
    crop = img[oy1:min(H, y2 + pad), ox1:min(W, x2 + pad)]
    bm = seg(barril, crop); sm = seg(sil, crop)
    xs, top, bot = contour_bot_top(bm)
    L = xs[-1] - xs[0]
    hl = head_left_of(bm, sm)
    xfront = xs[-1] if not hl else xs[0]   # frente = lado cabeza
    # suavizar contornos
    k = 9; pad2 = k // 2
    sm = lambda a: np.convolve(np.pad(a, pad2, mode='edge'), np.ones(k) / k, mode='valid')
    bsm = sm(bot); tsm = sm(top)
    height = bsm - tsm                  # alto del barril por columna
    n = len(xs)
    # region delantera (45% del largo desde el frente)
    if hl:  # frente a la izquierda
        front_idx = np.arange(0, int(0.45 * n))
    else:   # frente a la derecha
        front_idx = np.arange(int(0.55 * n), n)
    # Candidato A: maxima profundidad del torax en la zona delantera
    A_k = front_idx[int(np.argmax(height[front_idx]))]
    # Candidato B: maxima curvatura del contorno inferior en la zona delantera
    d2 = np.gradient(np.gradient(bsm))
    B_k = front_idx[int(np.argmax(np.abs(d2[front_idx])))]

    ov = img.copy()
    cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov[oy1:oy1 + crop.shape[0], ox1:ox1 + crop.shape[1]], cnts, -1, (255, 200, 0), 1)
    for kk in range(n):
        cv2.circle(ov, (ox1 + int(xs[kk]), oy1 + int(bsm[kk])), 1, (200, 200, 200), -1)

    def vline(kk, color, label, dy):
        gx = ox1 + int(xs[kk])
        cv2.line(ov, (gx, oy1 + int(tsm[kk])), (gx, oy1 + int(bsm[kk])), color, 2)
        cv2.putText(ov, label, (gx - 20, oy1 + int(tsm[kk]) - dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # girthFrac actual = 0.20 desde el frente (amarillo)
    gx20 = (xfront - 0.20 * L) if not hl else (xfront + 0.20 * L)
    g20k = int(np.clip(gx20 - xs[0], 0, n - 1))
    vline(g20k, (0, 255, 255), "20%", 6)
    vline(A_k, (0, 230, 0), "A maxprof", 22)        # verde
    vline(B_k, (255, 0, 255), "B inflex", 38)       # magenta
    fr = lambda kk: abs(xs[kk] - xfront) / L
    cv2.putText(ov, f"head={'LEFT' if hl else 'RIGHT'}  A={fr(A_k):.2f} B={fr(B_k):.2f}",
                (ox1, oy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    tile = ov[max(0, oy1 - 30):min(H, y2 + pad + 30), max(0, ox1 - 30):min(W, x2 + pad + 30)]
    tile = cv2.resize(tile, None, fx=2.0, fy=2.0)
    out = str(P / f'debug_axila_{ds}_{ind}.png')
    cv2.imwrite(out, tile); print('escrito', out, '| head_left=', hl)


if __name__ == '__main__':
    run('113_214', '14mayo')
    run('127_435', '20mayo')
    run('110_221', '14mayo')
