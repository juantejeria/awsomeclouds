"""Backfill: completa 'barril_dir' (sentido del animal) en los _resumen.json de
los modelos 3D ya generados, re-detectando sobre los frames de 'carpeta_origen'
con voto mayoritario. Mismo criterio que app.py/_detectar_sentido_barril y
muestra_lomo_cruz_cola.py (head_left).

El visor (viewer3d.js) usa barril_dir para ubicar el diámetro torácico del lado
de la cabeza. Mapeo: cabeza a la IZQUIERDA de la imagen -> 'left'.

Uso:  python backfill_barril_dir.py [--force]
  --force : recalcula aunque ya tenga barril_dir left/right.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

PROJ = Path(__file__).parent
DIRS = ['output_modelos3d_live_14mayo', 'output_modelos3d_live_20mayo',
        'output_modelos3d_live_14mayo_v7']
MAX_FRAMES = 9   # frames a votar por individuo
FORCE = '--force' in sys.argv

print("[init] cargando modelos...")
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


def sentido_frame(img):
    r = coco(img, classes=[19], conf=0.2, verbose=False)
    if not r or len(r[0].boxes) == 0:
        return None
    b = r[0].boxes.xyxy.cpu().numpy(); s = r[0].boxes.conf.cpu().numpy()
    i = int(np.argmax(s)); x1, y1, x2, y2 = [int(v) for v in b[i]]
    pad = max(20, int(0.08 * max(x2 - x1, y2 - y1)))
    H, W = img.shape[:2]
    crop = img[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
    bm = seg(barril, crop); sm = seg(sil, crop)
    if bm is None or sm is None:
        return None
    cols = np.where(bm.sum(0) > 0)[0]; rows = np.where(bm.sum(1) > 0)[0]
    if not len(cols) or not len(rows):
        return None
    bxmin, bxmax = int(cols[0]), int(cols[-1]); btop, bbot = int(rows[0]), int(rows[-1])
    bmid = (btop + bbot) // 2
    lm = int(sm[btop:bmid, :bxmin].sum()); rm = int(sm[btop:bmid, bxmax + 1:].sum())
    if lm == rm:
        return None
    return 'left' if lm > rm else 'right'


def carpeta_frames(meta):
    co = meta.get('carpeta_origen')
    if not co:
        return None
    p = (PROJ / co)
    return p if p.is_dir() else None


def main():
    total = ok = skip = fail = 0
    for d in DIRS:
        base = PROJ / d
        if not base.is_dir():
            continue
        print(f"\n=== {d} ===")
        for sub in sorted(base.iterdir()):
            if not sub.is_dir() or sub.name.startswith('_'):
                continue
            rj = next(iter(sub.glob('*_resumen.json')), None)
            if rj is None:
                continue
            total += 1
            meta = json.loads(rj.read_text())
            cur = (meta.get('barril_dir') or 'unknown')
            if cur in ('left', 'right') and not FORCE:
                print(f"  {sub.name:14s} ya tiene barril_dir={cur} (skip)"); skip += 1; continue
            fdir = carpeta_frames(meta)
            if fdir is None:
                print(f"  {sub.name:14s} sin carpeta_origen válida -> no se puede (skip)"); fail += 1; continue
            frames = sorted(fdir.glob('frame_*.jpg')) or sorted(fdir.glob('*.jpg')) or sorted(fdir.glob('*.png'))
            if not frames:
                print(f"  {sub.name:14s} sin frames en {fdir} (skip)"); fail += 1; continue
            # muestrear hasta MAX_FRAMES repartidos
            if len(frames) > MAX_FRAMES:
                idx = np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int)
                frames = [frames[i] for i in idx]
            votes = {'left': 0, 'right': 0}
            for fp in frames:
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                v = sentido_frame(img)
                if v in votes:
                    votes[v] += 1
            if votes['left'] == votes['right']:
                print(f"  {sub.name:14s} voto empatado {votes} -> dejo unknown"); fail += 1; continue
            new = 'left' if votes['left'] > votes['right'] else 'right'
            meta['barril_dir'] = new
            rj.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            print(f"  {sub.name:14s} barril_dir={new}  votos={votes}")
            ok += 1
    print(f"\n[done] total={total}  escritos={ok}  ya_tenian={skip}  sin_resolver={fail}")


if __name__ == '__main__':
    main()
