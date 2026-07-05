"""CV (desviacion relativa) del LARGO del barril (px) por individuo, en los 21
frames. Para saber, con numeros reales, que tan estable es el largo del barril
frame a frame. Reporta tambien alto del barril y largo/alto para referencia."""
import cv2, numpy as np, csv
from pathlib import Path
from ultralytics import YOLO

PROJ = Path(__file__).parent
SOURCES = {'14mayo': PROJ / 'checkpoints' / '14mayo', '20mayo': PROJ / 'checkpoints' / '20mayo'}
coco = YOLO(str(PROJ / 'yolov8n.pt'))
barril = YOLO(str(PROJ / 'barril_seg.pt'))


def detect(img):
    H, W = img.shape[:2]
    r = coco(img, classes=[19], conf=0.2, verbose=False)
    if not r or len(r[0].boxes) == 0:
        return None
    b = r[0].boxes.xyxy.cpu().numpy(); s = r[0].boxes.conf.cpu().numpy()
    i = int(np.argmax(s)); x1, y1, x2, y2 = [int(v) for v in b[i]]
    pad = max(20, int(0.08 * max(x2 - x1, y2 - y1)))
    return (max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad))


def seg(crop):
    r = barril(crop, conf=0.25, verbose=False)
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        return None
    m = r[0].masks.data.cpu().numpy(); a = np.array([x.sum() for x in m])
    k = a >= 0.05 * a.max(); s = np.max(m[k], axis=0)
    if s.shape != crop.shape[:2]:
        s = cv2.resize(s, (crop.shape[1], crop.shape[0]))
    return (s > 0.5).astype(np.uint8)


def cv(v):
    v = np.array(v, float)
    return 100 * v.std() / v.mean() if len(v) and v.mean() else 0.0


def main():
    allcv = []
    rows = []
    for dataset, base in SOURCES.items():
        if not base.exists():
            continue
        for ind in sorted(base.iterdir()):
            if not ind.is_dir():
                continue
            frames = sorted(ind.glob('frame_*.jpg'))
            if not frames:
                continue
            largos = []; altos = []
            for fp in frames:
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                d = detect(img)
                if d is None:
                    continue
                cx1, cy1, cx2, cy2 = d
                mb = seg(img[cy1:cy2, cx1:cx2])
                if mb is None:
                    continue
                cols = np.where(mb.sum(0) > 0)[0]; rws = np.where(mb.sum(1) > 0)[0]
                if not len(cols) or not len(rws):
                    continue
                largos.append(int(cols[-1] - cols[0]))
                altos.append(int(rws[-1] - rws[0]))
            if len(largos) < 3:
                print(f"  {dataset}/{ind.name}: pocos frames"); continue
            cl = cv(largos); ca = cv(altos)
            allcv.append(cl)
            rng = max(largos) - min(largos)
            rows.append((f"{dataset}/{ind.name}", len(largos), cl, ca, min(largos), max(largos), rng))
            print(f"  {dataset}/{ind.name:14s} n={len(largos):2d} | largo CV={cl:5.1f}% [{min(largos)}-{max(largos)} px, rango {rng}] | alto CV={ca:5.1f}%")
    if allcv:
        print(f"\n[resumen] largo barril: CV medio={np.mean(allcv):.1f}%  mediana={np.median(allcv):.1f}%  "
              f"min={np.min(allcv):.1f}%  max={np.max(allcv):.1f}%  (n={len(allcv)} individuos)")
        good = [c for c in allcv if c < 8]
        print(f"          individuos con CV<8%: {len(good)}/{len(allcv)}")


if __name__ == '__main__':
    main()
