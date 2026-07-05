"""Post-proceso de los modelos v8:
1) auto-detecta barril_dir (voto sobre frames de carpeta_origen) y lo guarda.
2) copia girth_frac_manual / verija_frac_manual desde los dirs viejos a los v8
   (solo 14mayo y 20mayo, que es donde el usuario anotó).
No toca los dirs viejos.
"""
import json
from pathlib import Path
import numpy as np, cv2
from ultralytics import YOLO

PROJ = Path(__file__).parent
V8 = {'output_modelos3d_live_14mayo_v8': 'output_modelos3d_live_14mayo',
      'output_modelos3d_live_20mayo_v8': 'output_modelos3d_live_20mayo',
      'output_modelos3d_live_6mayo_v8': None,
      'output_modelos3d_live_12junio_v8': None}
MAXF = 9

print("[init] modelos...")
coco = YOLO(str(PROJ/'yolov8n.pt')); barril = YOLO(str(PROJ/'barril_seg.pt')); sil = YOLO(str(PROJ/'silueta_seg.pt'))


def seg(model, crop):
    r = model(crop, conf=0.25, verbose=False)
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        return None
    m = r[0].masks.data.cpu().numpy(); a = np.array([float(x.sum()) for x in m])
    if a.max() <= 0:
        return None
    s = np.max(m[a >= 0.05*a.max()], axis=0)
    if s.shape != crop.shape[:2]:
        s = cv2.resize(s, (crop.shape[1], crop.shape[0]))
    return (s > 0.5).astype(np.uint8)


def sentido_frame(img):
    r = coco(img, classes=[19], conf=0.2, verbose=False)
    if not r or len(r[0].boxes) == 0:
        return None
    b = r[0].boxes.xyxy.cpu().numpy(); s = r[0].boxes.conf.cpu().numpy()
    i = int(np.argmax(s)); x1, y1, x2, y2 = [int(v) for v in b[i]]
    pad = max(20, int(0.08*max(x2-x1, y2-y1))); H, W = img.shape[:2]
    crop = img[max(0, y1-pad):min(H, y2+pad), max(0, x1-pad):min(W, x2+pad)]
    bm = seg(barril, crop); sm = seg(sil, crop)
    if bm is None or sm is None:
        return None
    cols = np.where(bm.sum(0) > 0)[0]; rows = np.where(bm.sum(1) > 0)[0]
    if not len(cols) or not len(rows):
        return None
    bxmin, bxmax = int(cols[0]), int(cols[-1]); btop, bbot = int(rows[0]), int(rows[-1]); bmid = (btop+bbot)//2
    lm = int(sm[btop:bmid, :bxmin].sum()); rm = int(sm[btop:bmid, bxmax+1:].sum())
    if lm == rm:
        return None
    return 'left' if lm > rm else 'right'


def detect_dir(frames_dir):
    fr = sorted(Path(frames_dir).glob('frame_*.jpg'))
    if not fr:
        return 'unknown'
    if len(fr) > MAXF:
        fr = [fr[i] for i in np.linspace(0, len(fr)-1, MAXF).astype(int)]
    votes = {'left': 0, 'right': 0}
    for fp in fr:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        v = sentido_frame(img)
        if v in votes:
            votes[v] += 1
    if votes['left'] == votes['right']:
        return 'unknown'
    return 'left' if votes['left'] > votes['right'] else 'right'


ndir = ncarry = 0
for v8dir, olddir in V8.items():
    base = PROJ/v8dir
    if not base.is_dir():
        continue
    print(f"\n=== {v8dir} ===")
    for sub in sorted(base.iterdir()):
        rj = next(sub.glob('*_resumen.json'), None)
        if not rj:
            continue
        meta = json.loads(rj.read_text())
        ind = sub.name
        # 1) barril_dir
        origen = meta.get('carpeta_origen')
        d = detect_dir(PROJ/origen if origen and not Path(origen).is_absolute() else origen) if origen else 'unknown'
        meta['barril_dir'] = d
        # 2) carryover girth/verija
        carried = ''
        if olddir:
            old = PROJ/olddir/ind/f'{ind}_resumen.json'
            if old.is_file():
                om = json.loads(old.read_text())
                for k in ('girth_frac_manual', 'verija_frac_manual'):
                    if om.get(k) is not None:
                        meta[k] = om[k]; carried += f' {k}={om[k]}'
                if carried:
                    ncarry += 1
        rj.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        ndir += 1
        print(f"  {ind:16} dir={d}{('  +copiado:'+carried) if carried else ''}")

print(f"\n[done] barril_dir seteado en {ndir} modelos | carryover girth/verija en {ncarry}")
