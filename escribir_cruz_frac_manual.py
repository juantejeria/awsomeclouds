"""
Escribe `cruz_frac` en el _resumen.json de cada modelo 3D usando los PUNTOS DE
CRUZ ANOTADOS A MANO en el editor de barril (frames_index.json), en vez de los
predichos por cruz_pose.pt.

Para cada individuo (de los datasets indicados):
  1. Toma todos sus frames con cruz manual en _barril_training/frames_index.json.
  2. Normaliza cada cruz contra el bbox de la mascara de barril (_mask.png =
     barril_seg), igual que detectar_cruz_modelos.py:
        cruz_xn = (cruz_x - barril_xmin) / ancho_barril
        cruz_yn = (cruz_y - barril_ymin) / alto_barril
  3. Toma la MEDIANA de cruz_xn / cruz_yn sobre sus frames (robusto al ruido).
  4. Convierte a cruz_frac con el barril_dir del modelo (misma convencion que
     escribir_cruz_frac.py):
        cruz_frac = (1 - cruz_xn) si barril_dir != 'left'  (frente en xMax)
        cruz_frac =      cruz_xn   si barril_dir == 'left'  (frente en xMin)
     y se recorta a [0, 0.5] (rango del visor).
  5. Actualiza SOLO cruz_frac / cruz_xn / cruz_yn / cruz_source / cruz_conf /
     cruz_n_manual en el _resumen.json. PRESERVA cualquier otro campo, en
     particular los manuales de la UI (cruz_frac_manual, girth_*, verija_*).

Hace backup .bak_manualcruz de cada _resumen.json antes de tocarlo.
Escritura atomica (.tmp + replace).

Uso:  python escribir_cruz_frac_manual.py            # 14mayo + 20mayo
      python escribir_cruz_frac_manual.py 14mayo     # solo uno
"""
import json
import os
import sys
import shutil
import statistics as st
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

PROJ = Path(__file__).parent
DATA_DIR = PROJ / 'output_modelos3d_grandes' / '_barril_training'
INDEX_FILE = DATA_DIR / 'frames_index.json'

# dataset (source en el index) -> carpeta de modelos 3d que sirve app.py
V8 = {
    '14mayo': 'output_modelos3d_live_14mayo_v8',
    '20mayo': 'output_modelos3d_live_20mayo_v8',
}


def resumen_path(dir_path):
    for f in sorted(os.listdir(dir_path)):
        if f.endswith('_resumen.json') or f == 'resumen.json':
            return os.path.join(dir_path, f)
    return None


def cruz_xn_yn(frame):
    """cruz normalizada contra el bbox de la mascara de barril del frame, o None."""
    m = cv2.imread(str(DATA_DIR / frame['mask']), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    ys, xs = (m > 127).nonzero()
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    cx, cy = float(frame['cruz']['x']), float(frame['cruz']['y'])
    return (cx - x0) / w, (cy - y0) / h


def main():
    datasets = sys.argv[1:] or list(V8.keys())
    for ds in datasets:
        if ds not in V8:
            print(f"[warn] dataset desconocido: {ds} (validos: {list(V8)})")
    datasets = [d for d in datasets if d in V8]

    frames = json.load(open(INDEX_FILE))
    if isinstance(frames, dict):
        frames = list(frames.values())

    # agrupar cruz_xn / cruz_yn manuales por (dataset, individuo)
    xn = defaultdict(list)
    yn = defaultdict(list)
    for f in frames:
        ds = f.get('source')
        if ds not in datasets:
            continue
        if not f.get('cruz'):
            continue
        r = cruz_xn_yn(f)
        if r is None:
            continue
        # individuo sin prefijo de dataset: '14mayo_100_137.5' -> '100_137.5'
        ind = f['individuo']
        ind = ind[len(ds) + 1:] if ind.startswith(ds + '_') else ind
        xn[(ds, ind)].append(r[0])
        yn[(ds, ind)].append(r[1])

    ok = skip = 0
    for (ds, ind), xs in sorted(xn.items()):
        dir_path = PROJ / V8[ds] / ind
        if not dir_path.is_dir():
            print(f"[skip] {ds}/{ind}: no existe carpeta de modelo"); skip += 1; continue
        rp = resumen_path(dir_path)
        if not rp:
            print(f"[skip] {ds}/{ind}: sin _resumen.json"); skip += 1; continue

        meta = json.load(open(rp))
        cxn = round(st.median(xs), 4)
        cyn = round(st.median(yn[(ds, ind)]), 4)
        facing_right = (meta.get('barril_dir') != 'left')
        cruz_frac = (1.0 - cxn) if facing_right else cxn
        cruz_frac = round(max(0.0, min(0.5, cruz_frac)), 4)

        # backup una sola vez
        bak = rp + '.bak_manualcruz'
        if not os.path.exists(bak):
            shutil.copy2(rp, bak)

        meta['cruz_frac'] = cruz_frac
        meta['cruz_xn'] = cxn
        meta['cruz_yn'] = cyn
        meta['cruz_source'] = 'manual_train'
        meta['cruz_conf'] = 1.0
        meta['cruz_n_manual'] = len(xs)

        tmp = rp + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp, rp)
        ok += 1
        tag = '' if facing_right else '  (mira IZQ)'
        print(f"  {ds}/{ind:14s}  n={len(xs):2d}  cruz_xn={cxn:.3f} -> cruz_frac={cruz_frac:.3f}"
              f"  [barril_dir={meta.get('barril_dir')}]{tag}"
              f"{'  (conserva cruz_frac_manual)' if meta.get('cruz_frac_manual') is not None else ''}")

    print(f"\n[done] escritos {ok}, saltados {skip}")


if __name__ == '__main__':
    main()
