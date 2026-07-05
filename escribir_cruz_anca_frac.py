"""
Deriva cruz_frac (desde el FRENTE) y anca_frac (desde el FONDO) de cada modelo 3D
a partir de los PUNTOS anotados a mano (cruz + anca) en los frames VALIDADOS del
editor de barril (frames_index.json), y los escribe en el _resumen.json.

Por cada individuo (datasets 6mayo/12junio/14mayo/20mayo):
  1. Toma sus frames con status='validated' que tengan cruz y anca.
  2. Normaliza cada punto contra el bbox de la máscara de barril (_mask.png):
        cruz_xn = (cruz_x - xmin)/ancho ;  anca_xn = (anca_x - xmin)/ancho
  3. Mediana sobre los frames -> cruz_xn, anca_xn robustos.
  4. Convierte con barril_dir (misma convención que el visor):
        cruz_frac (desde frente) = (1-cruz_xn) si dir!='left' else cruz_xn
        anca_frac (desde fondo)  =    anca_xn   si dir!='left' else (1-anca_xn)
     clamp a [0, 0.5].
  5. Actualiza SOLO cruz_frac/anca_frac (+ *_xn, *_source, *_n) en el resumen.
     Preserva cruz_frac_manual, verija_frac_manual, girth_*, etc.

Backup .bak_cruzanca por resumen. Escritura atómica.
Uso:  python escribir_cruz_anca_frac.py
"""
import json
import os
import glob
import shutil
import statistics as st
from pathlib import Path
from collections import defaultdict

import cv2

PROJ = Path(__file__).parent
DATA_DIR = PROJ / 'output_modelos3d_grandes' / '_barril_training'
INDEX_FILE = DATA_DIR / 'frames_index.json'

V8 = {
    '6mayo':   'output_modelos3d_live_6mayo_v8',
    '12junio': 'output_modelos3d_live_12junio_v8',
    '14mayo':  'output_modelos3d_live_14mayo_v8',
    '20mayo':  'output_modelos3d_live_20mayo_v8',
}


def resumen_path(dir_path):
    for f in sorted(os.listdir(dir_path)):
        if f.endswith('_resumen.json') or f == 'resumen.json':
            return os.path.join(dir_path, f)
    return None


def xn_of(frame, key):
    """x normalizada del punto `key` contra el bbox de la máscara de barril."""
    pt = frame.get(key)
    if not pt:
        return None
    m = cv2.imread(str(DATA_DIR / frame['mask']), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    ys, xs = (m > 127).nonzero()
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    w = max(1, x1 - x0)
    return (float(pt['x']) - x0) / w


def main():
    frames = json.load(open(INDEX_FILE))

    # agrupar xn de cruz/anca por (dataset, individuo) sobre frames VALIDADOS
    cruz_xn = defaultdict(list)
    anca_xn = defaultdict(list)
    for f in frames:
        ds = f.get('source')
        if ds not in V8 or f.get('status') != 'validated':
            continue
        ind = f['individuo']
        ind = ind[len(ds) + 1:] if ind.startswith(ds + '_') else ind
        cx = xn_of(f, 'cruz')
        ax = xn_of(f, 'anca')
        if cx is not None:
            cruz_xn[(ds, ind)].append(cx)
        if ax is not None:
            anca_xn[(ds, ind)].append(ax)

    keys = sorted(set(cruz_xn) | set(anca_xn))
    ok = skip = 0
    for (ds, ind) in keys:
        cxs = cruz_xn.get((ds, ind), [])
        axs = anca_xn.get((ds, ind), [])
        if not cxs or not axs:
            print(f"[skip] {ds}/{ind}: faltan puntos (cruz={len(cxs)} anca={len(axs)})"); skip += 1; continue
        dir_path = PROJ / V8[ds] / ind
        rp = resumen_path(dir_path) if dir_path.is_dir() else None
        if not rp:
            print(f"[skip] {ds}/{ind}: sin resumen"); skip += 1; continue
        meta = json.load(open(rp))
        facing_right = (meta.get('barril_dir') != 'left')

        cxn = round(st.median(cxs), 4)
        axn = round(st.median(axs), 4)
        cruz_frac = (1.0 - cxn) if facing_right else cxn          # desde el FRENTE
        anca_frac = axn if facing_right else (1.0 - axn)          # desde el FONDO
        cruz_frac = round(max(0.0, min(0.5, cruz_frac)), 4)
        anca_frac = round(max(0.0, min(0.5, anca_frac)), 4)

        bak = rp + '.bak_cruzanca'
        if not os.path.exists(bak):
            shutil.copy2(rp, bak)

        meta['cruz_frac'] = cruz_frac
        meta['cruz_xn'] = cxn
        meta['cruz_source'] = 'manual_train'
        meta['cruz_n_manual'] = len(cxs)
        meta['anca_frac'] = anca_frac
        meta['anca_xn'] = axn
        meta['anca_source'] = 'manual_train'
        meta['anca_n_manual'] = len(axs)

        tmp = rp + '.tmp'
        with open(tmp, 'w') as fo:
            json.dump(meta, fo, ensure_ascii=False, indent=2)
        os.replace(tmp, rp)
        ok += 1
        print(f"  {ds:8}/{ind:14} n_cruz={len(cxs):2d} n_anca={len(axs):2d}  "
              f"cruz_frac={cruz_frac:.3f}  anca_frac={anca_frac:.3f}  [dir={meta.get('barril_dir')}]")

    print(f"\n[done] escritos {ok}, saltados {skip}")


if __name__ == '__main__':
    main()
