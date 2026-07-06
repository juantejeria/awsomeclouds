"""¿La verija que marcaste cae sobre un CAMBIO BRUSCO del contorno inferior
(z=0) donde empiezan las patas? Para cada label: extrae el fondo ybot(x),
mide el salto |Δy| entre nodos, y comprueba si el label coincide con un salto
grande. Prueba un detector: 'mayor salto del fondo en la mitad trasera'.
"""
import json
from pathlib import Path
import numpy as np

PROJ = Path(__file__).parent
DATASETS = ['output_modelos3d_live_14mayo', 'output_modelos3d_live_20mayo']
NB = 160


def leer_ply(path):
    verts = []
    with open(path) as f:
        nv = 0
        while True:
            ln = f.readline()
            if ln.startswith('element vertex'):
                nv = int(ln.split()[-1])
            elif ln.startswith('end_header'):
                break
        for _ in range(nv):
            p = f.readline().split()
            verts.append((float(p[0]), float(p[1]), float(p[2])))
    return np.array(verts)


def fondo_z0(verts, nbins=NB):
    z = verts[:, 2]; zeps = 0.05 * (z.max() - z.min() + 1e-6)
    pts = verts[np.abs(z) <= max(zeps, 1e-6)]
    if len(pts) < 30:
        pts = verts
    x, y = pts[:, 0], pts[:, 1]
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    xc, ybot = [], []
    for i in range(nbins):
        sel = (x >= edges[i]) & ((x < edges[i + 1]) if i < nbins - 1 else (x <= edges[i + 1]))
        if sel.sum():
            xc.append(0.5 * (edges[i] + edges[i + 1])); ybot.append(y[sel].min())
    return np.array(xc), np.array(ybot)


rows = []
det_err = []
for ds in DATASETS:
    base = PROJ / ds
    if not base.is_dir():
        continue
    for d in sorted(base.iterdir()):
        rj = next(d.glob('*_resumen.json'), None)
        if not rj:
            continue
        meta = json.loads(rj.read_text())
        frac = meta.get('verija_frac_manual')
        if frac is None:
            continue
        ply = next(d.glob('*_3d.ply'), None)
        if not ply:
            continue
        verts = leer_ply(ply)
        xs = verts[:, 0]; xmin, xmax = xs.min(), xs.max(); L = xmax - xmin
        facing_right = (meta.get('barril_dir') != 'left')
        xrear = xmin if facing_right else xmax
        xver = xrear + frac * L if facing_right else xrear - frac * L
        xc, ybot = fondo_z0(verts)
        if len(xc) < 20:
            continue
        jump = np.abs(np.diff(ybot))                       # salto entre nodos
        jx = 0.5 * (xc[:-1] + xc[1:])
        # bin del label
        kv = int(np.argmin(np.abs(jx - xver)))
        jmax = jump.max() if jump.max() > 0 else 1.0
        jump_norm_en_label = jump[kv] / jmax              # 1.0 = el salto mas grande del animal
        rank = int((jump > jump[kv]).sum()) + 1            # 1 = es el mayor salto
        # DETECTOR: mayor salto del fondo en la mitad trasera (mid->fondo)
        mid = 0.5 * (xmin + xmax)
        rear = (jx <= mid) if facing_right else (jx >= mid)
        idx = np.where(rear)[0]
        kd = idx[int(np.argmax(jump[idx]))]
        xpred = jx[kd]
        frac_pred = abs(xpred - xrear) / L
        det_err.append(abs(frac_pred - frac) * L)
        rows.append({'ind': d.name, 'frac': frac, 'jrank': rank,
                     'jrel': jump_norm_en_label, 'frac_pred': frac_pred})

if not rows:
    print("sin labels"); raise SystemExit

jrel = np.array([r['jrel'] for r in rows])
jrank = np.array([r['jrank'] for r in rows])
de = np.array(det_err)
print(f"\n=== {len(rows)} verijas: ¿el label cae sobre un salto del fondo? ===")
print(f"  salto en el label (relativo al mayor del animal): media={jrel.mean():.2f}  "
      f"min={jrel.min():.2f}")
print(f"  ranking del salto en el label (1=el mayor): media={jrank.mean():.1f}  "
      f"es-top3 en {100*(jrank<=3).mean():.0f}% de los casos")
print(f"\n=== Detector 'mayor salto del fondo en mitad trasera' vs tus labels ===")
print(f"  error |Δx|: media={de.mean():.1f}cm  max={de.max():.1f}cm")
print(f"  (comparar con regla constante 29%: media 5.4cm)")
print(f"\n{'ind':14}{'frac_lbl':>9}{'frac_pred':>10}{'salto_rel':>10}{'rank':>6}")
for r in sorted(rows, key=lambda r: r['frac']):
    print(f"{r['ind']:14}{r['frac']:>9.2f}{r['frac_pred']:>10.2f}{r['jrel']:>10.2f}{r['jrank']:>6}")
