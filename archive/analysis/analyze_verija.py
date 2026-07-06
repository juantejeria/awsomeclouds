"""Analiza las verijas guardadas para derivar una regla automática.
Compara: (a) verija_frac constante, (b) distancia absoluta desde el fondo,
(c) si la altura de la sección en la verija es un % consistente del alto máximo
(firma geométrica detectable). Reporta cuál reproduce mejor tus etiquetas.
"""
import json
from pathlib import Path
import numpy as np

PROJ = Path(__file__).parent
DATASETS = ['output_modelos3d_live_14mayo', 'output_modelos3d_live_20mayo']


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


def contorno_z0(verts, nbins=120):
    z = verts[:, 2]; zeps = 0.05 * (z.max() - z.min() + 1e-6)
    pts = verts[np.abs(z) <= max(zeps, 1e-6)]
    if len(pts) < 30:
        pts = verts
    x, y = pts[:, 0], pts[:, 1]
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    xc, ytop, ybot = [], [], []
    for i in range(nbins):
        sel = (x >= edges[i]) & ((x < edges[i + 1]) if i < nbins - 1 else (x <= edges[i + 1]))
        if sel.sum():
            xc.append(0.5 * (edges[i] + edges[i + 1]))
            ytop.append(y[sel].max()); ybot.append(y[sel].min())
    return np.array(xc), np.array(ytop), np.array(ybot)


rows = []
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
        # altura de seccion (top-base) en z=0; uso TOP (lomo, limpio) sobre base 0
        xc, ytop, ybot = contorno_z0(verts)
        k = int(np.argmin(np.abs(xc - xver)))
        alto_ver = ytop[k]
        alto_max = np.percentile(ytop, 95)
        rows.append({'ind': d.name, 'ds': ds[-6:], 'frac': float(frac),
                     'L': L, 'dist_fondo_cm': frac * L,
                     'alto_ver': alto_ver, 'ratio_alto': alto_ver / alto_max})

if not rows:
    print("no hay verijas guardadas"); raise SystemExit

fr = np.array([r['frac'] for r in rows])
dist = np.array([r['dist_fondo_cm'] for r in rows])
ratio = np.array([r['ratio_alto'] for r in rows])
L = np.array([r['L'] for r in rows])


def stats(name, v, unit=''):
    print(f"  {name:24} media={v.mean():.3f}{unit}  std={v.std():.3f}  "
          f"min={v.min():.3f}  max={v.max():.3f}  CV={100*v.std()/abs(v.mean()):.1f}%")


print(f"\n=== {len(rows)} verijas guardadas ===")
stats('verija_frac (desde fondo)', fr)
stats('dist absoluta fondo (cm)', dist, 'cm')
stats('ratio alto seccion/max', ratio)
print(f"\n  corr(frac, L) = {np.corrcoef(fr, L)[0,1]:+.2f}  (¿la fracción depende del tamaño?)")

# Error de la regla "frac constante = media"
pred = fr.mean()
err_frac = np.abs(fr - pred) * L  # error en cm sobre cada animal
print(f"\n=== Regla A: frac constante = {pred:.3f} ===")
print(f"  error |Δx|: media={err_frac.mean():.1f}cm  max={err_frac.max():.1f}cm "
      f"(= {100*np.abs(fr-pred).mean():.1f}% del largo en promedio)")

# Regla B: distancia constante desde el fondo
predd = dist.mean()
err_d = np.abs(dist - predd)
print(f"\n=== Regla B: distancia constante = {predd:.1f}cm desde el fondo ===")
print(f"  error |Δx|: media={err_d.mean():.1f}cm  max={err_d.max():.1f}cm")

print("\n=== detalle ===")
print(f"{'ind':14}{'ds':8}{'frac':>6}{'dist_cm':>9}{'ratio_alto':>11}")
for r in sorted(rows, key=lambda r: r['frac']):
    print(f"{r['ind']:14}{r['ds']:8}{r['frac']:>6.2f}{r['dist_fondo_cm']:>9.1f}{r['ratio_alto']:>11.2f}")
