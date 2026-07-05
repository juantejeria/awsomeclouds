"""Detector de verija por 'cambio de paso' del fondo (z=0) RESTRINGIDO a una
ventana [w0,w1] medida desde el FRENTE. Compara varias ventanas y detectores
contra los labels guardados, y reporta el error.
"""
import json
from pathlib import Path
import numpy as np

PROJ = Path(__file__).parent
DATASETS = ['output_modelos3d_live_14mayo', 'output_modelos3d_live_20mayo']
NB = 200


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


# cargar labels + geometría
items = []
for ds in DATASETS:
    base = PROJ / ds
    if not base.is_dir():
        continue
    for d in sorted(base.iterdir()):
        rj = next(d.glob('*_resumen.json'), None)
        if not rj:
            continue
        meta = json.loads(rj.read_text())
        frac_r = meta.get('verija_frac_manual')
        if frac_r is None:
            continue
        ply = next(d.glob('*_3d.ply'), None)
        if not ply:
            continue
        verts = leer_ply(ply)
        xs = verts[:, 0]; xmin, xmax = xs.min(), xs.max(); L = xmax - xmin
        facing_right = (meta.get('barril_dir') != 'left')
        xfront = xmax if facing_right else xmin
        xrear = xmin if facing_right else xmax
        xlabel = xrear + frac_r * L if facing_right else xrear - frac_r * L
        xc, ybot = fondo_z0(verts)
        if len(xc) < 30:
            continue
        items.append(dict(ind=d.name, frac_r=frac_r, L=L, fr=facing_right,
                          xfront=xfront, xrear=xrear, xlabel=xlabel, xc=xc, ybot=ybot))


def x_at_front_frac(it, f):
    return it['xfront'] - f * it['L'] if it['fr'] else it['xfront'] + f * it['L']


def detect(it, w0, w1, modo='maxstep'):
    """Devuelve x del paso detectado dentro de [w0,w1] desde el frente."""
    xc, ybot = it['xc'], it['ybot']
    xa, xb = x_at_front_frac(it, w0), x_at_front_frac(it, w1)
    lo, hi = min(xa, xb), max(xa, xb)
    m = (xc >= lo) & (xc <= hi)
    idx = np.where(m)[0]
    if len(idx) < 3:
        return None
    # paso hacia abajo = el fondo CAE al avanzar hacia el fondo (empiezan las patas)
    # recorremos en sentido frente->fondo
    order = idx[::-1] if it['fr'] else idx          # fr: fondo=xMin => x creciente es hacia frente; recorrer decreciente
    jx = 0.5 * (xc[:-1] + xc[1:])
    djump = ybot[1:] - ybot[:-1]                     # signo
    aj = np.abs(djump)
    cand = [k for k in range(len(jx)) if lo <= jx[k] <= hi]
    if not cand:
        return None
    if modo == 'maxstep':
        kk = cand[int(np.argmax(aj[cand]))]
    elif modo == 'maxdrop':  # mayor caida del fondo hacia el fondo
        drop = (ybot[:-1] - ybot[1:]) if it['fr'] else (ybot[1:] - ybot[:-1])
        kk = cand[int(np.argmax(drop[cand]))]
    else:  # medstep: escalon de NIVEL (mediana lado-frente menos mediana lado-fondo)
        W = 8
        best, kk = -1e9, cand[0]
        for k in cand:
            if it['fr']:   # fondo a la izq (x menor): frente = x mayor (k+1..), fondo = x menor (..k)
                front = ybot[k + 1:k + 1 + W]; rearside = ybot[max(0, k - W):k]
            else:          # fondo a la der (x mayor)
                front = ybot[max(0, k - W):k]; rearside = ybot[k + 1:k + 1 + W]
            if len(front) < 3 or len(rearside) < 3:
                continue
            step = float(np.median(front) - np.median(rearside))  # baja al ir al fondo
            if step > best:
                best, kk = step, k
    return jx[kk]


def eval_win(w0, w1, modo):
    errs = []
    for it in items:
        xp = detect(it, w0, w1, modo)
        if xp is None:
            continue
        fr_pred = abs(xp - it['xrear']) / it['L']
        errs.append(abs(fr_pred - it['frac_r']) * it['L'])
    errs = np.array(errs)
    return errs.mean(), errs.max(), len(errs)


print(f"\n=== {len(items)} labels · detector por escalón del fondo, por ventana (desde frente) ===")
print(f"{'ventana':16}{'modo':10}{'err_medio':>11}{'err_max':>9}{'n':>4}")
for (w0, w1) in [(0.50, 1.00), (0.70, 0.90), (0.62, 0.82), (0.60, 0.85), (0.65, 0.88)]:
    for modo in ('maxstep', 'maxdrop', 'medstep'):
        em, ex, n = eval_win(w0, w1, modo)
        print(f"{f'{int(w0*100)}-{int(w1*100)}%':16}{modo:10}{em:>9.1f}cm{ex:>7.1f}cm{n:>4}")
print("\n(referencia: regla constante 29% desde el fondo -> err_medio 5.4cm)")
