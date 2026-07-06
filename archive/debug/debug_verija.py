"""Diagnóstico para la VERIJA: sobre el plano z=0 del PLY de barril, extrae el
contorno inferior (panza) y busca el 'cambio de nivel' en la MITAD TRASERA
(donde empiezan las patas). Marca girth (frente) y verija (trasero) y guarda un
plot por individuo.

Sentido: barril_dir del resumen ('left' => frente en xMin => trasero en xMax).
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = Path(__file__).parent


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


def contorno_z0(verts, nbins=120, zeps=None):
    z = verts[:, 2]
    if zeps is None:
        zeps = 0.05 * (z.max() - z.min() + 1e-6)
    m = np.abs(z) <= max(zeps, 1e-6)
    pts = verts[m] if m.sum() > 30 else verts
    x = pts[:, 0]; y = pts[:, 1]
    xmin, xmax = x.min(), x.max()
    edges = np.linspace(xmin, xmax, nbins + 1)
    xc, ybot, ytop = [], [], []
    for i in range(nbins):
        sel = (x >= edges[i]) & (x < edges[i + 1] if i < nbins - 1 else x <= edges[i + 1])
        if sel.sum() == 0:
            continue
        xc.append(0.5 * (edges[i] + edges[i + 1]))
        ybot.append(y[sel].min()); ytop.append(y[sel].max())
    return np.array(xc), np.array(ybot), np.array(ytop)


def run(model_dir, out):
    ply = next(Path(model_dir).glob('*_3d.ply'), None)
    if ply is None:
        print('sin ply', model_dir); return
    meta = {}
    rj = next(Path(model_dir).glob('*_resumen.json'), None)
    if rj:
        meta = json.loads(rj.read_text())
    bdir = meta.get('barril_dir', 'unknown')
    facing_right = (bdir != 'left')
    verts = leer_ply(ply)
    xc, ybot, ytop = contorno_z0(verts)
    if len(xc) < 10:
        print('contorno corto', model_dir); return
    L = xc[-1] - xc[0]
    # suavizar fondo (solo para dibujar / baseline)
    k = 7; pad = k // 2
    ybs = np.convolve(np.pad(ybot, pad, mode='edge'), np.ones(k) / k, mode='valid')
    mid = 0.5 * (xc[0] + xc[-1])
    # baseline de panza = mediana del fondo en el centro (zona lisa)
    central = np.abs(xc - mid) < 0.20 * L
    baseline = float(np.median(ybs[central])) if central.any() else float(np.median(ybs))
    bodyH = float(np.median(ytop)) - baseline
    delta = 0.22 * bodyH                       # un pico de pata cae > delta bajo la panza
    # escanear desde el MEDIO hacia el FONDO; verija = primer pico de pata (cambio de nivel)
    order = np.where(xc >= mid)[0] if not facing_right else np.where(xc <= mid)[0][::-1]
    # rear = opuesto a la cabeza: head right(facing_right)=> rear=xMin => recorrer mid->izq
    if facing_right:
        order = [i for i in range(len(xc)) if xc[i] <= mid][::-1]   # mid -> xMin
    else:
        order = [i for i in range(len(xc)) if xc[i] >= mid]          # mid -> xMax
    vk = order[-1] if order else 0
    for i in order:
        if ybot[i] < baseline - delta:
            vk = i
            break
    # girth (frente) referencia 20%
    xfront = xc[-1] if facing_right else xc[0]
    xg = xfront - 0.20 * L if facing_right else xfront + 0.20 * L
    gk = int(np.argmin(np.abs(xc - xg)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xc, ytop, '-', color='peru', lw=1, label='top (lomo)')
    ax.plot(xc, ybot, '-', color='gray', lw=1, label='fondo (panza)')
    ax.plot(xc, ybs, '-', color='black', lw=1.5, alpha=0.6, label='fondo suav.')
    ax.axvspan(xc[0] if facing_right else mid, mid if facing_right else xc[-1],
               color='blue', alpha=0.05, label='mitad trasera')
    ax.axvline(xc[gk], color='gold', lw=2, label='girth 20% (frente)')
    ax.axvline(xc[vk], color='magenta', lw=2, label='VERIJA (cambio nivel)')
    ax.set_title(f"{Path(model_dir).name}  | barril_dir={bdir}  | frente={'der' if facing_right else 'izq'}")
    ax.legend(fontsize=7, loc='upper right'); ax.set_aspect('equal')
    fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)
    print(f'escrito {out}  verija_x={xc[vk]:.1f} (frac trasero={abs(xc[vk]-xfront)/L:.2f})')


if __name__ == '__main__':
    casos = [('output_modelos3d_live_14mayo/113_214', 'debug_verija_113_214.png'),
             ('output_modelos3d_live_20mayo/127_435', 'debug_verija_127_435.png'),
             ('output_modelos3d_live_14mayo/110_221', 'debug_verija_110_221.png')]
    for d, o in casos:
        run(PROJ / d, str(PROJ / o))
