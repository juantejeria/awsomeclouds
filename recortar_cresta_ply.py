"""
Recorta la cresta del LOMO en los *_3d.ply existentes (datasets _v8 del visor).

No destructivo respecto a los datos guardados (cruz/anca/verija viven en los
*_resumen.json, que NO se tocan). Solo baja la punta superior de la malla usando
crest_trim_mesh.trim_top_crest. Reescribe el PLY ASCII preservando colores, caras,
comentarios y el orden/cantidad de vértices (solo cambia la columna Y de la punta).

BACKUP: hacer copia de los .ply ANTES de correr (ya existe en backups_ply/...).
Uso:  python recortar_cresta_ply.py            # aplica a todos
      python recortar_cresta_ply.py --dry      # solo reporta, no escribe
"""
import glob
import os
import sys
import numpy as np
from crest_trim_mesh import trim_top_crest

DATASETS = ['output_modelos3d_live_6mayo_v8', 'output_modelos3d_live_14mayo_v8',
            'output_modelos3d_live_20mayo_v8', 'output_modelos3d_live_12junio_v8']


def process(path, dry=False):
    raw = open(path).read()
    lines = raw.split('\n')
    iend = next(i for i, l in enumerate(lines) if l.strip() == 'end_header')
    header = lines[:iend + 1]
    nv = int(next(l for l in header if l.startswith('element vertex')).split()[2])
    nf = int(next(l for l in header if l.startswith('element face')).split()[2])
    body = lines[iend + 1:]
    vlines = body[:nv]
    flines = body[nv:nv + nf]

    xyz = np.zeros((nv, 3))
    rest = []
    for i, l in enumerate(vlines):
        p = l.split()
        xyz[i] = [float(p[0]), float(p[1]), float(p[2])]
        rest.append(p[3:])  # color u otras props, verbatim

    V2 = trim_top_crest(xyz)
    moved = int((xyz[:, 1] > V2[:, 1] + 1e-6).sum())
    ymax0, ymax1 = xyz[:, 1].max(), V2[:, 1].max()

    if not dry:
        out_v = []
        for i in range(nv):
            cols = (' ' + ' '.join(rest[i])) if rest[i] else ''
            out_v.append(f"{xyz[i,0]:.2f} {V2[i,1]:.2f} {xyz[i,2]:.2f}{cols}")
        new = '\n'.join(header + out_v + flines)
        if not new.endswith('\n'):
            new += '\n'
        open(path, 'w').write(new)
    return ymax0, ymax1, moved, nv


def main():
    dry = '--dry' in sys.argv
    rows = []
    for d in DATASETS:
        for ind in sorted(glob.glob(d + '/*')):
            if not os.path.isdir(ind):
                continue
            for ply in glob.glob(os.path.join(ind, '*_3d.ply')):
                y0, y1, moved, nv = process(ply, dry)
                rows.append((os.path.relpath(ply), y0, y1, moved, nv))
                print(f"  {os.path.basename(os.path.dirname(ply)):16} "
                      f"yMax {y0:6.1f} -> {y1:6.1f}  (-{y0-y1:4.1f})  movidos {moved:3}/{nv}")
    drop = np.mean([y0 - y1 for _, y0, y1, _, _ in rows]) if rows else 0
    print(f"\n[{'DRY ' if dry else ''}done] {len(rows)} PLYs  | recorte medio de yMax = {drop:.1f} cm")


if __name__ == '__main__':
    main()
