"""
CSV de 14mayo + 20mayo con las columnas pedidas, calculadas sobre los modelos 3D
v8 usando el cruz_frac MANUAL (cruz_source='manual_train') y verija_frac_manual.

Columnas:
  nombre
  altura_calc_cm          -> meta['altura_real_cm'] (= altura calculada del
                             pipeline; mismo criterio que tabla_corte.csv de ayer)
  diam_verija_cm          -> perímetro real de la sección en el plano de la verija
  diam_cruz_cm            -> perímetro real de la sección en el plano de la cruz
  distancia_cm            -> separación entre el plano de la cruz y el de la verija
  vol_entre_diam_piso_L   -> volumen ENTRE los dos diámetros, con PISO MÍNIMO en
                             el yMin de la verija (clip_above), igual que el visor
  vol_s2_s3_sin_piso_L    -> S2+S3 (2 cuartos centrales del tramo cruz->verija),
                             SIN piso (sección transversal completa, incluye panza)

Reusa el algoritmo de tabla_volumen_corte.py (mismas funciones de corte/clip).
Salida: output_cruz_modelos/tabla_diametros_mayo.csv
Uso:  python tabla_diametros_mayo.py
"""
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import trimesh

from tabla_volumen_corte import (
    slice_section, section_info, clipped_volume_liters,
)

PROJ = Path(__file__).parent


def perim_visor(V, F, xp):
    """Perímetro de la sección IGUAL que el visor 3D (viewer3d.js): suma la
    longitud de cada segmento crudo del corte malla-plano (cada triángulo que
    cruza aporta un segmento). Fiel al contorno real; NO reordena por ángulo.
    """
    import math
    per = 0.0
    for tri in V[F]:
        hits = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            p, q = tri[a], tri[b]
            dp, dq = p[0] - xp, q[0] - xp
            if (dp < 0) != (dq < 0):
                t = dp / (dp - dq)
                hits.append((p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t))
        if len(hits) == 2:
            per += math.hypot(hits[0][0] - hits[1][0], hits[0][1] - hits[1][1])
    return per
V8 = {
    '14mayo': 'output_modelos3d_live_14mayo_v8',
    '20mayo': 'output_modelos3d_live_20mayo_v8',
}


def main():
    rows = []
    for dataset, d in V8.items():
        for ind_dir in sorted(glob.glob(str(PROJ / d / '*'))):
            if not os.path.isdir(ind_dir):
                continue
            ply = glob.glob(os.path.join(ind_dir, '*_3d.ply'))
            rj = glob.glob(os.path.join(ind_dir, '*_resumen.json'))
            if not ply or not rj:
                continue
            meta = json.load(open(rj[0]))
            mesh = trimesh.load(ply[0], process=False)
            V = np.asarray(mesh.vertices, float)
            F = np.asarray(mesh.faces, int)
            if len(F) == 0:
                continue

            x = V[:, 0]; xmin, xmax = x.min(), x.max(); L = xmax - xmin
            facing_right = (meta.get('barril_dir') != 'left')
            xfront = xmax if facing_right else xmin
            xrear = xmin if facing_right else xmax

            # cruz: preferir manual de la UI; si no, el cruz_frac (manual_train)
            cf = meta.get('cruz_frac_manual')
            if cf is None:
                cf = meta.get('cruz_frac', 0.20)
            vf = meta.get('verija_frac_manual', 0.25)
            cf = min(max(float(cf), 0.0), 0.5); vf = min(max(float(vf), 0.0), 0.5)
            x_cruz = xfront - cf * L if facing_right else xfront + cf * L
            x_ver = xrear + vf * L if facing_right else xrear - vf * L

            sc = slice_section(V, F, x_cruz)
            sv = slice_section(V, F, x_ver)
            if sc is None or sv is None:
                print(f"[warn] {dataset}/{os.path.basename(ind_dir)}: sección vacía"); continue
            ic = section_info(sc); iv = section_info(sv)

            xlo, xhi = min(x_cruz, x_ver), max(x_cruz, x_ver)
            dist = xhi - xlo

            # (6) volumen ENTRE los diámetros con piso mínimo de la verija
            yfloor = iv['ymin']
            vol_piso = clipped_volume_liters(V, F, xlo, xhi, yfloor)

            # (7) S2+S3 sin piso: 4 cuartos del tramo cruz->verija, completos
            yfull = V[:, 1].min() - 1.0
            step = (x_ver - x_cruz) / 4.0
            secs = []
            for i in range(4):
                a = x_cruz + i * step
                b = x_cruz + (i + 1) * step
                secs.append(clipped_volume_liters(V, F, min(a, b), max(a, b), yfull, 16))
            vol_s2s3 = secs[1] + secs[2]

            rows.append({
                '_dataset': dataset,
                'nombre': os.path.basename(ind_dir),
                'altura_calc_cm': meta.get('altura_real_cm'),
                'diam_verija_cm': round(perim_visor(V, F, x_ver), 1),   # = visor
                'diam_cruz_cm': round(perim_visor(V, F, x_cruz), 1),    # = visor
                'distancia_cm': round(dist, 1),
                'vol_entre_diam_piso_L': round(vol_piso, 1),
                'vol_s2_s3_sin_piso_L': round(vol_s2s3, 1),
            })
            r = rows[-1]
            print(f"  {dataset:7} {r['nombre']:12} alt={r['altura_calc_cm']}  "
                  f"diam(ver/cruz)={r['diam_verija_cm']:.0f}/{r['diam_cruz_cm']:.0f}cm  "
                  f"dist={r['distancia_cm']:.1f}  vol_piso={r['vol_entre_diam_piso_L']:.1f}L  "
                  f"S2+S3={r['vol_s2_s3_sin_piso_L']:.1f}L")

    rows.sort(key=lambda r: (r['_dataset'], r['nombre']))
    cols = ['nombre', 'altura_calc_cm', 'diam_verija_cm', 'diam_cruz_cm',
            'distancia_cm', 'vol_entre_diam_piso_L', 'vol_s2_s3_sin_piso_L']
    out = PROJ / 'output_cruz_modelos' / 'tabla_diametros_mayo.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print(f"\n[done] {len(rows)} individuos -> {out}")


if __name__ == '__main__':
    main()
