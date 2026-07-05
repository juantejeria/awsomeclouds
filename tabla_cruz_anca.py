"""
Tabla por individuo (4 datasets) de los diámetros CRUZ y ANCA (anotados a mano) y
sus volúmenes, calculada sobre las mallas 3D v8. Solo individuos con cruz_frac y
anca_frac en el resumen (los anotados).

Columnas:
  nombre
  alt_calc_cm                = altura_real_cm del resumen
  diam_anca_cm               = perímetro (método visor) de la sección en la anca
  diam_cruz_cm               = perímetro (método visor) de la sección en la cruz
  distancia_cm               = separación entre el plano de la cruz y el de la anca
  distancia_x1.5_cm          = distancia * 1.5
  vol_piso_verija_L          = volumen cruz↔anca con piso en el yMin de la verija
  vol_piso_meddist_L         = volumen cruz↔anca con piso a 0.5*distancia hacia abajo
                               desde la línea superior (topline) entre cruz y anca
  vol_sin_piso_central_L     = volumen sin piso del tramo central (punto medio ±0.25*dist)

Salida: output_cruz_modelos/tabla_cruz_anca.csv
Uso:  python tabla_cruz_anca.py
"""
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import trimesh

from tabla_volumen_corte import slice_section, section_info, clipped_volume_liters
from tabla_diametros_mayo import perim_visor

PROJ = Path(__file__).parent
MIN_VAL_FRAMES = 10   # mínimo de frames validados (cruz y anca) para entrar a la tabla
V8 = {
    '6mayo':   'output_modelos3d_live_6mayo_v8',
    '12junio': 'output_modelos3d_live_12junio_v8',
    '14mayo':  'output_modelos3d_live_14mayo_v8',
    '20mayo':  'output_modelos3d_live_20mayo_v8',
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
            if meta.get('cruz_frac') is None or meta.get('anca_frac') is None:
                continue  # solo los anotados (cruz+anca)
            # exigir >= MIN_VAL_FRAMES frames validados (cruz y anca)
            n_cz = meta.get('cruz_n_manual') or 0
            n_an = meta.get('anca_n_manual') or 0
            if min(n_cz, n_an) < MIN_VAL_FRAMES:
                print(f"[skip] {dataset}/{os.path.basename(ind_dir)}: solo {min(n_cz, n_an)} frames validados (<{MIN_VAL_FRAMES})")
                continue
            mesh = trimesh.load(ply[0], process=False)
            V = np.asarray(mesh.vertices, float)
            F = np.asarray(mesh.faces, int)
            if len(F) == 0:
                continue

            x = V[:, 0]; xmin, xmax = x.min(), x.max(); L = xmax - xmin
            facing_right = (meta.get('barril_dir') != 'left')
            xfront = xmax if facing_right else xmin
            xrear = xmin if facing_right else xmax

            # MISMA precedencia que el visor (viewer3d.js): manual > auto.
            cf = float(meta['cruz_frac_manual'] if meta.get('cruz_frac_manual') is not None else meta['cruz_frac'])
            af = float(meta['anca_frac_manual'] if meta.get('anca_frac_manual') is not None else meta['anca_frac'])
            vf = float(meta['verija_frac_manual'] if meta.get('verija_frac_manual') is not None else 0.25)

            x_cruz = xfront - cf * L if facing_right else xfront + cf * L
            x_anca = xrear + af * L if facing_right else xrear - af * L
            x_ver = xrear + vf * L if facing_right else xrear - vf * L

            sc = slice_section(V, F, x_cruz)
            sa = slice_section(V, F, x_anca)
            sv = slice_section(V, F, x_ver)
            if sc is None or sa is None:
                print(f"[warn] {dataset}/{os.path.basename(ind_dir)}: sección cruz/anca vacía"); continue
            ic = section_info(sc); ia = section_info(sa)

            xlo, xhi = min(x_cruz, x_anca), max(x_cruz, x_anca)
            dist = xhi - xlo

            # (7) piso en el yMin de la verija
            if sv is not None:
                yfloor_ver = section_info(sv)['ymin']
                vol_piso_verija = clipped_volume_liters(V, F, xlo, xhi, yfloor_ver)
            else:
                vol_piso_verija = None

            # (8) piso a 0.5*dist hacia abajo desde la topline (techo del barril entre cruz y anca)
            ytop = max(ic['ymax'], ia['ymax'])      # línea superior (lomo)
            yfloor_mid = ytop - 0.5 * dist
            vol_piso_meddist = clipped_volume_liters(V, F, xlo, xhi, yfloor_mid)

            # (9) sin piso, tramo central: punto medio ± 0.25*dist
            yfull = V[:, 1].min() - 1.0
            xmid = (x_cruz + x_anca) / 2.0
            vol_central = clipped_volume_liters(V, F, xmid - 0.25 * dist, xmid + 0.25 * dist, yfull, 18)  # nsteps=18 = visor

            rows.append({
                '_dataset': dataset,
                'nombre': os.path.basename(ind_dir),
                'alt_calc_cm': meta.get('altura_real_cm'),
                'diam_anca_cm': round(perim_visor(V, F, x_anca), 1),
                'diam_cruz_cm': round(perim_visor(V, F, x_cruz), 1),
                'distancia_cm': round(dist, 1),
                'distancia_x1.5_cm': round(dist * 1.5, 1),
                'vol_piso_verija_L': (round(vol_piso_verija, 1) if vol_piso_verija is not None else ''),
                'vol_piso_meddist_L': round(vol_piso_meddist, 1),
                'vol_sin_piso_central_L': round(vol_central, 1),
            })
            r = rows[-1]
            print(f"  {dataset:8} {r['nombre']:14} alt={r['alt_calc_cm']}  "
                  f"diam(anca/cruz)={r['diam_anca_cm']:.0f}/{r['diam_cruz_cm']:.0f}cm  dist={r['distancia_cm']:.1f}  "
                  f"vol(ver/mid/centr)={r['vol_piso_verija_L']}/{r['vol_piso_meddist_L']}/{r['vol_sin_piso_central_L']}L")

    rows.sort(key=lambda r: (r['_dataset'], r['nombre']))
    cols = ['nombre', 'alt_calc_cm', 'diam_anca_cm', 'diam_cruz_cm', 'distancia_cm',
            'distancia_x1.5_cm', 'vol_piso_verija_L', 'vol_piso_meddist_L', 'vol_sin_piso_central_L']
    out = PROJ / 'output_cruz_modelos' / 'tabla_cruz_anca.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print(f"\n[done] {len(rows)} individuos -> {out}")


if __name__ == '__main__':
    main()
