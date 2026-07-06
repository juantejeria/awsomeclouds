"""
Valida que tabla_cruz_anca.csv == lo que calcula el visor 3D (viewer3d.js).

Reimplementa de forma independiente la lógica del visor:
  - precedencia de fracs: *_frac_manual > *_frac (cruz/anca) ; verija_manual > 0.25
  - planos: x_cruz = xFront - cf*L ; x_anca = xRear + af*L ; x_ver = xRear + vf*L
  - diámetro = PERÍMETRO por suma de segmentos crudos del corte (= viewer _sliceMeshAtX),
    NO el polígono angular (que sobreestimaba — bug de la vez pasada).
  - distancia = |x_cruz - x_anca|
Compara diámetros/distancia recomputados contra el CSV y reporta el máximo desvío.
Además muestra cuánto diferiría el diámetro con el método angular (el viejo).
"""
import csv
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import trimesh

from tabla_volumen_corte import slice_section, section_info

PROJ = Path(__file__).parent
V8 = {
    '6mayo': 'output_modelos3d_live_6mayo_v8', '12junio': 'output_modelos3d_live_12junio_v8',
    '14mayo': 'output_modelos3d_live_14mayo_v8', '20mayo': 'output_modelos3d_live_20mayo_v8',
}


def perim_raw(V, F, xp):
    """Perímetro = suma de cada segmento crudo del corte (idéntico a viewer3d.js)."""
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


def main():
    csv_rows = {r['nombre']: r for r in csv.DictReader(open('output_cruz_modelos/tabla_cruz_anca.csv'))}

    max_d_csv = 0.0      # |CSV - visor| (debe ser ~0)
    max_d_ang = 0.0      # |visor - angular| (lo que evitamos)
    worst_csv = worst_ang = None
    n = 0
    print(f"{'individuo':14}{'diam_cruz CSV/visor':>22}{'diam_anca CSV/visor':>22}{'Δang_max':>10}")
    for ds, d in V8.items():
        for ind_dir in sorted(glob.glob(str(PROJ / d / '*'))):
            name = os.path.basename(ind_dir)
            if name not in csv_rows:
                continue
            rj = glob.glob(f'{ind_dir}/*_resumen.json'); ply = glob.glob(f'{ind_dir}/*_3d.ply')
            if not rj or not ply:
                continue
            meta = json.load(open(rj[0]))
            mesh = trimesh.load(ply[0], process=False)
            V = np.asarray(mesh.vertices, float); F = np.asarray(mesh.faces, int)
            x = V[:, 0]; xmin, xmax = x.min(), x.max(); L = xmax - xmin
            fr = (meta.get('barril_dir') != 'left')
            xfront = xmax if fr else xmin; xrear = xmin if fr else xmax
            cf = meta['cruz_frac_manual'] if meta.get('cruz_frac_manual') is not None else meta['cruz_frac']
            af = meta['anca_frac_manual'] if meta.get('anca_frac_manual') is not None else meta['anca_frac']
            x_cruz = xfront - cf * L if fr else xfront + cf * L
            x_anca = xrear + af * L if fr else xrear - af * L

            # visor (raw) vs angular
            dc_v = perim_raw(V, F, x_cruz); da_v = perim_raw(V, F, x_anca)
            dc_a = section_info(slice_section(V, F, x_cruz))['perim']
            da_a = section_info(slice_section(V, F, x_anca))['perim']
            dist_v = abs(x_cruz - x_anca)

            row = csv_rows[name]
            dc_csv = float(row['diam_cruz_cm']); da_csv = float(row['diam_anca_cm']); dist_csv = float(row['distancia_cm'])

            for diff, lab in [(abs(dc_csv - dc_v), (name, 'cruz')), (abs(da_csv - da_v), (name, 'anca')),
                              (abs(dist_csv - dist_v), (name, 'dist'))]:
                if diff > max_d_csv:
                    max_d_csv = diff; worst_csv = lab
            for diff, lab in [(abs(dc_v - dc_a), (name, 'cruz')), (abs(da_v - da_a), (name, 'anca'))]:
                if diff > max_d_ang:
                    max_d_ang = diff; worst_ang = lab
            n += 1
            dmax_ang = max(abs(dc_v - dc_a), abs(da_v - da_a))
            print(f"{name:14}{dc_csv:8.1f}/{dc_v:<7.1f}  {da_csv:8.1f}/{da_v:<7.1f}  {dmax_ang:8.1f}")

    print("\n" + "=" * 60)
    print(f"Validados: {n} individuos")
    print(f"Máx |CSV - visor(raw)|  = {max_d_csv:.3f} cm   (en {worst_csv})   -> debe ser ~0")
    print(f"Máx |visor - angular|   = {max_d_ang:.2f} cm   (en {worst_ang})   <- desvío que tendría el método viejo")
    print("OK: el CSV usa el MISMO perímetro que la UI (suma de segmentos)." if max_d_csv < 0.5
          else "ATENCIÓN: hay desvío CSV vs visor, revisar.")


if __name__ == '__main__':
    main()
