"""
Tabla por individuo del VOLUMEN DE CORTE cruz↔anca barriendo el % de corte
desde 40% hasta 70% (de a 1%).

Replica exactamente el "Vol. corte" del visor 3D (viewer3d.js):
  - El tramo va del plano de la CRUZ al plano de la ANCA (xlo..xhi).
  - El piso del corte se mide SIEMPRE hacia abajo desde el punto más alto del
    diámetro de la CRUZ:  yfloor = cruz.ymax - (pct/100) * dist
    (dist = distancia horizontal cruz↔anca).
  - Volumen = integral del área de cada sección recortada al piso, sobre X.

Usa los mismos puntos guardados por individuo:
  cruz_frac_manual (>cruz_frac>0.20)  y  anca_frac_manual (>anca_frac>0.25).

Salida:
  output_cruz_modelos/tabla_corte_barrido.csv   (columnas vol_40 .. vol_70, en L)
Uso:  python tabla_corte_barrido.py
"""
import csv
import glob
import json
import os
import numpy as np
import trimesh

# Reusamos las funciones de slicing/volumen del script existente (idénticas al visor).
from tabla_volumen_corte import (
    slice_section, section_info, clipped_volume_liters, V8, PROJ, NSTEPS,
)

# % de corte a barrer (inclusive)
PCTS = list(range(40, 71))  # 40,41,...,70

# Individuos ocultos en el visor 3D (se excluyen también de la tabla, para que
# coincida con lo que se ve). Quitar de acá si se quieren incluir.
OCULTOS = {
    ('12junio', '000_392'),
    ('12junio', '000_392A'),
    ('12junio', '000_448A'),
    ('12junio', '000_459'),
}


def _clamp(v, lo, hi):
    return min(max(float(v), lo), hi)


def main():
    rows = []
    for dataset, d in V8.items():
        for ind_dir in sorted(glob.glob(str(PROJ / d / '*'))):
            if not os.path.isdir(ind_dir):
                continue
            nombre = os.path.basename(ind_dir)
            if (dataset, nombre) in OCULTOS:
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

            x = V[:, 0]
            xmin, xmax = x.min(), x.max()
            L = xmax - xmin
            facing_right = (meta.get('barril_dir') != 'left')
            xfront = xmax if facing_right else xmin
            xrear = xmin if facing_right else xmax

            cf = meta.get('cruz_frac_manual')
            if cf is None:
                cf = meta.get('cruz_frac', 0.20)
            af = meta.get('anca_frac_manual')
            if af is None:
                af = meta.get('anca_frac', 0.25)
            cf = _clamp(cf, 0.0, 0.5)
            af = _clamp(af, 0.0, 0.5)

            x_cruz = xfront - cf * L if facing_right else xfront + cf * L
            x_anca = xrear + af * L if facing_right else xrear - af * L

            sc = slice_section(V, F, x_cruz)
            sa = slice_section(V, F, x_anca)
            if sc is None or sa is None:
                print(f"[warn] {dataset}/{nombre}: sección cruz/anca vacía -> salteado")
                continue
            ic = section_info(sc)
            ia = section_info(sa)

            xlo, xhi = min(x_cruz, x_anca), max(x_cruz, x_anca)
            dist = xhi - xlo
            y_top = ic['ymax']  # punto más alto del diámetro de la CRUZ

            row = {
                'nombre': nombre,
                '_dataset': dataset,
                'altura_calc_cm': meta.get('altura_real_cm'),
                'dist_cruz_anca_cm': round(dist, 1),
                'diam_cruz_cm': round(ic['perim'], 1),
                'diam_anca_cm': round(ia['perim'], 1),
            }
            for pct in PCTS:
                yfloor = y_top - (pct / 100.0) * dist
                vol = clipped_volume_liters(V, F, xlo, xhi, yfloor, NSTEPS)
                row[f'vol_{pct}pct_L'] = round(vol, 1)
            rows.append(row)

            # consola: muestra algunos puntos del barrido
            muestra = '  '.join(f"{p}%={row[f'vol_{p}pct_L']:.0f}" for p in (40, 50, 60, 70))
            print(f"  {dataset:8} {nombre:12} dist={dist:6.1f}cm  {muestra}")

    rows.sort(key=lambda r: (r['_dataset'], r['nombre']))

    cols = (['nombre', '_dataset', 'altura_calc_cm', 'dist_cruz_anca_cm',
             'diam_cruz_cm', 'diam_anca_cm'] + [f'vol_{p}pct_L' for p in PCTS])
    out = PROJ / 'output_cruz_modelos' / 'tabla_corte_barrido.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        # encabezado: renombramos _dataset -> dataset
        w.writerow({c: ('dataset' if c == '_dataset' else c) for c in cols})
        for r in rows:
            w.writerow(r)

    print(f"\n[done] {len(rows)} individuos x {len(PCTS)} cortes (40%–70%) -> {out}")


if __name__ == '__main__':
    main()
