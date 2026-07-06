"""
Tabla por individuo (modelos live v8) del corte cruz↔verija:
  - nombre
  - distancia entre el plano de la verija y el de la cruz (cm)  [= separación de las dos secciones]
  - volumen de la sección recortada (L): caja entre ambos planos con piso
    horizontal en el punto mínimo de la verija.

Replica exactamente el algoritmo del visor 3D (viewer3d.js): corta la malla en
planos X, arma el polígono real de cada sección, lo recorta al piso (y >= yMin
verija) e integra área·dx.

Salida: output_cruz_modelos/tabla_corte.csv  (+ tabla por consola)
Uso:  python tabla_volumen_corte.py
"""
import csv
import glob
import json
import os
from pathlib import Path
import numpy as np
import trimesh

PROJ = Path(__file__).resolve().parents[1]
V8 = {
    '6mayo':  'output_modelos3d_live_6mayo_v8',
    '14mayo': 'output_modelos3d_live_14mayo_v8',
    '20mayo': 'output_modelos3d_live_20mayo_v8',
    '12junio': 'output_modelos3d_live_12junio_v8',
}
NSTEPS = 36


def slice_section(V, F, xp):
    """Intersección malla–plano X=xp. Devuelve puntos (z,y) del contorno + bbox."""
    P = V[F]  # (nf,3,3)
    pts = []
    for tri in P:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            p, q = tri[a], tri[b]
            dp, dq = p[0] - xp, q[0] - xp
            if (dp < 0) != (dq < 0):
                t = dp / (dp - dq)
                pts.append((p[2] + (q[2] - p[2]) * t, p[1] + (q[1] - p[1]) * t))
    if len(pts) < 3:
        return None
    a = np.array(pts)
    return a


def section_info(a):
    """Polígono ordenado + bbox (ymin,ymax,zmin,zmax) + perímetro real."""
    ymin, ymax = a[:, 1].min(), a[:, 1].max()
    zmin, zmax = a[:, 0].min(), a[:, 0].max()
    yc, zc = (ymin + ymax) / 2, (zmin + zmax) / 2
    # puntos únicos (redondeo como el visor) y orden angular
    seen = set(); uniq = []
    for z, y in a:
        k = (round(z * 50), round(y * 50))
        if k in seen:
            continue
        seen.add(k); uniq.append((z, y))
    u = np.array(uniq)
    ang = np.arctan2(u[:, 1] - yc, u[:, 0] - zc)
    poly = u[np.argsort(ang)]
    # perímetro real (suma de segmentos del contorno cerrado)
    perim = float(np.sum(np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1)))
    return {'poly': poly, 'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax,
            'yc': yc, 'zc': zc, 'perim': perim}


def clip_above(poly, yfloor):
    """Sutherland-Hodgman: recorta a y >= yfloor. poly: array (n,2)=(z,y)."""
    out = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]; prev = poly[i - 1]
        ci = cur[1] >= yfloor; pi = prev[1] >= yfloor
        if ci:
            if not pi:
                t = (yfloor - prev[1]) / (cur[1] - prev[1])
                out.append((prev[0] + (cur[0] - prev[0]) * t, yfloor))
            out.append((cur[0], cur[1]))
        elif pi:
            t = (yfloor - prev[1]) / (cur[1] - prev[1])
            out.append((prev[0] + (cur[0] - prev[0]) * t, yfloor))
    return np.array(out) if len(out) >= 3 else None


def poly_area(poly):
    z = poly[:, 0]; y = poly[:, 1]
    return abs(np.sum(z * np.roll(y, -1) - np.roll(z, -1) * y)) / 2


def clipped_volume_liters(V, F, xlo, xhi, yfloor, nsteps=NSTEPS):
    if xhi - xlo <= 0:
        return 0.0
    dx = (xhi - xlo) / nsteps
    vol = 0.0
    for i in range(nsteps):
        a = slice_section(V, F, xlo + (i + 0.5) * dx)
        if a is None:
            continue
        info = section_info(a)
        cl = clip_above(info['poly'], yfloor)
        if cl is not None:
            vol += poly_area(cl) * dx
    return vol / 1000.0


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
            yfloor = iv['ymin']
            xlo, xhi = min(x_cruz, x_ver), max(x_cruz, x_ver)
            dist = xhi - xlo
            vol = clipped_volume_liters(V, F, xlo, xhi, yfloor)

            # 4 secciones iguales a lo largo del cuerpo, numeradas CRUZ(1)->VERIJA(4).
            # Cada sección = su cuarto de X, COMPLETA (sin el piso de la verija):
            # se integra toda la sección transversal real, incluyendo la panza.
            yfull = V[:, 1].min() - 1.0  # piso por debajo de todo -> no recorta nada
            step = (x_ver - x_cruz) / 4.0
            secs = []
            for i in range(4):
                a = x_cruz + i * step
                b = x_cruz + (i + 1) * step
                secs.append(round(clipped_volume_liters(V, F, min(a, b), max(a, b), yfull, 16), 1))

            rows.append({
                'nombre': os.path.basename(ind_dir),
                '_dataset': dataset,  # solo para ordenar; no se escribe
                'altura_calc_cm': meta.get('altura_real_cm'),
                'distancia_cm': round(dist, 1),
                'diam_verija_cm': round(iv['perim'], 1),   # "diámetro" = perímetro de la sección (conv. tabla_v8)
                'diam_cruz_cm': round(ic['perim'], 1),
                'vol_s2_s3_L': round(secs[1] + secs[2], 1),  # suma de S2 + S3 (único valor)
            })
            print(f"  {dataset:8} {rows[-1]['nombre']:12} alt={meta.get('altura_real_cm')}  "
                  f"dist={dist:6.1f}cm  diam(ver/cruz)={iv['perim']:.0f}/{ic['perim']:.0f}cm  "
                  f"S2+S3={secs[1] + secs[2]:.1f}L")

    rows.sort(key=lambda r: (r['_dataset'], r['nombre']))
    cols = ['nombre', 'altura_calc_cm', 'distancia_cm', 'diam_verija_cm', 'diam_cruz_cm', 'vol_s2_s3_L']
    out = PROJ / 'output_cruz_modelos' / 'tabla_corte.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

    print(f"\n{'nombre':14}{'alt_cm':>8}{'dist_cm':>9}{'diam_ver':>10}{'diam_cruz':>11}{'S2+S3_L':>10}")
    print('-' * 62)
    for r in rows:
        print(f"{r['nombre']:14}{str(r['altura_calc_cm']):>8}{r['distancia_cm']:>9.1f}"
              f"{r['diam_verija_cm']:>10.1f}{r['diam_cruz_cm']:>11.1f}{r['vol_s2_s3_L']:>10.1f}")
    print(f"\n[done] {len(rows)} individuos -> {out}")


if __name__ == '__main__':
    main()
