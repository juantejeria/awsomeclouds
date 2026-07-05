"""Para individuos con AMBOS guardados (girth_frac_manual + verija_frac_manual):
- distancia en cm entre el plano torácico y el de la verija
- volumen de la malla ENTRE esos dos planos (integral de A(x) dx)

A(x) = área de la sección (intersección malla–plano x), ordenando el contorno
por ángulo alrededor del centroide (secciones tipo barril ~convexas).
Auto-chequeo: integrar TODO el rango ≈ volumen_malla_cerrada (resumen).
"""
import json, math
from pathlib import Path

PROJ = Path(__file__).parent
DATASETS = ['output_modelos3d_live_14mayo', 'output_modelos3d_live_20mayo']
NSLAB = 160          # slices para integrar el tramo
NFULL = 300          # slices para el auto-chequeo (rango completo)


def leer_ply(path):
    verts, faces = [], []
    with open(path) as f:
        nv = nf = 0
        while True:
            line = f.readline()
            if line.startswith('element vertex'):
                nv = int(line.split()[-1])
            elif line.startswith('element face'):
                nf = int(line.split()[-1])
            elif line.startswith('end_header'):
                break
        for _ in range(nv):
            p = f.readline().split()
            verts.append((float(p[0]), float(p[1]), float(p[2])))
        for _ in range(nf):
            p = f.readline().split()
            n = int(p[0])
            if n == 3:
                faces.append((int(p[1]), int(p[2]), int(p[3])))
            elif n == 4:
                a, b, c, d = int(p[1]), int(p[2]), int(p[3]), int(p[4])
                faces.append((a, b, c)); faces.append((a, c, d))
    return verts, faces


def slice_pts(verts, faces, xpl):
    pts = []
    for (i, j, k) in faces:
        tri = (verts[i], verts[j], verts[k])
        for a in range(3):
            p = tri[a]; q = tri[(a + 1) % 3]
            dp = p[0] - xpl; dq = q[0] - xpl
            if (dp < 0 and dq >= 0) or (dp >= 0 and dq < 0):
                t = dp / (dp - dq)
                pts.append((p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t))
    return pts


def slice_perim(verts, faces, xpl):
    """Perímetro real de la sección = suma de longitudes de los segmentos."""
    per = 0.0
    for (i, j, k) in faces:
        tri = (verts[i], verts[j], verts[k]); hits = []
        for a in range(3):
            p = tri[a]; q = tri[(a + 1) % 3]
            dp = p[0] - xpl; dq = q[0] - xpl
            if (dp < 0 and dq >= 0) or (dp >= 0 and dq < 0):
                t = dp / (dp - dq)
                hits.append((p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t))
        if len(hits) == 2:
            per += math.hypot(hits[0][0] - hits[1][0], hits[0][1] - hits[1][1])
    return per


def area_seccion(pts):
    if len(pts) < 3:
        return 0.0
    yc = sum(p[0] for p in pts) / len(pts)
    zc = sum(p[1] for p in pts) / len(pts)
    pts = sorted(pts, key=lambda p: math.atan2(p[1] - zc, p[0] - yc))
    a = 0.0
    for i in range(len(pts)):
        y0, z0 = pts[i]; y1, z1 = pts[(i + 1) % len(pts)]
        a += y0 * z1 - y1 * z0
    return abs(a) * 0.5


def integrar(verts, faces, x0, x1, n):
    if x1 < x0:
        x0, x1 = x1, x0
    dx = (x1 - x0) / n
    vol = 0.0
    for s in range(n):
        x = x0 + (s + 0.5) * dx
        vol += area_seccion(slice_pts(verts, faces, x)) * dx
    return vol


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
        gf = meta.get('girth_frac_manual'); vf = meta.get('verija_frac_manual')
        if gf is None or vf is None:
            continue
        ply = next(d.glob('*_3d.ply'), None)
        if not ply:
            continue
        verts, faces = leer_ply(ply)
        xs = [v[0] for v in verts]; xmin, xmax = min(xs), max(xs); L = xmax - xmin
        facing_right = (meta.get('barril_dir') != 'left')
        xg = (xmax - gf * L) if facing_right else (xmin + gf * L)      # torácico (frente)
        xv = (xmin + vf * L) if facing_right else (xmax - vf * L)      # verija (fondo)
        dist = abs(xg - xv)
        perim_g = slice_perim(verts, faces, xg)
        perim_v = slice_perim(verts, faces, xv)
        vol_tramo = integrar(verts, faces, xg, xv, NSLAB) / 1000.0      # L
        vol_full_chk = integrar(verts, faces, xmin, xmax, NFULL) / 1000.0
        vol_ref = meta.get('vol_barril_litros')
        rows.append((ds.replace('output_modelos3d_live_', ''), d.name,
                     perim_g, perim_v, dist, vol_tramo, vol_full_chk, vol_ref))

print(f"\n{'dataset':9} {'individuo':14} {'perim_torax':>11} {'perim_verija':>13} "
      f"{'dist':>7} {'vol_entre_L':>12} {'%barril':>8}")
print('-' * 82)
for ds, ind, pg, pv, dist, volt, vfull, vref in rows:
    if vref and vfull > 0:
        vol_cal = volt * vref / vfull; pct = 100 * vol_cal / vref
    else:
        vol_cal = volt; pct = float('nan')
    print(f"{ds:9} {ind:14} {pg:>9.1f}cm {pv:>11.1f}cm {dist:>5.1f}cm "
          f"{vol_cal:>10.1f}L {pct:>7.0f}%")
print('-' * 82)
print(f"n={len(rows)} individuos (con torácico + verija guardados)")
