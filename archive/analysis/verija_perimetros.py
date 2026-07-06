"""Lista el DIÁMETRO VERIJA (perímetro real, alto, prof) de cada individuo,
calculado en TU posición guardada manualmente (verija_frac_manual, desde el
fondo). Mismo corte de malla que el visor. Solo individuos con verija guardada.
"""
import json, math
from pathlib import Path

PROJ = Path(__file__).parent
DATASETS = ['output_modelos3d_live_14mayo', 'output_modelos3d_live_20mayo']


def leer_ply(path):
    verts, faces = [], []
    with open(path) as f:
        nverts = nfaces = 0
        while True:
            line = f.readline()
            if line.startswith('element vertex'):
                nverts = int(line.split()[-1])
            elif line.startswith('element face'):
                nfaces = int(line.split()[-1])
            elif line.startswith('end_header'):
                break
        for _ in range(nverts):
            p = f.readline().split()
            verts.append((float(p[0]), float(p[1]), float(p[2])))
        for _ in range(nfaces):
            p = f.readline().split()
            n = int(p[0])
            if n == 3:
                faces.append((int(p[1]), int(p[2]), int(p[3])))
            elif n == 4:
                a, b, c, d = int(p[1]), int(p[2]), int(p[3]), int(p[4])
                faces.append((a, b, c)); faces.append((a, c, d))
    return verts, faces


def seccion(verts, faces, xplane):
    segs = []
    ymin = zmin = math.inf; ymax = zmax = -math.inf
    for (i, j, k) in faces:
        tri = (verts[i], verts[j], verts[k]); hits = []
        for a in range(3):
            p = tri[a]; q = tri[(a + 1) % 3]
            dp = p[0] - xplane; dq = q[0] - xplane
            if (dp < 0 and dq >= 0) or (dp >= 0 and dq < 0):
                t = dp / (dp - dq)
                hits.append((p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t))
        if len(hits) == 2:
            (y0, z0), (y1, z1) = hits
            segs.append(math.hypot(y1 - y0, z1 - z0))
            for (y, z) in hits:
                ymin = min(ymin, y); ymax = max(ymax, y)
                zmin = min(zmin, z); zmax = max(zmax, z)
    if not segs:
        return None
    return sum(segs), ymax - ymin, zmax - zmin


def verija_de(model_dir):
    rj = next(model_dir.glob('*_resumen.json'), None)
    if not rj:
        return None
    meta = json.loads(rj.read_text())
    frac = meta.get('verija_frac_manual')
    if frac is None:
        return None
    ply = next(model_dir.glob('*_3d.ply'), None)
    if not ply:
        return None
    verts, faces = leer_ply(ply)
    if not verts or not faces:
        return None
    xs = [v[0] for v in verts]; xmin, xmax = min(xs), max(xs); L = xmax - xmin
    facing_right = (meta.get('barril_dir') != 'left')
    xrear = xmin if facing_right else xmax
    xplane = xrear + frac * L if facing_right else xrear - frac * L
    sec = seccion(verts, faces, xplane)
    for _ in range(4):
        if sec:
            break
        xplane += (1 if facing_right else -1) * max(0.5, L * 0.01)
        sec = seccion(verts, faces, xplane)
    if not sec:
        return None
    perim, vert, depth = sec
    return {'perim': perim, 'vert': vert, 'depth': depth, 'frac': frac,
            'dir': meta.get('barril_dir', 'unknown')}


rows = []
for ds in DATASETS:
    base = PROJ / ds
    if not base.is_dir():
        continue
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        r = verija_de(d)
        if r:
            rows.append((ds.replace('output_modelos3d_live_', ''), d.name, r))

print(f"\n{'dataset':9} {'individuo':14} {'perim_verija':>12} {'alto':>8} {'prof':>7} {'frac_fondo':>11}")
print('-' * 70)
for ds, ind, r in rows:
    print(f"{ds:9} {ind:14} {r['perim']:>10.1f}cm {r['vert']:>6.1f}cm {r['depth']:>5.1f}cm {r['frac']*100:>9.0f}%")
if rows:
    ps = [r['perim'] for _, _, r in rows]
    print('-' * 70)
    print(f"n={len(rows)} con verija guardada  |  perím: min={min(ps):.1f}  max={max(ps):.1f}  "
          f"medio={sum(ps)/len(ps):.1f} cm")
