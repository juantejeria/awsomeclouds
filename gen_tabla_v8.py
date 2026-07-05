"""Tabla de los 65 individuos v8: nombre, altura calc, diam verija, diam torácico,
distancia entre planos, volumen del tramo entre diámetros.
Usa girth_frac_manual / verija_frac_manual guardados en cada resumen v8.
Salida: imprime tabla + escribe tabla_v8.csv
"""
import json, math, csv
from pathlib import Path

PROJ = Path(__file__).parent
DIRS = [('14mayo', 'output_modelos3d_live_14mayo_v8'),
        ('20mayo', 'output_modelos3d_live_20mayo_v8'),
        ('6mayo',  'output_modelos3d_live_6mayo_v8'),
        ('12junio', 'output_modelos3d_live_12junio_v8')]
NSLAB, NFULL = 120, 160


def leer_ply(path):
    verts, faces = [], []
    with open(path) as f:
        nv = nf = 0
        while True:
            ln = f.readline()
            if ln.startswith('element vertex'): nv = int(ln.split()[-1])
            elif ln.startswith('element face'): nf = int(ln.split()[-1])
            elif ln.startswith('end_header'): break
        for _ in range(nv):
            p = f.readline().split(); verts.append((float(p[0]), float(p[1]), float(p[2])))
        for _ in range(nf):
            p = f.readline().split(); n = int(p[0])
            if n == 3: faces.append((int(p[1]), int(p[2]), int(p[3])))
            elif n == 4:
                a, b, c, d = int(p[1]), int(p[2]), int(p[3]), int(p[4]); faces += [(a, b, c), (a, c, d)]
    return verts, faces


def slice_pts(verts, faces, xpl):
    pts = []
    for (i, j, k) in faces:
        tri = (verts[i], verts[j], verts[k])
        for a in range(3):
            p = tri[a]; q = tri[(a+1) % 3]; dp = p[0]-xpl; dq = q[0]-xpl
            if (dp < 0 and dq >= 0) or (dp >= 0 and dq < 0):
                t = dp/(dp-dq); pts.append((p[1]+(q[1]-p[1])*t, p[2]+(q[2]-p[2])*t))
    return pts


def perim_at(verts, faces, xpl):
    per = 0.0
    for (i, j, k) in faces:
        tri = (verts[i], verts[j], verts[k]); hits = []
        for a in range(3):
            p = tri[a]; q = tri[(a+1) % 3]; dp = p[0]-xpl; dq = q[0]-xpl
            if (dp < 0 and dq >= 0) or (dp >= 0 and dq < 0):
                t = dp/(dp-dq); hits.append((p[1]+(q[1]-p[1])*t, p[2]+(q[2]-p[2])*t))
        if len(hits) == 2:
            per += math.hypot(hits[0][0]-hits[1][0], hits[0][1]-hits[1][1])
    return per


def area_sec(pts):
    if len(pts) < 3: return 0.0
    yc = sum(p[0] for p in pts)/len(pts); zc = sum(p[1] for p in pts)/len(pts)
    pts = sorted(pts, key=lambda p: math.atan2(p[1]-zc, p[0]-yc)); a = 0.0
    for i in range(len(pts)):
        y0, z0 = pts[i]; y1, z1 = pts[(i+1) % len(pts)]; a += y0*z1 - y1*z0
    return abs(a)*0.5


def integrar(verts, faces, x0, x1, n):
    if x1 < x0: x0, x1 = x1, x0
    dx = (x1-x0)/n; v = 0.0
    for s in range(n):
        v += area_sec(slice_pts(verts, faces, x0+(s+0.5)*dx))*dx
    return v


rows = []
for ds, d in DIRS:
    base = PROJ/d
    if not base.is_dir(): continue
    for sub in sorted(base.iterdir()):
        rj = next(sub.glob('*_resumen.json'), None)
        if not rj: continue
        m = json.loads(rj.read_text()); ind = sub.name
        alt = m.get('altura_real_cm'); gf = m.get('girth_frac_manual'); vf = m.get('verija_frac_manual')
        row = {'dataset': ds, 'individuo': ind, 'altura_calc_cm': alt,
               'diam_verija_cm': None, 'diam_torax_cm': None, 'dist_cm': None, 'vol_tramo_L': None}
        if gf is not None and vf is not None:
            ply = next(sub.glob('*_3d.ply'), None)
            if ply:
                verts, faces = leer_ply(ply)
                xs = [v[0] for v in verts]; xmin, xmax = min(xs), max(xs); L = xmax-xmin
                fr = (m.get('barril_dir') != 'left')
                xg = (xmax-gf*L) if fr else (xmin+gf*L)
                xv = (xmin+vf*L) if fr else (xmax-vf*L)
                row['diam_torax_cm'] = round(perim_at(verts, faces, xg), 1)
                row['diam_verija_cm'] = round(perim_at(verts, faces, xv), 1)
                row['dist_cm'] = round(abs(xg-xv), 1)
                vref = m.get('vol_barril_litros') or 0
                vfull = integrar(verts, faces, xmin, xmax, NFULL)/1000.0
                vtr = integrar(verts, faces, xg, xv, NSLAB)/1000.0
                row['vol_tramo_L'] = round(vtr*vref/vfull, 1) if (vfull > 0 and vref) else round(vtr, 1)
        rows.append(row)

# salida
hdr = ['dataset', 'individuo', 'altura_calc_cm', 'diam_verija_cm', 'diam_torax_cm', 'dist_cm', 'vol_tramo_L']
with open(PROJ/'tabla_v8.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader(); w.writerows(rows)

def fmt(v): return '-' if v is None else (f'{v:.1f}' if isinstance(v, float) else str(v))
cur = None
print(f"{'individuo':16}{'alt_calc':>9}{'verija':>9}{'torax':>9}{'dist':>8}{'vol_L':>8}")
for r in rows:
    if r['dataset'] != cur:
        cur = r['dataset']; print(f"\n--- {cur} ---")
    print(f"{r['individuo']:16}{fmt(r['altura_calc_cm']):>9}{fmt(r['diam_verija_cm']):>9}"
          f"{fmt(r['diam_torax_cm']):>9}{fmt(r['dist_cm']):>8}{fmt(r['vol_tramo_L']):>8}")
comp = sum(1 for r in rows if r['vol_tramo_L'] is not None)
print(f"\n[total] {len(rows)} individuos | con torácico+verija guardados: {comp} | faltan: {len(rows)-comp}")
print("CSV: tabla_v8.csv")
