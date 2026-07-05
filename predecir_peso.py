"""Busca una constante que prediga el PESO REAL de los 15 individuos con
torácico + verija guardados. Target = peso real (2do nº del nombre de carpeta).
Features: perim_torax, perim_verija, dist, vol_entre, vol_barril, altura real/calc.
Para cada predictor candidato evalúa la constante k=peso/feature (su CV) y el
error de predecir peso = k*feature.
"""
import json, math
from pathlib import Path
import numpy as np

PROJ = Path(__file__).parent
DATASETS = ['output_modelos3d_live_14mayo', 'output_modelos3d_live_20mayo']


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
            p = f.readline().split(); verts.append((float(p[0]), float(p[1]), float(p[2])))
        for _ in range(nf):
            p = f.readline().split(); n = int(p[0])
            if n == 3:
                faces.append((int(p[1]), int(p[2]), int(p[3])))
            elif n == 4:
                a, b, c, d = int(p[1]), int(p[2]), int(p[3]), int(p[4])
                faces += [(a, b, c), (a, c, d)]
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


def perim_at(verts, faces, xpl):
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


def area_sec(pts):
    if len(pts) < 3:
        return 0.0
    yc = sum(p[0] for p in pts) / len(pts); zc = sum(p[1] for p in pts) / len(pts)
    pts = sorted(pts, key=lambda p: math.atan2(p[1] - zc, p[0] - yc))
    a = 0.0
    for i in range(len(pts)):
        y0, z0 = pts[i]; y1, z1 = pts[(i + 1) % len(pts)]
        a += y0 * z1 - y1 * z0
    return abs(a) * 0.5


def integrar(verts, faces, x0, x1, n):
    if x1 < x0:
        x0, x1 = x1, x0
    dx = (x1 - x0) / n; v = 0.0
    for s in range(n):
        v += area_sec(slice_pts(verts, faces, x0 + (s + 0.5) * dx)) * dx
    return v


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
        gf, vf = meta.get('girth_frac_manual'), meta.get('verija_frac_manual')
        if gf is None or vf is None:
            continue
        try:
            altura_real, peso_real = [float(t) for t in d.name.split('_')[:2]]
        except Exception:
            continue
        ply = next(d.glob('*_3d.ply'), None)
        verts, faces = leer_ply(ply)
        xs = [v[0] for v in verts]; xmin, xmax = min(xs), max(xs); L = xmax - xmin
        fr = (meta.get('barril_dir') != 'left')
        xg = (xmax - gf * L) if fr else (xmin + gf * L)
        xv = (xmin + vf * L) if fr else (xmax - vf * L)
        dist = abs(xg - xv)
        pt = perim_at(verts, faces, xg); pv = perim_at(verts, faces, xv)
        vol_ref = meta.get('vol_barril_litros') or 0
        vfull = integrar(verts, faces, xmin, xmax, 240) / 1000.0
        vtramo = integrar(verts, faces, xg, xv, 140) / 1000.0
        vol_entre = vtramo * vol_ref / vfull if vfull > 0 and vol_ref else vtramo
        rows.append(dict(ind=d.name, peso=peso_real, alt_real=altura_real,
                         alt_calc=meta.get('altura_real_cm'), pt=pt, pv=pv, dist=dist,
                         L=L, vol_entre=vol_entre, vol_barril=vol_ref))

peso = np.array([r['peso'] for r in rows])
print(f"\n=== {len(rows)} individuos | peso real: {peso.min():.0f}–{peso.max():.0f} kg (medio {peso.mean():.0f}) ===\n")

# candidatos: nombre -> valor por individuo
def col(key):
    return np.array([r[key] for r in rows], float)

cand = {
    'vol_entre (L)':            col('vol_entre'),
    'vol_barril (L)':           col('vol_barril'),
    'perim_torax^2 * dist':     col('pt')**2 * col('dist') / 1e4,
    'perim_torax^2 * L_barril': col('pt')**2 * col('L') / 1e4,
    'perim_torax^2 * alt_real': col('pt')**2 * col('alt_real') / 1e4,
    # ── variantes con VERIJA ──
    'perim_verija^2 * dist':    col('pv')**2 * col('dist') / 1e4,
    'perim_verija^2 * L_barril':col('pv')**2 * col('L') / 1e4,
    'perim_verija^2 * alt_calc':col('pv')**2 * col('alt_calc') / 1e4,
    'perim_verija^2 * alt_real':col('pv')**2 * col('alt_real') / 1e4,
    'perim_verija^3':           col('pv')**3 / 1e5,
    'perim_tx*perim_vrj*dist':  col('pt') * col('pv') * col('dist') / 1e4,
    '(pt^2+pv^2)/2 * dist':     (col('pt')**2 + col('pv')**2) / 2 * col('dist') / 1e4,
}

print(f"{'predictor':26}{'k=peso/feat':>13}{'CV_k':>8}{'R2(orig)':>10}{'MAPE_pred':>11}")
print('-' * 70)
res = []
for name, x in cand.items():
    k = peso / x
    kbar = k.mean(); cv = 100 * k.std() / kbar
    # fit por origen: peso ~ kbar*x ; error
    pred = kbar * x
    mape = 100 * np.mean(np.abs(pred - peso) / peso)
    # R2 (a traves del origen)
    ss_res = np.sum((peso - pred)**2); ss_tot = np.sum((peso - peso.mean())**2)
    r2 = 1 - ss_res / ss_tot
    res.append((cv, name, kbar, r2, mape))
    print(f"{name:26}{kbar:>13.4f}{cv:>7.1f}%{r2:>10.2f}{mape:>10.1f}%")

res.sort()
print('-' * 70)
best = res[0]
print(f"\nMEJOR constante: '{best[1]}'  k={best[2]:.4f}  (CV {best[0]:.1f}%, MAPE {best[4]:.1f}%)")
print("\n=== detalle por individuo ===")
print(f"{'ind':14}{'peso_real':>9}{'vol_entre':>10}{'pt':>7}{'pv':>7}{'dist':>7}{'alt_calc':>9}")
for r in rows:
    print(f"{r['ind']:14}{r['peso']:>9.0f}{r['vol_entre']:>10.1f}{r['pt']:>7.0f}{r['pv']:>7.0f}{r['dist']:>7.0f}{(r['alt_calc'] or 0):>9.1f}")


# ── Ajustes con intercepto / alométrico + validación leave-one-out ──
def loo_linear(x, y):
    n = len(x); errs = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        A = np.vstack([x[m], np.ones(m.sum())]).T
        a, b = np.linalg.lstsq(A, y[m], rcond=None)[0]
        pred = a * x[i] + b
        errs.append(abs(pred - y[i]) / y[i])
    A = np.vstack([x, np.ones(n)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b, 100 * np.mean(errs)


def loo_power(x, y):
    n = len(x); errs = []
    lx, ly = np.log(x), np.log(y)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        A = np.vstack([lx[m], np.ones(m.sum())]).T
        b, la = np.linalg.lstsq(A, ly[m], rcond=None)[0]
        pred = np.exp(la) * x[i]**b
        errs.append(abs(pred - y[i]) / y[i])
    A = np.vstack([lx, np.ones(n)]).T
    b, la = np.linalg.lstsq(A, ly, rcond=None)[0]
    return np.exp(la), b, 100 * np.mean(errs)


print("\n=== Ajustes (LOO = error honesto fuera de muestra) — top 5 por CV ===")
for cv, name, kbar, r2, mape in res[:5]:
    x = cand[name]
    a, b, loom = loo_linear(x, peso)
    A, B, loop = loo_power(x, peso)
    print(f"  {name:26} k={kbar:.4f}  LOO(origen)~{mape:.1f}%  "
          f"LOO(intcpt)={loom:.1f}%  LOO(alom)={loop:.1f}%")
