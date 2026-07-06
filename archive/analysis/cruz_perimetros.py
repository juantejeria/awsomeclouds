"""Lista el DIÁMETRO DE LA CRUZ de cada individuo, calculado igual que el visor
3D pero en el punto de la CRUZ: corta la malla del _3d.ply con el plano
x = xFront ∓ cruz_frac*L y mide el contorno real de la sección (perímetro, alto,
profundidad). Usa los modelos de BARRIL (torso) de los dirs v8 que sirve app.py.

cruz_frac viene del _resumen.json (lo escribió escribir_cruz_frac_manual.py a
partir de los puntos anotados a mano). Sentido (frente) = barril_dir.

NO regenera ni modifica nada: solo lee mallas + resumen y reporta. Salida CSV en
output_cruz_modelos/cruz_diametro.csv

Uso:  python cruz_perimetros.py
"""
import csv
import json
import math
from pathlib import Path

PROJ = Path(__file__).parent
DATASETS = ['output_modelos3d_live_14mayo_v8', 'output_modelos3d_live_20mayo_v8']
OUT_CSV = PROJ / 'output_cruz_modelos' / 'cruz_diametro.csv'


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
    """(perim, alto, profundidad) de la interseccion malla-plano x=xplane, o None."""
    segs = []
    ymin = zmin = math.inf; ymax = zmax = -math.inf
    for (i, j, k) in faces:
        tri = (verts[i], verts[j], verts[k])
        hits = []
        for a in range(3):
            p = tri[a]; q = tri[(a + 1) % 3]
            dp = p[0] - xplane; dq = q[0] - xplane
            if (dp < 0 and dq >= 0) or (dp >= 0 and dq < 0):
                t = dp / (dp - dq)
                y = p[1] + (q[1] - p[1]) * t
                z = p[2] + (q[2] - p[2]) * t
                hits.append((y, z))
        if len(hits) == 2:
            (y0, z0), (y1, z1) = hits
            segs.append(math.hypot(y1 - y0, z1 - z0))
            for (y, z) in hits:
                ymin = min(ymin, y); ymax = max(ymax, y)
                zmin = min(zmin, z); zmax = max(zmax, z)
    if not segs:
        return None
    return sum(segs), ymax - ymin, zmax - zmin


def cruz_de(model_dir):
    ply = next(model_dir.glob('*_3d.ply'), None)
    if ply is None:
        return None
    rj = next(model_dir.glob('*_resumen.json'), None)
    meta = json.loads(rj.read_text()) if rj else {}
    if meta.get('cruz_frac') is None:
        return None
    verts, faces = leer_ply(ply)
    if not verts or not faces:
        return None
    xs = [v[0] for v in verts]
    xmin, xmax = min(xs), max(xs); L = xmax - xmin
    if L <= 0:
        return None
    facing_right = (meta.get('barril_dir') != 'left')
    xfront = xmax if facing_right else xmin
    frac = float(meta['cruz_frac'])
    xplane = xfront - frac * L if facing_right else xfront + frac * L
    sec = seccion(verts, faces, xplane)
    for _ in range(4):
        if sec:
            break
        xplane += (-1 if facing_right else 1) * max(0.5, L * 0.01)
        sec = seccion(verts, faces, xplane)
    if not sec:
        return None
    perim, vert, depth = sec
    return {'perim': perim, 'vert': vert, 'depth': depth, 'frac': frac,
            'dir': meta.get('barril_dir', 'unknown'),
            'altura': meta.get('altura_real_cm'),
            'cruz_n': meta.get('cruz_n_manual'),
            'L': L}


def main():
    rows = []
    for ds in DATASETS:
        base = PROJ / ds
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or d.name.startswith('_'):
                continue
            r = cruz_de(d)
            if r:
                rows.append((ds.replace('output_modelos3d_live_', '').replace('_v8', ''), d.name, r))

    print(f"\n{'dataset':8} {'individuo':14} {'perim_cruz':>10} {'alto_sec':>9} {'prof':>7} "
          f"{'largoL':>7} {'frac':>5} {'altura':>7} {'dir':>5} {'n':>3}")
    print('-' * 92)
    for ds, ind, r in rows:
        alt = f"{r['altura']:.0f}" if r['altura'] else '  -'
        print(f"{ds:8} {ind:14} {r['perim']:>8.1f}cm {r['vert']:>7.1f}cm {r['depth']:>5.1f}cm "
              f"{r['L']:>6.1f} {r['frac']*100:>4.0f}% {alt:>7} {r['dir']:>5} {str(r['cruz_n'] or ''):>3}")
    if rows:
        ps = [r['perim'] for _, _, r in rows]
        vs = [r['vert'] for _, _, r in rows]
        print('-' * 92)
        print(f"n={len(rows)}  perim_cruz: min={min(ps):.1f} max={max(ps):.1f} medio={sum(ps)/len(ps):.1f} cm "
              f"| alto: medio={sum(vs)/len(vs):.1f} cm")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'individuo', 'perim_cruz_cm', 'alto_sec_cm', 'prof_cm',
                    'largo_barril_cm', 'cruz_frac', 'altura_real_cm', 'barril_dir', 'cruz_n_manual'])
        for ds, ind, r in rows:
            w.writerow([ds, ind, round(r['perim'], 1), round(r['vert'], 1), round(r['depth'], 1),
                        round(r['L'], 1), r['frac'], r['altura'], r['dir'], r['cruz_n']])
    print(f"\n[csv] -> {OUT_CSV}")


if __name__ == '__main__':
    main()
