"""Cose un PLY shell simétrico (2 mitades z+/z- desconectadas) en una sola
malla cerrada uniendo los lazos de borde por proximidad XY.

Uso:
    python coser_shell.py input.ply output.ply
"""
import sys
import numpy as np
from collections import defaultdict, Counter


def leer_ply(path):
    verts, colors, faces = [], [], []
    n_v = n_f = 0
    has_color = False
    header = True
    comments = []
    with open(path) as f:
        for line in f:
            if header:
                comments.append(line) if line.startswith('comment') else None
                if line.startswith('element vertex'):
                    n_v = int(line.split()[2])
                elif line.startswith('element face'):
                    n_f = int(line.split()[2])
                elif line.startswith('property') and ('red' in line or 'green' in line or 'blue' in line):
                    has_color = True
                elif line.strip() == 'end_header':
                    header = False
                continue
            parts = line.split()
            if len(verts) < n_v:
                verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if has_color and len(parts) >= 6:
                    colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
            elif len(faces) < n_f:
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return (np.array(verts, dtype=float),
            np.array(colors, dtype=np.uint8) if colors else None,
            np.array(faces, dtype=int),
            comments)


def aristas_borde(faces):
    """Retorna lista de aristas (a,b) con a<b que aparecen en exactamente 1 cara."""
    cnt = Counter()
    for a, b, c in faces:
        cnt[(min(a, b), max(a, b))] += 1
        cnt[(min(b, c), max(b, c))] += 1
        cnt[(min(a, c), max(a, c))] += 1
    return [e for e, k in cnt.items() if k == 1]


def lazos_de_aristas(boundary_edges):
    """Convierte lista de aristas de borde en lazos cerrados ordenados."""
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = set()
    loops = []
    for v0 in list(adj.keys()):
        if v0 in seen:
            continue
        loop = [v0]
        seen.add(v0)
        prev = -1
        cur = v0
        while True:
            nxt_candidates = [w for w in adj[cur] if w != prev]
            if not nxt_candidates:
                break
            nxt = nxt_candidates[0]
            if nxt in seen:
                break
            seen.add(nxt)
            loop.append(nxt)
            prev = cur
            cur = nxt
        loops.append(loop)
    return loops


def coser(verts, faces):
    """Identifica las 2 componentes (z+ y z-), encuentra sus lazos de borde
    y agrega triángulos laterales que las cosen por correspondencia 1:1
    en el plano XY (vecino más cercano)."""
    # Componentes por vértice (BFS por adyacencia de caras)
    adj = defaultdict(set)
    for a, b, c in faces:
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])
    seen = set()
    comps = []
    for v in range(len(verts)):
        if v in seen:
            continue
        stack = [v]
        comp = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.add(u)
            stack.extend(adj[u] - seen)
        comps.append(comp)

    if len(comps) < 2:
        print(f"[coser] solo {len(comps)} componentes; no hay nada que coser.")
        return faces
    if len(comps) > 2:
        # Ignorar islas pequeñas (vértices sin caras u otros artefactos);
        # cosemos las 2 componentes mayores como z+/z-.
        comps_sorted = sorted(comps, key=len, reverse=True)
        print(f"[coser] {len(comps)} componentes detectadas; usando las 2 mayores "
              f"({len(comps_sorted[0])}, {len(comps_sorted[1])} verts) como z+/z-.")
        comps = comps_sorted[:2]

    # Identificar cuál es z+ y cuál z-
    c0_z = np.mean([verts[i, 2] for i in comps[0]])
    c1_z = np.mean([verts[i, 2] for i in comps[1]])
    if c0_z > c1_z:
        comp_pos, comp_neg = comps[0], comps[1]
    else:
        comp_pos, comp_neg = comps[1], comps[0]

    # Caras de cada componente
    f_pos = np.array([f for f in faces if f[0] in comp_pos])
    f_neg = np.array([f for f in faces if f[0] in comp_neg])

    # Lazos de borde de cada componente
    bp = aristas_borde(f_pos)
    bn = aristas_borde(f_neg)
    loops_p = lazos_de_aristas(bp)
    loops_n = lazos_de_aristas(bn)
    if not loops_p or not loops_n:
        print("[coser] no hay lazos de borde, malla ya cerrada.")
        return faces
    loop_p = max(loops_p, key=len)
    loop_n = max(loops_n, key=len)
    print(f"[coser] lazo z+: {len(loop_p)} verts | lazo z-: {len(loop_n)} verts")

    # Para cada vértice del lazo z+, buscar su vecino más cercano en XY del lazo z-
    pp = verts[loop_p][:, :2]
    pn = verts[loop_n][:, :2]
    # match j = argmin |pp[i] - pn[j]|
    match_pn = []
    for i in range(len(loop_p)):
        d = np.linalg.norm(pn - pp[i], axis=1)
        match_pn.append(int(np.argmin(d)))

    # Generar triángulos de cosido recorriendo el lazo z+ en orden
    nuevos = []
    n = len(loop_p)
    for i in range(n):
        i_next = (i + 1) % n
        a = loop_p[i]
        b = loop_p[i_next]
        c = loop_n[match_pn[i]]
        d = loop_n[match_pn[i_next]]
        # Quadrilátero (a,b,d,c) → 2 triángulos
        if c == d:
            # Solo 1 triángulo (vecinos compartidos)
            nuevos.append([a, b, c])
        else:
            nuevos.append([a, b, d])
            nuevos.append([a, d, c])

    print(f"[coser] {len(nuevos)} triángulos laterales agregados")
    return np.vstack([faces, np.array(nuevos, dtype=int)])


def escribir_ply(path, verts, colors, faces, comments):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        for c in comments:
            f.write(c)
        f.write("comment Cosido z+/z- por coser_shell.py\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for i, p in enumerate(verts):
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{p[0]:.2f} {p[1]:.2f} {p[2]:.2f} {r} {g} {b}\n")
            else:
                f.write(f"{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}\n")
        for t in faces:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python coser_shell.py input.ply output.ply")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    verts, colors, faces, comments = leer_ply(src)
    print(f"[in]  {src}: {len(verts)} verts, {len(faces)} faces")
    new_faces = coser(verts, faces)
    escribir_ply(dst, verts, colors, new_faces, comments)
    print(f"[out] {dst}: {len(verts)} verts, {len(new_faces)} faces")
