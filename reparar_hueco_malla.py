"""Detecta y rellena huecos internos en una malla 2D triangulada (PLY plano).

Algoritmo:
1. Calcula longitudes de aristas; identifica triángulos con arista atípica.
2. Los elimina; eso abre un agujero topológico donde había triángulos largos.
3. Detecta lazos de borde tras filtrar; el lazo externo es el más grande,
   los demás son agujeros internos.
4. Genera grid de vértices nuevos sobre la zona del agujero (con la misma
   densidad que el resto, derivada de la mediana de aristas).
5. Re-triangula todos los vértices con Delaunay y filtra por estar dentro
   del polígono externo y con aristas no demasiado largas.

Uso:
    python reparar_hueco_malla.py input_lateral.ply output_lateral.ply

Regenera la triangulación 2D. Para regenerar el _3d.ply (simétrico) y el
_volumen.ply, ver `aplicar_reparar_test2barril.py`.
"""
import sys
import numpy as np
import cv2
from collections import defaultdict, Counter
from scipy.spatial import Delaunay


def leer_ply_plano(path):
    verts, colors, faces = [], [], []
    n_v = n_f = 0
    has_color = False
    header = True
    comments = []
    with open(path) as f:
        for line in f:
            if header:
                if line.startswith('comment'):
                    comments.append(line)
                if line.startswith('element vertex'):
                    n_v = int(line.split()[2])
                elif line.startswith('element face'):
                    n_f = int(line.split()[2])
                elif line.startswith('property') and any(k in line for k in ('red', 'green', 'blue')):
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


def _edges_from_faces(faces):
    cnt = Counter()
    for a, b, c in faces:
        cnt[(min(a, b), max(a, b))] += 1
        cnt[(min(b, c), max(b, c))] += 1
        cnt[(min(a, c), max(a, c))] += 1
    return cnt


def _boundary_loops(faces):
    cnt = _edges_from_faces(faces)
    boundary = [e for e, k in cnt.items() if k == 1]
    adj = defaultdict(list)
    for a, b in boundary:
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
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt:
                break
            n = nxt[0]
            if n in seen:
                # cierra el lazo
                break
            seen.add(n)
            loop.append(n)
            prev = cur
            cur = n
        loops.append(loop)
    return loops


def _loop_area(verts2d, loop):
    pts = verts2d[loop]
    x, y = pts[:, 0], pts[:, 1]
    n = len(loop)
    return 0.5 * abs(sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i] for i in range(n)))


def _max_edge(t, verts2d):
    return max(np.linalg.norm(verts2d[t[i]] - verts2d[t[(i + 1) % 3]]) for i in range(3))


def _rasterizar_malla(verts2d, faces, scale=5, margin=2.0):
    x_min = verts2d[:, 0].min() - margin
    y_min = verts2d[:, 1].min() - margin
    x_max = verts2d[:, 0].max() + margin
    y_max = verts2d[:, 1].max() + margin
    w = int((x_max - x_min) * scale) + 1
    h = int((y_max - y_min) * scale) + 1
    mask = np.zeros((h, w), dtype=np.uint8)
    for t in faces:
        pts = verts2d[list(t)].copy()
        pts[:, 0] = (pts[:, 0] - x_min) * scale
        pts[:, 1] = (pts[:, 1] - y_min) * scale
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask, (x_min, y_min, scale)


def _reparar_columnas_y_filas(mask, axis='both', frac_alto=0.45, envelope='global',
                              dilatacion_x=4):
    """envelope: 'local' = min/max de las vecinas inmediatas; 'global' = min/max
    de TODAS las cols válidas (para cuando el barril es UNA pieza con corte
    interno, no dos piezas como cabeza+barril)."""
    """Igual que la versión anterior pero devuelve un set de los índices X
    (columnas) que fueron vaciadas o interpoladas. Ese set marca las columnas
    de "hueco" para que afuera podamos sembrar nodos nuevos ahí."""
    cols_reparadas_total = set()
    for ax in (['cols'] if axis == 'cols' else ['cols', 'rows']):
        m = (mask if ax == 'cols' else mask.T).copy()
        h, w = m.shape
        cols_sum = m.sum(axis=0)
        valid = np.where(cols_sum > 0)[0]
        if len(valid) < 2:
            continue
        heights = np.zeros(w, dtype=np.int32)
        for c in valid:
            r = np.where(m[:, c] > 0)[0]
            heights[c] = int(r[-1] - r[0] + 1)
        h_med = int(np.median(heights[valid]))
        umbral_h = max(2, int(frac_alto * h_med))
        cols_anom = set()
        for c in valid:
            if heights[c] < umbral_h:
                m[:, c] = 0
                cols_anom.add(int(c))
        cols_sum = m.sum(axis=0)
        valid = np.where(cols_sum > 0)[0]
        if len(valid) < 2:
            if ax == 'cols':
                mask[:] = m
                cols_reparadas_total |= cols_anom
            else:
                mask[:] = m.T
            continue
        c0, c1 = int(valid[0]), int(valid[-1])
        top = np.full(w, -1, dtype=np.int32)
        bot = np.full(w, -1, dtype=np.int32)
        for c in valid:
            r = np.where(m[:, c] > 0)[0]
            top[c], bot[c] = int(r[0]), int(r[-1])
        # Pre-cálculo del envelope GLOBAL: min top y max bot de todas las cols
        # válidas (NO afectadas por el vaciado anómalo). Sirve cuando el corte
        # parte un barril único: las vecinas inmediatas también están afectadas
        # y el envelope local subestima la altura del barril.
        global_top = int(top[valid].min()) if len(valid) > 0 else 0
        global_bot = int(bot[valid].max()) if len(valid) > 0 else h - 1

        cols_interp = set()
        c = c0 + 1
        while c < c1:
            if top[c] < 0:
                gs = c
                ge = c
                while ge + 1 < c1 and top[ge + 1] < 0:
                    ge += 1
                if envelope == 'global':
                    tk, bk = global_top, global_bot
                else:
                    L, R = gs - 1, ge + 1
                    tL, bL = int(top[L]), int(bot[L])
                    tR, bR = int(top[R]), int(bot[R])
                    tk = min(tL, tR)
                    bk = max(bL, bR)
                for ck in range(gs, ge + 1):
                    if bk >= tk:
                        m[tk:bk + 1, ck] = 255
                        top[ck], bot[ck] = tk, bk
                        cols_interp.add(int(ck))
                c = ge + 1
            else:
                c += 1
        # DILATACIÓN HORIZONTAL: extender la zona reparada `dilatacion_x` cols
        # a cada lado, llevándolas al envelope global. Esto suaviza la
        # transición visual sin alterar columnas lejanas (que pueden ser
        # geometría natural como cuello/cabeza/cola).
        if (cols_interp or cols_anom) and dilatacion_x > 0:
            cols_rep_array = sorted(cols_interp | cols_anom)
            global_top = int(top[valid].min()) if len(valid) else 0
            global_bot = int(bot[valid].max()) if len(valid) else h - 1
            cols_dilat = set()
            for cr in cols_rep_array:
                for delta in range(-dilatacion_x, dilatacion_x + 1):
                    cc = cr + delta
                    if cc < c0 or cc > c1:
                        continue
                    if cc in cols_rep_array or cc in cols_dilat:
                        continue
                    if top[cc] < 0:
                        continue
                    changed = False
                    if top[cc] > global_top:
                        m[global_top:top[cc], cc] = 255
                        top[cc] = global_top
                        changed = True
                    if bot[cc] < global_bot:
                        m[bot[cc] + 1:global_bot + 1, cc] = 255
                        bot[cc] = global_bot
                        changed = True
                    if changed:
                        cols_dilat.add(int(cc))
            if cols_dilat:
                cols_interp |= cols_dilat

        if ax == 'cols':
            mask[:] = m
            cols_reparadas_total |= cols_anom | cols_interp
        else:
            mask[:] = m.T
    return cols_reparadas_total


def reparar(verts3d, faces, colors, factor_largo=2.0, factor_grid=1.0, verbose=True):
    """Detecta y rellena huecos por interpolación columnar de la máscara
    rasterizada, luego re-triangula. Funciona aunque el hueco "toque" el borde
    externo (concavidad pasante)."""
    verts2d = verts3d[:, :2]
    n_orig = len(verts2d)

    # 1. Longitudes de aristas para fijar densidad
    all_edges = set()
    for a, b, c in faces:
        for e in [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))]:
            all_edges.add(e)
    lens = np.array([np.linalg.norm(verts2d[a] - verts2d[b]) for a, b in all_edges])
    median_edge = float(np.median(lens))
    if verbose:
        print(f"[reparar] mediana arista={median_edge:.2f} cm")

    # 2. Filtrar triángulos largos para que NO contribuyan a la rasterización
    threshold = factor_largo * median_edge
    keep = np.array([_max_edge(t, verts2d) <= threshold for t in faces])
    good_faces = faces[keep]
    if verbose:
        print(f"[reparar] {(~keep).sum()}/{len(faces)} triángulos largos descartados de la rasterización")

    # 3. Rasterizar SOLO los triángulos buenos → la máscara queda con el hueco
    SCALE = 5
    mask_short, (x_min, y_min, scale) = _rasterizar_malla(verts2d, good_faces, scale=SCALE)

    # 4. Interpolar columnas vacías (el hueco vertical se rellena entre vecinas)
    cols_reparadas_set = _reparar_columnas_y_filas(mask_short, axis='cols')
    if verbose:
        print(f"[reparar] columnas rellenadas: {len(cols_reparadas_set)}")

    # Limpieza morfológica suave
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_rep = cv2.morphologyEx(mask_short, cv2.MORPH_CLOSE, k, iterations=1)

    # 5. Detectar contorno externo de la máscara reparada
    contours, _ = cv2.findContours(mask_rep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if verbose:
            print("[reparar] sin contorno tras reparación. abortando.")
        return verts3d, faces, colors
    contorno = max(contours, key=cv2.contourArea)
    contorno_xy = contorno.reshape(-1, 2).astype(float)
    contorno_xy[:, 0] = contorno_xy[:, 0] / scale + x_min
    contorno_xy[:, 1] = contorno_xy[:, 1] / scale + y_min

    # 6. Generar nuevos vértices en TODA la zona del hueco para que la malla
    # nueva acompañe la silueta natural del barril (no solo una columna delgada).
    # Para cada columna reparada, sembramos:
    #   - 2 puntos de contorno (top y bot interpolados),
    #   - puntos interiores cada `step` cm a lo largo de y.
    step = median_edge * factor_grid
    new_pts = []
    new_pts_set = set()  # para no duplicar (precision: round a 0.1cm)

    def _add_pt(gx, gy, min_dist=None):
        if min_dist is None:
            min_dist = step * 0.4
        key = (round(gx, 1), round(gy, 1))
        if key in new_pts_set:
            return False
        d_orig = np.linalg.norm(verts2d - np.array([gx, gy]), axis=1).min()
        if d_orig < min_dist:
            return False
        new_pts_set.add(key)
        new_pts.append([gx, gy])
        return True

    # Encontrar top/bot por columna en la máscara reparada
    rep_h, rep_w = mask_rep.shape
    cols_sum = mask_rep.sum(axis=0)
    valid = np.where(cols_sum > 0)[0]
    cols_orig_good_sum = _rasterizar_malla(verts2d, good_faces, scale=scale, margin=2.0)[0]
    H = max(mask_rep.shape[0], cols_orig_good_sum.shape[0])
    W = max(mask_rep.shape[1], cols_orig_good_sum.shape[1])
    mr = np.zeros((H, W), dtype=np.uint8); mr[:mask_rep.shape[0], :mask_rep.shape[1]] = mask_rep
    mo = np.zeros((H, W), dtype=np.uint8); mo[:cols_orig_good_sum.shape[0], :cols_orig_good_sum.shape[1]] = cols_orig_good_sum
    mask_rep = mr

    # Usar las columnas reparadas detectadas durante el rellenado
    cols_huecos = sorted(cols_reparadas_set)

    # Sembrar puntos en cada columna del hueco: top + bot + INTERIORES.
    # Los interiores son CRÍTICOS para que el shell 3D tenga profundidad en
    # la zona reparada — sin ellos, profundidad_eliptica da depth=0 (los nodos
    # top/bot están en y extremas) y el barril aparece "mordido" desde arriba.
    # step_x ≈ step/2 para densidad horizontal; step_y = step para densidad
    # vertical de los interiores.
    step_x = step * 0.6
    step_y_px = max(1, int(step * scale))
    last_gx = -1e9
    for cx_px in cols_huecos:
        gx = cx_px / scale + x_min
        if gx - last_gx < step_x:
            continue
        if cx_px >= W:
            continue
        rows = np.where(mr[:, cx_px] > 0)[0]
        if rows.size == 0:
            continue
        last_gx = gx
        top_y_px, bot_y_px = int(rows[0]), int(rows[-1])
        _add_pt(gx, top_y_px / scale + y_min, min_dist=0.3)  # lomo
        _add_pt(gx, bot_y_px / scale + y_min, min_dist=0.3)  # panza
        for ry_px in range(top_y_px + step_y_px, bot_y_px, step_y_px):
            _add_pt(gx, ry_px / scale + y_min, min_dist=0.3)
    if verbose:
        print(f"[reparar] columnas-hueco detectadas: {len(cols_huecos)}, "
              f"vértices nuevos sembrados: {len(new_pts)}")

    if verbose:
        print(f"[reparar] vértices nuevos totales: {len(new_pts)}")

    if new_pts:
        new_pts_arr = np.array(new_pts)
        verts2d_full = np.vstack([verts2d, new_pts_arr])
    else:
        verts2d_full = verts2d

    # 8. Re-triangular con Delaunay y filtrar por máscara reparada
    tri = Delaunay(verts2d_full)
    final_tris = []
    for s in tri.simplices:
        c = verts2d_full[s].mean(0)
        cx = int((c[0] - x_min) * scale)
        cy = int((c[1] - y_min) * scale)
        if not (0 <= cx < mask_rep.shape[1] and 0 <= cy < mask_rep.shape[0]):
            continue
        if mask_rep[cy, cx] == 0:
            continue
        if _max_edge(s, verts2d_full) > threshold * 1.3:
            continue
        final_tris.append(s)
    final_tris = np.array(final_tris, dtype=int)

    # 9. Reconstruir z y colores
    z_orig = verts3d[:, 2] if verts3d.shape[1] >= 3 else np.zeros(n_orig)
    if len(new_pts) > 0:
        z_new = np.zeros(len(new_pts))
        verts3d_full = np.column_stack([verts2d_full, np.concatenate([z_orig, z_new])])
    else:
        verts3d_full = np.column_stack([verts2d_full, z_orig])

    if colors is not None and len(new_pts) > 0:
        nc = []
        for p in new_pts_arr:
            di = np.linalg.norm(verts2d - p, axis=1)
            nc.append(colors[int(np.argmin(di))])
        colors_full = np.vstack([colors, np.array(nc, dtype=np.uint8)])
    else:
        colors_full = colors

    if verbose:
        print(f"[reparar] resultado: {len(verts3d_full)} verts, {len(final_tris)} tris "
              f"(antes: {n_orig} verts, {len(faces)} tris)")
    return verts3d_full, final_tris, colors_full


def escribir_ply_plano(path, verts3d, colors, faces, comments):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        for c in comments:
            f.write(c)
        f.write("comment Hueco rellenado por reparar_hueco_malla.py\n")
        f.write(f"element vertex {len(verts3d)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for i, p in enumerate(verts3d):
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{p[0]:.2f} {p[1]:.2f} {p[2]:.2f} {r} {g} {b}\n")
            else:
                f.write(f"{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}\n")
        for t in faces:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python reparar_hueco_malla.py input.ply output.ply")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    verts, colors, faces, comments = leer_ply_plano(src)
    print(f"[in]  {src}: {len(verts)} verts, {len(faces)} faces")
    new_v, new_f, new_c = reparar(verts, faces, colors)
    escribir_ply_plano(dst, new_v, new_c, new_f, comments)
    print(f"[out] {dst}: {len(new_v)} verts, {len(new_f)} faces")
