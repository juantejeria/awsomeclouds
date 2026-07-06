"""
Reconstrucción 3D de vaca desde múltiples fotos.
Pipeline con OpenCV puro: Features → Matching → SfM → Nube de puntos → Malla.
"""

import cv2
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from pathlib import Path
import json
import struct
from core.calibracion import DENSIDAD_KG_L


def detectar_y_matchear(img1, img2):
    """Detecta features SIFT y matchea entre dos imágenes."""
    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    if desc1 is None or desc2 is None:
        return [], [], []

    # FLANN matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1, desc2, k=2)

    # Ratio test de Lowe
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    return good, pts1, pts2


def estimar_pose(pts1, pts2, K):
    """Estima la pose relativa entre dos cámaras."""
    E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask_e)
    return R, t, mask_pose


def triangular_puntos(K, R1, t1, R2, t2, pts1, pts2):
    """Triangula puntos 3D desde dos vistas."""
    P1 = K @ np.hstack([R1, t1])
    P2 = K @ np.hstack([R2, t2])

    pts1_h = pts1.T.reshape(2, -1)
    pts2_h = pts2.T.reshape(2, -1)

    points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = (points_4d[:3] / points_4d[3]).T

    return points_3d


def filtrar_puntos(points_3d, max_dist=50):
    """Filtra outliers por distancia al centroide."""
    centroid = np.median(points_3d, axis=0)
    dists = np.linalg.norm(points_3d - centroid, axis=1)
    mask = dists < max_dist
    return points_3d[mask]


def guardar_ply(path, points, colors=None):
    """Guarda nube de puntos como archivo PLY."""
    n = len(points)
    header = f"""ply
format ascii 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(path, 'w') as f:
        f.write(header)
        for i in range(n):
            x, y, z = points[i]
            if colors is not None and i < len(colors):
                r, g, b = colors[i]
            else:
                r, g, b = 139, 90, 43  # marrón vaca
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def guardar_ply_con_malla(path, points, triangles, colors=None):
    """Guarda malla triangulada como archivo PLY."""
    n_verts = len(points)
    n_faces = len(triangles)
    header = f"""ply
format ascii 1.0
element vertex {n_verts}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {n_faces}
property list uchar int vertex_indices
end_header
"""
    with open(path, 'w') as f:
        f.write(header)
        for i in range(n_verts):
            x, y, z = points[i]
            if colors is not None and i < len(colors):
                r, g, b = colors[i]
            else:
                r, g, b = 139, 90, 43
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def obtener_colores(img, pts2d, pts3d_indices):
    """Extrae colores de la imagen para los puntos 3D."""
    h, w = img.shape[:2]
    colors = []
    for pt in pts2d:
        x, y = int(round(pt[0])), int(round(pt[1]))
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        b, g, r = img[y, x]
        colors.append([r, g, b])
    return np.array(colors)


def visualizar_2d(points_3d, output_path, titulo=""):
    """Genera visualización 2D del modelo 3D desde múltiples ángulos."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Modelo 3D - {titulo}', fontsize=14, fontweight='bold')

    # Centrar puntos
    centroid = np.mean(points_3d, axis=0)
    pts = points_3d - centroid

    vistas = [
        ('Vista Lateral', 0, 0),
        ('Vista Frontal', 0, 90),
        ('Vista Superior', 90, 0),
        ('Vista 3D', 30, 45),
    ]

    for idx, (nombre, elev, azim) in enumerate(vistas):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='sienna', s=1, alpha=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(nombre)
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Visualización guardada: {output_path}")


def calcular_volumen_convex_hull(points_3d):
    """Calcula volumen del convex hull de la nube de puntos."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points_3d)
        return hull.volume, hull.area
    except Exception as e:
        print(f"  WARN: No se pudo calcular convex hull: {e}")
        return 0, 0


def fusionar_patas(mask_full, mask_torso):
    """Reduce las patas de la silueta a exactamente 2 (delantera + trasera).

    Cuando la vaca se ve de perfil, las 4 patas (2 near + 2 far) aparecen
    en la silueta. Al espejar, cada una se duplica → 4 por lado.
    Esta función analiza el perfil inferior, detecta los "picos" de patas
    (puntos que bajan más), y si hay más de 2, fusiona pares cercanos.

    Método: para cada columna X, mide cuánto baja la máscara debajo del torso.
    Eso forma un perfil de "profundidad de pata". Los picos en ese perfil son
    las patas. Si hay picos cercanos, se fusionan quedándose con el más profundo.
    """
    mask = mask_full.copy()
    h, w = mask.shape

    # Encontrar borde inferior del torso
    torso_rows = np.where(mask_torso > 0)[0]
    if len(torso_rows) == 0:
        return mask
    torso_bottom = torso_rows.max()

    # Rango X de la máscara completa
    full_cols = np.where(mask > 0)[1]
    if len(full_cols) == 0:
        return mask
    x_min, x_max = full_cols.min(), full_cols.max()

    # Para cada columna X, medir cuánto baja la máscara debajo del torso
    # (= profundidad de la pata en esa columna)
    leg_depth = np.zeros(w)
    bottom_y = np.zeros(w, dtype=int)
    for x in range(x_min, x_max + 1):
        col = mask[torso_bottom:, x]
        rows_on = np.where(col > 0)[0]
        if len(rows_on) > 0:
            leg_depth[x] = rows_on.max()
            bottom_y[x] = torso_bottom + rows_on.max()

    if leg_depth.max() < 5:
        return mask  # Sin patas significativas

    # Suavizar el perfil para evitar ruido
    kernel_size = max(5, int((x_max - x_min) * 0.03))
    if kernel_size % 2 == 0:
        kernel_size += 1
    from scipy.ndimage import uniform_filter1d
    leg_smooth = uniform_filter1d(leg_depth.astype(float), size=kernel_size)

    # Encontrar picos (patas) en el perfil suavizado
    # Un pico es una columna donde leg_smooth es un máximo local
    # y tiene al menos 30% de la profundidad máxima
    threshold = leg_smooth.max() * 0.30
    peaks = []
    for x in range(x_min + 2, x_max - 1):
        if leg_smooth[x] >= threshold:
            if leg_smooth[x] >= leg_smooth[x - 1] and leg_smooth[x] >= leg_smooth[x + 1]:
                # Verificar que es un máximo local real (no meseta)
                if leg_smooth[x] > leg_smooth[max(x_min, x - 5)] or leg_smooth[x] > leg_smooth[min(x_max, x + 5)]:
                    peaks.append((x, leg_smooth[x]))

    if len(peaks) <= 2:
        print(f"  [FusionarPatas] {len(peaks)} picos detectados, no se requiere fusión")
        return mask  # Ya tiene 2 o menos patas, OK

    # Agrupar picos cercanos (patas del mismo par near/far)
    # Distancia mínima entre grupos = 25% del ancho total
    min_group_dist = (x_max - x_min) * 0.25
    groups = [[peaks[0]]]
    for i in range(1, len(peaks)):
        if peaks[i][0] - groups[-1][-1][0] < min_group_dist:
            groups[-1].append(peaks[i])
        else:
            groups.append([peaks[i]])

    print(f"  [FusionarPatas] {len(peaks)} picos → {len(groups)} grupos de patas")

    if len(groups) <= 2:
        # Cada grupo puede tener múltiples picos (near/far del mismo par)
        # Fusionar cada grupo en una sola pata
        for group in groups:
            if len(group) <= 1:
                continue
            # Encontrar el pico más profundo del grupo
            best_peak = max(group, key=lambda p: p[1])
            best_x = best_peak[0]
            # Rango X del grupo
            gx_min = min(p[0] for p in group)
            gx_max = max(p[0] for p in group)

            # Para cada fila en la zona de patas, mantener solo la pata
            # centrada en best_x con el ancho de la pata más ancha
            for y in range(torso_bottom, h):
                row = mask[y, gx_min:gx_max + 1]
                cols_on = np.where(row > 0)[0]
                if len(cols_on) < 2:
                    continue

                # Encontrar segmentos en este rango
                segments = []
                seg_start = cols_on[0]
                for j in range(1, len(cols_on)):
                    if cols_on[j] - cols_on[j - 1] > 1:
                        segments.append((gx_min + seg_start, gx_min + cols_on[j - 1]))
                        seg_start = cols_on[j]
                segments.append((gx_min + seg_start, gx_min + cols_on[-1]))

                if len(segments) > 1:
                    # Múltiples segmentos = patas near/far separadas → rellenar entre ellos
                    fill_start = segments[0][0]
                    fill_end = segments[-1][1]
                    mask[y, fill_start:fill_end + 1] = 255

        # Suavizar
        kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_m, iterations=2)
        print(f"  [FusionarPatas] Fusión completada (dentro de grupos)")
        return mask

    # Más de 2 grupos → hay patas extra
    # Quedarnos con los 2 grupos con picos más profundos
    group_scores = [(i, max(p[1] for p in g)) for i, g in enumerate(groups)]
    group_scores.sort(key=lambda x: x[1], reverse=True)
    keep_indices = set([group_scores[0][0], group_scores[1][0]])

    # Eliminar los grupos que no se mantienen
    for i, group in enumerate(groups):
        if i in keep_indices:
            # Fusionar picos dentro del grupo (near/far)
            if len(group) > 1:
                gx_min = min(p[0] for p in group)
                gx_max = max(p[0] for p in group)
                for y in range(torso_bottom, h):
                    row = mask[y, gx_min:gx_max + 1]
                    cols_on = np.where(row > 0)[0]
                    if len(cols_on) >= 2:
                        mask[y, gx_min + cols_on[0]:gx_min + cols_on[-1] + 1] = 255
        else:
            # Eliminar esta pata: borrar columnas de este grupo debajo del torso
            gx_min = min(p[0] for p in group)
            gx_max = max(p[0] for p in group)
            # No borrar si se superpone con grupo mantenido
            for y in range(torso_bottom, h):
                # Solo borrar si la fila no es parte del torso
                if mask_torso[y, gx_min:gx_max + 1].max() == 0:
                    mask[y, gx_min:gx_max + 1] = 0

    kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_m, iterations=2)
    print(f"  [FusionarPatas] Eliminadas {len(groups) - 2} patas extra, quedan 2 grupos")
    return mask


def modelo_hibrido(frames, masks, cow_height_cm, bboxes=None, masks_full=None, on_progress=None):
    """
    Modelo 3D híbrido: usa el mejor frame (mayor silueta completa),
    genera volumen con profundidad sintética elíptica.
    Colorea el barril (torso) distinto de extremidades/cabeza.
    Calcula peso total y peso del barril por separado.

    frames: list of BGR images
    masks: list of binary masks (torso only)
    cow_height_cm: altura real para escalar
    bboxes: list of YOLO bboxes [x1,y1,x2,y2]
    masks_full: list of full binary masks (animal completo, sin recorte de torso)
    on_progress: callback(step, total, message)

    Returns: dict con métricas o None.
    """
    from scipy.spatial import ConvexHull

    if len(frames) < 1:
        return None

    total_steps = 5
    step = 0

    def progress(s, msg):
        nonlocal step
        step = s
        print(f"  [Híbrido] {msg}")
        if on_progress:
            on_progress(s, total_steps, msg)

    # ── Step 1: Seleccionar mejor frame (mayor área de silueta completa) ──
    progress(1, "Seleccionando mejor frame...")
    # Usar mask_full si disponible para elegir (silueta más grande = más visible)
    select_masks = masks_full if masks_full is not None else masks
    best_idx = 0
    best_area = 0
    for i, m in enumerate(select_masks):
        if m is not None:
            area = np.count_nonzero(m)
            if area > best_area:
                best_area = area
                best_idx = i

    if best_area == 0:
        print("  [Híbrido] ERROR: ningún mask con área > 0")
        return None

    mask_torso = masks[best_idx]
    mask_full = masks_full[best_idx] if masks_full is not None else mask_torso
    frame = frames[best_idx]
    print(f"  [Híbrido] Mejor frame: {best_idx} (área_full={best_area} px)")

    # Fusionar patas duplicadas (near/far) para que el modelo tenga 2 patas por lado
    mask_full = fusionar_patas(mask_full, mask_torso)

    # ── Step 2: Samplear puntos 2D de la silueta COMPLETA ──
    progress(2, "Muestreando silueta completa...")

    # Encontrar contorno de la mask completa
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contorno = max(contours, key=cv2.contourArea)

    # Samplear borde
    c = contorno.reshape(-1, 2)
    n_borde = 120  # más puntos porque es silueta completa
    pts_b = c[::max(1, len(c) // n_borde)]

    # Samplear interior (grid) sobre mask_full
    n_interior = 60
    ys_on, xs_on = np.where(mask_full > 0)
    if len(xs_on) == 0:
        return None

    cols = int(np.sqrt(n_interior) * 1.5) + 2
    rows = int(np.sqrt(n_interior)) + 2
    gx = np.linspace(xs_on.min(), xs_on.max(), cols + 2)[1:-1]
    gy = np.linspace(ys_on.min(), ys_on.max(), rows + 2)[1:-1]
    mgx, mgy = np.meshgrid(gx, gy)
    grid = np.column_stack([mgx.ravel(), mgy.ravel()]).astype(int)
    interior = [pt for pt in grid if 0 <= pt[1] < mask_full.shape[0] and 0 <= pt[0] < mask_full.shape[1] and mask_full[pt[1], pt[0]] > 0]
    pts_i = np.array(interior) if interior else np.array([]).reshape(0, 2)

    puntos_px = np.vstack([pts_b, pts_i]) if len(pts_i) > 0 else pts_b
    puntos_px = np.unique(puntos_px, axis=0)

    if len(puntos_px) < 4:
        return None

    # Clasificar cada punto: barril vs extremidades
    is_barril = np.array([
        mask_torso[min(pt[1], mask_torso.shape[0] - 1), min(pt[0], mask_torso.shape[1] - 1)] > 0
        for pt in puntos_px
    ])
    n_barril = is_barril.sum()
    n_extremidades = len(is_barril) - n_barril
    print(f"  [Híbrido] Puntos: {n_barril} barril + {n_extremidades} extremidades = {len(puntos_px)} total")

    # Triangular
    from scipy.spatial import Delaunay
    tri = Delaunay(puntos_px)
    tris = []
    for s in tri.simplices:
        cx, cy = puntos_px[s].mean(axis=0).astype(int)
        if 0 <= cy < mask_full.shape[0] and 0 <= cx < mask_full.shape[1] and mask_full[cy, cx] > 0:
            tris.append(s)
    tris = np.array(tris) if tris else np.array([]).reshape(0, 3).astype(int)

    # ── Step 3: Escalar a cm y aplicar profundidad elíptica ──
    progress(3, "Generando modelo 3D con profundidad elíptica...")

    # Escalar: usar bbox YOLO (altura completa de la vaca)
    if bboxes is not None and best_idx < len(bboxes):
        bbox = bboxes[best_idx]
        bx1, by1, bx2, by2 = bbox
        bbox_h_px = by2 - by1
    else:
        bbox_ys = np.where(mask_full > 0)[0]
        bbox_xs = np.where(mask_full > 0)[1]
        bbox = [int(bbox_xs.min()), int(bbox_ys.min()), int(bbox_xs.max()), int(bbox_ys.max())]
        bbox_h_px = bbox_ys.max() - bbox_ys.min()

    if bbox_h_px < 10:
        return None

    escala = cow_height_cm / bbox_h_px
    print(f"  [Híbrido] Escala: {escala:.4f} cm/px (bbox_h={bbox_h_px}px, cow_h={cow_height_cm}cm)")
    puntos_cm = puntos_px.astype(float) * escala
    puntos_cm[:, 1] = puntos_cm[:, 1].max() - puntos_cm[:, 1]  # flip Y

    # Profundidad elíptica (fórmula de procesar_frame)
    ys = puntos_cm[:, 1]
    y_min, y_max = ys.min(), ys.max()
    y_range = y_max - y_min if y_max > y_min else 1
    y_center = y_min + y_range * 0.4

    depths = []
    for pt in puntos_cm:
        d = min(abs(pt[1] - y_center) / (y_range * 0.5), 1.0)
        depths.append(y_range * 0.25 * np.sqrt(max(0, 1 - d ** 2)))
    depths = np.array(depths)

    # Espejar: +depth y -depth
    pts_r = np.column_stack([puntos_cm[:, 0], puntos_cm[:, 1], depths])
    pts_l = np.column_stack([puntos_cm[:, 0], puntos_cm[:, 1], -depths])
    pts_3d = np.vstack([pts_r, pts_l])

    # ── Colores: barril = color real, extremidades = tinte azulado ──
    colores = []
    for idx, pt in enumerate(puntos_px):
        py = min(pt[1], frame.shape[0] - 1)
        px = min(pt[0], frame.shape[1] - 1)
        b, g, r = frame[py, px]
        if is_barril[idx]:
            # Barril: color real de la imagen
            colores.append([int(r), int(g), int(b)])
        else:
            # Extremidades/cabeza: tinte azul-gris para distinguir
            gray = int(0.3 * r + 0.5 * g + 0.2 * b)
            colores.append([max(0, gray - 60), max(0, gray - 40), min(255, gray + 120)])
    colores = np.array(colores)
    # Duplicar colores para ambos lados (espejo)
    colors_all = np.vstack([colores, colores])

    # is_barril también duplicado para ambos lados
    is_barril_3d = np.concatenate([is_barril, is_barril])

    # Triángulos duplicados para ambos lados
    n_pts = len(puntos_cm)
    tris_r = tris.copy() if len(tris) > 0 else np.array([]).reshape(0, 3).astype(int)
    tris_l = (tris.copy() + n_pts) if len(tris) > 0 else np.array([]).reshape(0, 3).astype(int)
    if len(tris_l) > 0:
        tris_l = tris_l[:, [0, 2, 1]]  # flip winding
    all_tris = np.vstack([tris_r, tris_l]) if len(tris_r) > 0 else np.array([]).reshape(0, 3).astype(int)

    # ── Step 4: Volumen total y volumen barril ──
    progress(4, "Calculando volumen total y barril...")

    # Volumen TOTAL (animal completo)
    try:
        hull = ConvexHull(pts_3d)
        vol_cm3 = hull.volume
        vol_litros = vol_cm3 / 1000.0
        sup_cm2 = hull.area
    except Exception:
        vol_cm3 = vol_litros = sup_cm2 = 0

    # Volumen BARRIL (solo puntos del torso)
    barril_pts = pts_3d[is_barril_3d]
    vol_barril_cm3 = 0
    vol_barril_litros = 0
    if len(barril_pts) >= 4:
        try:
            hull_b = ConvexHull(barril_pts)
            vol_barril_cm3 = hull_b.volume
            vol_barril_litros = vol_barril_cm3 / 1000.0
        except Exception:
            pass

    x_min, x_max = puntos_cm[:, 0].min(), puntos_cm[:, 0].max()
    y_min2, y_max2 = puntos_cm[:, 1].min(), puntos_cm[:, 1].max()
    largo_cm = x_max - x_min
    alto_modelo_cm = y_max2 - y_min2  # alto de la silueta sampled (< cow_height_cm)
    # El alto REAL del animal es cow_height_cm (de calibración con postes, promedio lineal)
    alto_cm = cow_height_cm
    ancho_cm = depths.max() * 2

    # Alto del barril solamente
    barril_puntos_cm = puntos_cm[is_barril]
    alto_barril_cm = 0
    if len(barril_puntos_cm) > 0:
        alto_barril_cm = barril_puntos_cm[:, 1].max() - barril_puntos_cm[:, 1].min()

    # Peso total y peso barril (densidad ~1.03 kg/L)
    peso_kg = round(float(vol_litros) * DENSIDAD_KG_L, 2)
    peso_barril_kg = round(float(vol_barril_litros) * DENSIDAD_KG_L, 2)

    # ── Step 5: Guardar PLY ──
    progress(5, "Finalizando modelo...")

    print(f"  [Híbrido] Resultado: {len(pts_3d)} pts, vol_total={vol_litros:.1f}L ({peso_kg}kg), "
          f"vol_barril={vol_barril_litros:.1f}L ({peso_barril_kg}kg), "
          f"alto_animal={alto_cm:.1f}cm, alto_barril={alto_barril_cm:.1f}cm, largo={largo_cm:.1f}cm")

    return {
        'points_3d': pts_3d,
        'colors': colors_all,
        'triangles': all_tris,
        'volumen_cm3': float(round(vol_cm3, 1)),
        'volumen_litros': float(round(vol_litros, 1)),
        'superficie_cm2': float(round(sup_cm2, 1)),
        'peso_kg': float(peso_kg),
        'peso_barril_kg': float(peso_barril_kg),
        'volumen_barril_litros': float(round(vol_barril_litros, 1)),
        'alto_cm': float(round(alto_cm, 1)),
        'alto_barril_cm': float(round(alto_barril_cm, 1)),
        'largo_cm': float(round(largo_cm, 1)),
        'ancho_cm': float(round(ancho_cm, 1)),
        'num_points': int(len(pts_3d)),
        'num_pairs': 0,
        'num_triangles': int(len(all_tris)),
        'scale_factor': float(round(escala, 6)),
        'method': 'hibrido',
        'best_frame_idx': int(best_idx),
        'puntos_barril': int(n_barril * 2),
        'puntos_extremidades': int(n_extremidades * 2),
        # Datos para imagen resumen
        '_frame': frame,
        '_bbox': bbox,
        '_mask': mask_full,
        '_mask_torso': mask_torso,
        '_puntos_px': puntos_px,
        '_puntos_cm': puntos_cm,
        '_tris': tris,
        '_colores': colores,
        '_is_barril': is_barril,
    }


def sfm_desde_frames(frames, masks, cow_height_cm, bboxes=None, masks_full=None, on_progress=None):
    """
    Modelo Multi-frame: procesa CADA frame con profundidad elíptica,
    descarta outliers por IQR, y promedia volúmenes/pesos.
    El modelo 3D visual es del frame más cercano a la mediana.

    frames: list of BGR images
    masks: list of binary masks (torso)
    cow_height_cm: altura real para escalar
    bboxes: list of YOLO bboxes [x1,y1,x2,y2]
    masks_full: list of full binary masks (animal completo)
    on_progress: callback(step, total, message)

    Returns: dict con métricas promediadas o None.
    """
    from scipy.spatial import ConvexHull
    import statistics

    if len(frames) < 2:
        return None

    n_frames = len(frames)
    total_steps = n_frames + 2  # +2 for IQR filter + final model
    current_step = 0

    def progress(s, msg):
        print(f"  [Multi] {msg}")
        if on_progress:
            on_progress(s, total_steps, msg)

    # ── Phase 1: Procesar cada frame con profundidad elíptica ──
    per_frame_results = []

    for fi in range(n_frames):
        current_step += 1
        progress(current_step, f"Procesando frame {fi+1}/{n_frames}...")

        # Usar mask_full si disponible, sino mask torso
        mask_use_raw = masks_full[fi] if masks_full is not None and fi < len(masks_full) else masks[fi]
        mask_torso = masks[fi]
        # Fusionar patas duplicadas
        mask_use = fusionar_patas(mask_use_raw, mask_torso) if mask_use_raw is not None else mask_use_raw
        frame = frames[fi]

        if mask_use is None or np.count_nonzero(mask_use) < 100:
            print(f"  [Multi] Frame {fi}: SKIP (mask vacía)")
            continue

        # Contorno de la mask
        contours, _ = cv2.findContours(mask_use, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contorno = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contorno) < 200:
            continue

        # Samplear borde + interior
        c = contorno.reshape(-1, 2)
        pts_b = c[::max(1, len(c) // 100)]

        ys_on, xs_on = np.where(mask_use > 0)
        cols = int(np.sqrt(50) * 1.5) + 2
        rows = int(np.sqrt(50)) + 2
        gx = np.linspace(xs_on.min(), xs_on.max(), cols + 2)[1:-1]
        gy = np.linspace(ys_on.min(), ys_on.max(), rows + 2)[1:-1]
        mgx, mgy = np.meshgrid(gx, gy)
        grid = np.column_stack([mgx.ravel(), mgy.ravel()]).astype(int)
        interior = [pt for pt in grid if 0 <= pt[1] < mask_use.shape[0] and 0 <= pt[0] < mask_use.shape[1] and mask_use[pt[1], pt[0]] > 0]
        pts_i = np.array(interior) if interior else np.array([]).reshape(0, 2)

        puntos_px = np.vstack([pts_b, pts_i]) if len(pts_i) > 0 else pts_b
        puntos_px = np.unique(puntos_px, axis=0)

        if len(puntos_px) < 4:
            continue

        # Triangular
        try:
            tri = Delaunay(puntos_px)
        except Exception:
            continue
        tris = []
        for s in tri.simplices:
            cx, cy = puntos_px[s].mean(axis=0).astype(int)
            if 0 <= cy < mask_use.shape[0] and 0 <= cx < mask_use.shape[1] and mask_use[cy, cx] > 0:
                tris.append(s)
        tris = np.array(tris) if tris else np.array([]).reshape(0, 3).astype(int)

        # Escalar a cm usando bbox YOLO
        if bboxes is not None and fi < len(bboxes):
            bx1, by1, bx2, by2 = bboxes[fi]
            bbox_h_px = by2 - by1
        else:
            bbox_ys = np.where(mask_use > 0)[0]
            bbox_h_px = bbox_ys.max() - bbox_ys.min()

        if bbox_h_px < 10:
            continue

        escala = cow_height_cm / bbox_h_px
        puntos_cm = puntos_px.astype(float) * escala
        puntos_cm[:, 1] = puntos_cm[:, 1].max() - puntos_cm[:, 1]

        # Profundidad elíptica
        ys = puntos_cm[:, 1]
        y_min, y_max = ys.min(), ys.max()
        y_range = y_max - y_min if y_max > y_min else 1
        y_center = y_min + y_range * 0.4

        depths = []
        for pt in puntos_cm:
            d = min(abs(pt[1] - y_center) / (y_range * 0.5), 1.0)
            depths.append(y_range * 0.25 * np.sqrt(max(0, 1 - d ** 2)))
        depths = np.array(depths)

        pts_r = np.column_stack([puntos_cm[:, 0], puntos_cm[:, 1], depths])
        pts_l = np.column_stack([puntos_cm[:, 0], puntos_cm[:, 1], -depths])
        pts_3d = np.vstack([pts_r, pts_l])

        # Volumen total
        try:
            hull = ConvexHull(pts_3d)
            vol_cm3 = hull.volume
            vol_litros = vol_cm3 / 1000.0
            sup_cm2 = hull.area
        except Exception:
            continue

        if vol_litros <= 0:
            continue

        # Volumen barril
        is_barril = np.array([
            mask_torso[min(pt[1], mask_torso.shape[0] - 1), min(pt[0], mask_torso.shape[1] - 1)] > 0
            for pt in puntos_px
        ])
        is_barril_3d = np.concatenate([is_barril, is_barril])
        barril_pts = pts_3d[is_barril_3d]
        vol_barril_litros = 0
        if len(barril_pts) >= 4:
            try:
                vol_barril_litros = ConvexHull(barril_pts).volume / 1000.0
            except Exception:
                pass

        x_range = puntos_cm[:, 0].max() - puntos_cm[:, 0].min()

        per_frame_results.append({
            'frame_idx': fi,
            'vol_litros': vol_litros,
            'vol_barril_litros': vol_barril_litros,
            'sup_cm2': sup_cm2,
            'largo_cm': x_range,
            'ancho_cm': depths.max() * 2,
            'escala': escala,
            'pts_3d': pts_3d,
            'puntos_px': puntos_px,
            'puntos_cm': puntos_cm,
            'tris': tris,
            'depths': depths,
            'is_barril': is_barril,
        })

        print(f"  [Multi] Frame {fi}: vol={vol_litros:.1f}L, barril={vol_barril_litros:.1f}L, largo={x_range:.1f}cm")

    if len(per_frame_results) < 1:
        print(f"  [Multi] ERROR: ningún frame generó resultado")
        return None

    # ── Phase 2: Filtrar outliers por IQR ──
    current_step += 1
    progress(current_step, f"Filtrando outliers ({len(per_frame_results)} frames)...")

    vols = [r['vol_litros'] for r in per_frame_results]
    validos = per_frame_results
    descartados = []

    if len(vols) >= 3:
        q1 = np.percentile(vols, 25)
        q3 = np.percentile(vols, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        validos = [r for r in per_frame_results if lower <= r['vol_litros'] <= upper]
        descartados = [r for r in per_frame_results if r not in validos]

        if descartados:
            print(f"  [Multi] Outliers descartados: {len(descartados)} frames")
            for d in descartados:
                print(f"    Frame {d['frame_idx']}: {d['vol_litros']:.1f}L")

    if len(validos) == 0:
        validos = per_frame_results  # fallback: usar todos

    # ── Phase 3: Promediar y generar modelo del frame mediano ──
    current_step += 1
    progress(current_step, "Generando modelo promediado...")

    avg_vol = statistics.mean([r['vol_litros'] for r in validos])
    avg_vol_barril = statistics.mean([r['vol_barril_litros'] for r in validos])
    avg_sup = statistics.mean([r['sup_cm2'] for r in validos])
    avg_largo = statistics.mean([r['largo_cm'] for r in validos])
    avg_ancho = statistics.mean([r['ancho_cm'] for r in validos])

    # Frame más cercano a la mediana (para visualización 3D)
    vol_median = statistics.median([r['vol_litros'] for r in validos])
    best = min(validos, key=lambda r: abs(r['vol_litros'] - vol_median))
    best_fi = best['frame_idx']

    # Construir modelo 3D del frame mediano
    frame = frames[best_fi]
    puntos_px = best['puntos_px']
    puntos_cm = best['puntos_cm']
    tris = best['tris']
    depths = best['depths']
    is_barril = best['is_barril']
    pts_3d = best['pts_3d']

    # Colores con barril destacado
    colores = []
    mask_torso = masks[best_fi]
    for idx, pt in enumerate(puntos_px):
        py = min(pt[1], frame.shape[0] - 1)
        px = min(pt[0], frame.shape[1] - 1)
        b, g, r = frame[py, px]
        if is_barril[idx]:
            colores.append([int(r), int(g), int(b)])
        else:
            gray = int(0.3 * r + 0.5 * g + 0.2 * b)
            colores.append([max(0, gray - 60), max(0, gray - 40), min(255, gray + 120)])
    colores = np.array(colores)
    colors_all = np.vstack([colores, colores])

    # Triángulos duplicados
    n_pts = len(puntos_cm)
    tris_r = tris.copy() if len(tris) > 0 else np.array([]).reshape(0, 3).astype(int)
    tris_l = (tris.copy() + n_pts) if len(tris) > 0 else np.array([]).reshape(0, 3).astype(int)
    if len(tris_l) > 0:
        tris_l = tris_l[:, [0, 2, 1]]
    all_tris = np.vstack([tris_r, tris_l]) if len(tris_r) > 0 else np.array([]).reshape(0, 3).astype(int)

    # Métricas promediadas
    peso_kg = round(float(avg_vol) * DENSIDAD_KG_L, 2)
    peso_barril_kg = round(float(avg_vol_barril) * DENSIDAD_KG_L, 2)
    alto_cm = cow_height_cm  # altura real calibrada

    vol_std = statistics.stdev([r['vol_litros'] for r in validos]) if len(validos) > 1 else 0

    print(f"  [Multi] Resultado: {len(validos)}/{len(per_frame_results)} frames válidos, "
          f"vol_avg={avg_vol:.1f}L (std={vol_std:.1f}), peso={peso_kg}kg, "
          f"barril={peso_barril_kg}kg, largo={avg_largo:.1f}cm, frame_visual={best_fi}")

    return {
        'points_3d': pts_3d,
        'colors': colors_all,
        'triangles': all_tris,
        'volumen_cm3': float(round(avg_vol * 1000, 1)),
        'volumen_litros': float(round(avg_vol, 1)),
        'superficie_cm2': float(round(avg_sup, 1)),
        'peso_kg': float(peso_kg),
        'peso_barril_kg': float(peso_barril_kg),
        'volumen_barril_litros': float(round(avg_vol_barril, 1)),
        'alto_cm': float(round(alto_cm, 1)),
        'largo_cm': float(round(avg_largo, 1)),
        'ancho_cm': float(round(avg_ancho, 1)),
        'num_points': int(len(pts_3d)),
        'num_pairs': int(len(validos)),
        'num_triangles': int(len(all_tris)),
        'scale_factor': float(round(best['escala'], 6)),
        'method': 'multiframe',
        'frames_total': int(len(per_frame_results)),
        'frames_validos': int(len(validos)),
        'frames_descartados': int(len(descartados)),
        'vol_std': float(round(vol_std, 1)),
        'best_frame_idx': int(best_fi),
        # Datos para imagen resumen
        '_frame': frame,
        '_bbox': bboxes[best_fi] if bboxes is not None and best_fi < len(bboxes) else None,
        '_mask': masks_full[best_fi] if masks_full is not None and best_fi < len(masks_full) else masks[best_fi],
        '_mask_torso': masks[best_fi],
        '_puntos_px': puntos_px,
        '_puntos_cm': puntos_cm,
        '_tris': tris,
        '_colores': colores,
        '_is_barril': is_barril,
    }


def generar_imagen_resumen(result, output_path, vaca_name='vaca'):
    """Genera imagen PNG resumen tipo 2x2 con detección, malla, modelo texturizado y métricas.

    result: dict retornado por modelo_hibrido() o sfm_desde_frames()
    output_path: ruta donde guardar el PNG
    vaca_name: nombre para el título
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    frame = result.get('_frame')
    bbox = result.get('_bbox')
    mask = result.get('_mask')
    mask_torso = result.get('_mask_torso')
    puntos_px = result.get('_puntos_px')
    puntos_cm = result.get('_puntos_cm')
    tris = result.get('_tris')
    colores = result.get('_colores')
    is_barril = result.get('_is_barril')

    if frame is None or puntos_px is None or puntos_cm is None:
        print(f"  [Resumen] Datos insuficientes para generar imagen")
        return False

    method = result.get('method', '')
    peso_kg = result.get('peso_kg', 0)
    peso_barril_kg = result.get('peso_barril_kg', 0)
    escala = result.get('scale_factor', 0)
    largo_cm = result.get('largo_cm', 0)
    alto_cm = result.get('alto_cm', 0)
    ancho_cm = result.get('ancho_cm', 0)
    vol_litros = result.get('volumen_litros', 0)
    vol_cm3 = result.get('volumen_cm3', 0)
    sup_cm2 = result.get('superficie_cm2', 0)
    n_tris = result.get('num_triangles', 0)
    n_pts = result.get('num_points', 0)

    method_label = 'Híbrido' if method == 'hibrido' else 'Multi-frame'
    title = f'Modelo Escalado - {vaca_name} ({peso_kg} kg) [{method_label}]'

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ── Panel 1: Detección + bbox ──
    axes[0, 0].imshow(img_rgb)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='lime', linewidth=2)
        axes[0, 0].add_patch(rect)
    # Dibujar contorno de máscara torso
    if mask_torso is not None:
        contours_t, _ = cv2.findContours(mask_torso, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_t:
            pts = cnt.reshape(-1, 2)
            axes[0, 0].plot(pts[:, 0], pts[:, 1], 'c-', linewidth=1, alpha=0.7)
    axes[0, 0].set_title(f'Detección (escala: {escala:.4f} cm/px)')
    axes[0, 0].axis('off')

    # ── Panel 2: Malla sobre imagen ──
    axes[0, 1].imshow(img_rgb, alpha=0.3)
    if tris is not None and len(tris) > 0:
        axes[0, 1].triplot(puntos_px[:, 0], puntos_px[:, 1], tris, 'b-', linewidth=0.4)
    axes[0, 1].plot(puntos_px[:, 0], puntos_px[:, 1], 'r.', markersize=1.5)
    # Colorear puntos barril vs extremidades
    if is_barril is not None:
        barril_pts = puntos_px[is_barril]
        ext_pts = puntos_px[~is_barril]
        if len(barril_pts) > 0:
            axes[0, 1].plot(barril_pts[:, 0], barril_pts[:, 1], 'm.', markersize=2.5, alpha=0.5)
        if len(ext_pts) > 0:
            axes[0, 1].plot(ext_pts[:, 0], ext_pts[:, 1], 'c.', markersize=2.5, alpha=0.5)
    axes[0, 1].set_title(f'Malla ({n_tris // 2} triángulos × 2 lados)')
    axes[0, 1].axis('off')

    # ── Panel 3: Modelo con textura + medidas ──
    axes[1, 0].set_facecolor('#1a1a2e')
    if tris is not None and len(tris) > 0 and colores is not None:
        polygons = [puntos_cm[t] for t in tris]
        face_colors = [(colores[t] / 255.0).mean(axis=0) for t in tris]
        pc = PolyCollection(polygons, facecolors=face_colors, edgecolors='none', alpha=0.9)
        axes[1, 0].add_collection(pc)
        axes[1, 0].set_xlim(puntos_cm[:, 0].min() - 2, puntos_cm[:, 0].max() + 2)
        axes[1, 0].set_ylim(puntos_cm[:, 1].min() - 2, puntos_cm[:, 1].max() + 2)

    # Flechas de medida
    x_min_cm, x_max_cm = puntos_cm[:, 0].min(), puntos_cm[:, 0].max()
    y_min_cm, y_max_cm = puntos_cm[:, 1].min(), puntos_cm[:, 1].max()
    axes[1, 0].annotate('', xy=(x_max_cm, y_min_cm - 3), xytext=(x_min_cm, y_min_cm - 3),
                        arrowprops=dict(arrowstyle='<->', color='yellow', lw=1.5))
    axes[1, 0].text((x_min_cm + x_max_cm) / 2, y_min_cm - 5, f'{largo_cm:.0f} cm',
                    color='yellow', ha='center', fontsize=10, fontweight='bold')
    axes[1, 0].annotate('', xy=(x_max_cm + 3, y_max_cm), xytext=(x_max_cm + 3, y_min_cm),
                        arrowprops=dict(arrowstyle='<->', color='cyan', lw=1.5))
    axes[1, 0].text(x_max_cm + 5, (y_min_cm + y_max_cm) / 2, f'{alto_cm:.0f} cm',
                    color='cyan', ha='left', fontsize=10, fontweight='bold', rotation=90)
    axes[1, 0].set_title('Modelo con Textura (cm)')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].axis('off')

    # ── Panel 4: Info ──
    axes[1, 1].axis('off')

    extra_lines = ''
    if method == 'multiframe':
        frames_v = result.get('frames_validos', '?')
        frames_t = result.get('frames_total', '?')
        vol_std = result.get('vol_std', 0)
        extra_lines = f"""
    MULTI-FRAME:
    Frames válidos:    {frames_v}/{frames_t}
    Desv. estándar:    {vol_std:.1f} L"""

    info_text = f"""
    {vaca_name.upper()} - MODELO ESCALADO

    Método:            {method_label}
    Peso total:        {peso_kg} kg
    Peso barril:       {peso_barril_kg} kg
    Escala:            {escala:.4f} cm/px

    MEDIDAS:
    Largo:             {largo_cm:.1f} cm
    Alto:              {alto_cm:.1f} cm
    Ancho (prof):      {ancho_cm:.1f} cm
    Volumen:           {vol_cm3:.0f} cm³
                       {vol_litros:.1f} litros
    Superficie:        {sup_cm2:.0f} cm²

    Triángulos:        {n_tris}
    Puntos 3D:         {n_pts}{extra_lines}
    """
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Resumen] Imagen guardada: {output_path}")
    return True


def detectar_y_matchear_masked(img1, img2, mask1, mask2):
    """SIFT feature detection and matching with masks - only detects within cow silhouette."""
    sift = cv2.SIFT_create(nfeatures=5000)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    mask1_u8 = mask1.astype(np.uint8) if mask1.dtype != np.uint8 else mask1
    mask2_u8 = mask2.astype(np.uint8) if mask2.dtype != np.uint8 else mask2
    # Ensure masks are binary 0/255
    mask1_u8 = (mask1_u8 > 0).astype(np.uint8) * 255
    mask2_u8 = (mask2_u8 > 0).astype(np.uint8) * 255

    kp1, desc1 = sift.detectAndCompute(gray1, mask=mask1_u8)
    kp2, desc2 = sift.detectAndCompute(gray2, mask=mask2_u8)

    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return [], np.array([]), np.array([])

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) < 8:
        return [], np.array([]), np.array([])

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    return good, pts1, pts2


def detectar_plano_simetria(cloud):
    """Detects the symmetry plane of a partial point cloud using PCA.

    The cloud is from ~180° coverage (one side of the cow).
    The eigenvector with the smallest eigenvalue = normal of the symmetry plane
    (the "thin" axis of the partial reconstruction).

    Returns: (centroid, normal) where normal is the symmetry plane normal vector.
    """
    centroid = np.mean(cloud, axis=0)
    centered = cloud - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns sorted ascending, so index 0 = smallest eigenvalue
    normal = eigenvectors[:, 0]
    # Ensure normal points in a consistent direction (positive)
    if normal[2] < 0:
        normal = -normal
    print(f"  [Simetria] Eigenvalues: {eigenvalues}")
    print(f"  [Simetria] Normal del plano: {normal}")
    return centroid, normal


def espejar_nube(cloud, colors, centroid, normal):
    """Mirrors the point cloud across the symmetry plane.

    For each point p: p_mirror = p - 2 * dot(p - centroid, normal) * normal
    Returns concatenated original + mirrored cloud and colors.
    """
    normal = normal / np.linalg.norm(normal)
    diff = cloud - centroid
    dots = diff @ normal  # (N,)
    mirrored = cloud - 2.0 * np.outer(dots, normal)

    cloud_full = np.vstack([cloud, mirrored])
    colors_full = np.vstack([colors, colors]) if colors is not None and len(colors) == len(cloud) else None

    print(f"  [Espejo] {len(cloud)} pts originales + {len(mirrored)} espejados = {len(cloud_full)} total")
    return cloud_full, colors_full


def escalar_a_cm(cloud, cow_height_cm):
    """Scales the SfM point cloud from arbitrary units to centimeters.

    Uses PCA to identify the height axis (second eigenvector, vertical),
    then scales so that the extent along that axis equals cow_height_cm.
    """
    centroid = np.mean(cloud, axis=0)
    centered = cloud - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigenvalues sorted ascending: [smallest, middle, largest]
    # For a cow: largest = length (along body), middle = height, smallest = width/depth
    # Height axis = eigenvector with middle eigenvalue (index 1)
    height_axis = eigenvectors[:, 1]

    projections = centered @ height_axis
    extent = projections.max() - projections.min()

    if extent < 1e-6:
        print(f"  [Escalar] WARNING: extent en eje de altura casi cero ({extent})")
        return cloud, 1.0

    scale = cow_height_cm / extent
    cloud_cm = cloud * scale

    print(f"  [Escalar] Extent altura: {extent:.4f} unidades SfM -> {cow_height_cm} cm (scale={scale:.4f})")
    return cloud_cm, scale


def sfm_real_desde_frames(frames, masks, cow_height_cm, bboxes=None, masks_full=None, on_progress=None):
    """Real SfM pipeline: SIFT with masks -> Essential matrix -> Triangulation -> Mirror 180°.

    Uses actual photogrammetry (not synthetic elliptic depth).
    Assumes ~180° coverage from one side, mirrors to complete the cow.

    frames: list of BGR images
    masks: list of binary masks (torso only, for barril volume)
    cow_height_cm: real height for scaling
    bboxes: list of YOLO bboxes [x1,y1,x2,y2]
    masks_full: list of full binary masks (complete animal)
    on_progress: callback(step, total, message)

    Returns: dict compatible with modelo_hibrido()/sfm_desde_frames() or None.
    Falls back to modelo_hibrido() if SfM fails.
    """
    import statistics

    n_frames = len(frames)
    if n_frames < 2:
        print("  [SfM Real] ERROR: se necesitan al menos 2 frames")
        return modelo_hibrido(frames, masks, cow_height_cm, bboxes=bboxes, masks_full=masks_full, on_progress=on_progress)

    total_steps = 7
    step = 0

    def progress(s, msg):
        nonlocal step
        step = s
        print(f"  [SfM Real] {msg}")
        if on_progress:
            on_progress(s, total_steps, msg)

    # ── Step 1: Prepare masks for each frame ──
    progress(1, f"Preparando mascaras para {n_frames} frames...")

    frame_masks = []
    for i in range(n_frames):
        if masks_full is not None and i < len(masks_full) and masks_full[i] is not None:
            fm = masks_full[i]
        elif masks is not None and i < len(masks) and masks[i] is not None:
            fm = masks[i]
        else:
            fm = None
        frame_masks.append(fm)

    # ── Step 2: Estimate camera intrinsics ──
    h, w = frames[0].shape[:2]
    focal = max(h, w) * 1.2
    K = np.array([
        [focal, 0, w / 2.0],
        [0, focal, h / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)

    # ── Step 3: SfM - match pairs and triangulate ──
    progress(2, "Matching y triangulacion entre pares de frames...")

    all_points_3d = []
    all_colors = []
    n_pares_ok = 0

    R_global = np.eye(3)
    t_global = np.zeros((3, 1))

    # Consecutive pairs
    for i in range(n_frames - 1):
        img1, img2 = frames[i], frames[i + 1]
        m1, m2 = frame_masks[i], frame_masks[i + 1]

        if m1 is not None and m2 is not None:
            good, pts1, pts2 = detectar_y_matchear_masked(img1, img2, m1, m2)
        else:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            good, pts1, pts2 = detectar_y_matchear(gray1, gray2)

        if len(good) < 15:
            print(f"    Par [{i}→{i+1}]: {len(good)} matches (SKIP)")
            continue

        R, t, mask_pose = estimar_pose(pts1, pts2, K)
        if R is None:
            continue

        inliers = mask_pose.ravel() > 0
        pts1_in, pts2_in = pts1[inliers], pts2[inliers]

        if inliers.sum() < 8:
            continue

        R2_rel = R
        t2_rel = t
        points_3d = triangular_puntos(K, R_global, t_global, R2_rel, t2_rel, pts1_in, pts2_in)

        # Filter: Z > 0 (in front of camera)
        z_pos = points_3d[:, 2] > 0
        points_3d = points_3d[z_pos]
        pts2_filt = pts2_in[z_pos]

        if len(points_3d) == 0:
            continue

        points_3d = filtrar_puntos(points_3d)
        n_valid = min(len(points_3d), len(pts2_filt))
        points_3d = points_3d[:n_valid]

        colors = obtener_colores(img2, pts2_filt[:n_valid], range(n_valid))
        all_points_3d.append(points_3d)
        all_colors.append(colors[:len(points_3d)])
        n_pares_ok += 1

        print(f"    Par [{i}→{i+1}]: {len(good)} matches, {inliers.sum()} inliers, {len(points_3d)} pts 3D")

        # Update global pose
        R_global = R2_rel @ R_global
        t_global = R2_rel @ t_global + t2_rel

    # Non-consecutive pairs (skip 1)
    progress(3, "Matching pares no consecutivos (skip 1)...")
    for i in range(n_frames - 2):
        img1, img2 = frames[i], frames[i + 2]
        m1, m2 = frame_masks[i], frame_masks[i + 2]

        if m1 is not None and m2 is not None:
            good, pts1, pts2 = detectar_y_matchear_masked(img1, img2, m1, m2)
        else:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            good, pts1, pts2 = detectar_y_matchear(gray1, gray2)

        if len(good) < 15:
            continue

        R, t, mask_pose = estimar_pose(pts1, pts2, K)
        if R is None:
            continue

        inliers = mask_pose.ravel() > 0
        pts1_in, pts2_in = pts1[inliers], pts2[inliers]

        if inliers.sum() < 8:
            continue

        points_3d = triangular_puntos(K, np.eye(3), np.zeros((3, 1)), R, t, pts1_in, pts2_in)
        z_pos = points_3d[:, 2] > 0
        points_3d = points_3d[z_pos]
        pts2_f = pts2_in[z_pos]

        if len(points_3d) == 0:
            continue

        points_3d = filtrar_puntos(points_3d)
        n_valid = min(len(points_3d), len(pts2_f))
        colors = obtener_colores(img2, pts2_f[:n_valid], range(n_valid))
        all_points_3d.append(points_3d[:n_valid])
        all_colors.append(colors[:n_valid])

        print(f"    Par [{i}→{i+2}]: {n_valid} pts")

    # ── Check if we got enough points ──
    if not all_points_3d or n_pares_ok == 0:
        print("  [SfM Real] WARNING: SfM fallo, no hay puntos 3D. Fallback a modelo_hibrido().")
        return modelo_hibrido(frames, masks, cow_height_cm, bboxes=bboxes, masks_full=masks_full, on_progress=on_progress)

    cloud = np.vstack(all_points_3d)
    colors_raw = np.vstack(all_colors)

    # Final outlier filter
    cloud = filtrar_puntos(cloud, max_dist=30)
    colors_raw = colors_raw[:len(cloud)]

    print(f"  [SfM Real] Nube parcial: {len(cloud)} puntos de {n_pares_ok} pares")

    if len(cloud) < 20:
        print("  [SfM Real] WARNING: muy pocos puntos ({len(cloud)}). Fallback a modelo_hibrido().")
        return modelo_hibrido(frames, masks, cow_height_cm, bboxes=bboxes, masks_full=masks_full, on_progress=on_progress)

    # ── Step 4: Detect symmetry plane ──
    progress(4, "Detectando plano de simetria (PCA)...")
    centroid, normal = detectar_plano_simetria(cloud)

    # ── Step 5: Mirror the cloud ──
    progress(5, "Espejando nube de puntos...")
    cloud_full, colors_full = espejar_nube(cloud, colors_raw, centroid, normal)

    # ── Step 6: Scale to cm ──
    progress(6, f"Escalando a centimetros (altura={cow_height_cm} cm)...")
    cloud_cm, scale_factor = escalar_a_cm(cloud_full, cow_height_cm)

    # ── Step 7: Mesh + volume ──
    progress(7, "Generando malla y calculando volumen...")

    # Project to 2D (principal lateral components) for Delaunay
    cent = np.mean(cloud_cm, axis=0)
    centered = cloud_cm - cent
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Use the two largest eigenvectors for 2D projection (lateral view)
    proj_axes = eigenvectors[:, 1:]  # middle and largest
    pts_2d = centered @ proj_axes

    triangles = np.array([]).reshape(0, 3).astype(int)
    try:
        tri = Delaunay(pts_2d)
        valid_tris = []
        for t_idx in tri.simplices:
            p0, p1, p2 = cloud_cm[t_idx]
            edges = [
                np.linalg.norm(p1 - p0),
                np.linalg.norm(p2 - p1),
                np.linalg.norm(p0 - p2),
            ]
            max_edge = max(edges)
            if max_edge < np.percentile([np.linalg.norm(p1 - p0) for p0, p1 in zip(cloud_cm[:-1], cloud_cm[1:])], 95) * 3:
                valid_tris.append(t_idx)
        if valid_tris:
            triangles = np.array(valid_tris)
        print(f"  [SfM Real] Triangulos: {len(triangles)} (de {len(tri.simplices)})")
    except Exception as e:
        print(f"  [SfM Real] Delaunay fallo: {e}")

    # Volume via ConvexHull
    vol_cm3 = 0
    vol_litros = 0
    sup_cm2 = 0
    try:
        hull = ConvexHull(cloud_cm)
        vol_cm3 = hull.volume
        vol_litros = vol_cm3 / 1000.0
        sup_cm2 = hull.area
    except Exception as e:
        print(f"  [SfM Real] ConvexHull fallo: {e}")

    # Dimensions
    x_min, x_max = cloud_cm[:, 0].min(), cloud_cm[:, 0].max()
    y_min, y_max = cloud_cm[:, 1].min(), cloud_cm[:, 1].max()
    z_min, z_max = cloud_cm[:, 2].min(), cloud_cm[:, 2].max()

    # Use PCA to get proper dimensions
    extents = []
    for i in range(3):
        proj = centered @ eigenvectors[:, i]
        extents.append(proj.max() - proj.min())
    # extents[0] = smallest (width/depth), extents[1] = middle (height), extents[2] = largest (length)
    ancho_cm = extents[0] * scale_factor if extents[0] > 0 else z_max - z_min
    alto_cm = cow_height_cm  # by definition
    largo_cm = extents[2] * scale_factor if extents[2] > 0 else x_max - x_min
    # Recalculate from scaled cloud
    largo_cm = max(x_max - x_min, y_max - y_min, z_max - z_min)
    alto_cm_measured = sorted([x_max - x_min, y_max - y_min, z_max - z_min])[1]
    ancho_cm = min(x_max - x_min, y_max - y_min, z_max - z_min)

    peso_kg = round(float(vol_litros) * DENSIDAD_KG_L, 2)

    # Barrel volume (using torso mask from best frame)
    peso_barril_kg = round(float(vol_litros) * 0.7 * DENSIDAD_KG_L, 2)  # estimate: barrel ≈ 70% of total
    vol_barril_litros = round(vol_litros * 0.7, 1)

    # Use best frame for visualization data
    best_idx = 0
    best_area = 0
    select_masks = masks_full if masks_full is not None else masks
    for i, m in enumerate(select_masks):
        if m is not None:
            area = np.count_nonzero(m)
            if area > best_area:
                best_area = area
                best_idx = i

    frame = frames[best_idx]
    mask_full_best = masks_full[best_idx] if masks_full is not None and best_idx < len(masks_full) else masks[best_idx]
    mask_torso_best = masks[best_idx] if best_idx < len(masks) else None
    bbox_best = bboxes[best_idx] if bboxes is not None and best_idx < len(bboxes) else None

    # Sample 2D points for visualization (from best frame mask)
    puntos_px_vis = None
    puntos_cm_vis = None
    tris_vis = None
    colores_vis = None
    is_barril_vis = None

    if mask_full_best is not None:
        contours, _ = cv2.findContours(mask_full_best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contorno = max(contours, key=cv2.contourArea)
            c = contorno.reshape(-1, 2)
            pts_b = c[::max(1, len(c) // 100)]
            ys_on, xs_on = np.where(mask_full_best > 0)
            if len(xs_on) > 0:
                cols_g = int(np.sqrt(50) * 1.5) + 2
                rows_g = int(np.sqrt(50)) + 2
                gx = np.linspace(xs_on.min(), xs_on.max(), cols_g + 2)[1:-1]
                gy = np.linspace(ys_on.min(), ys_on.max(), rows_g + 2)[1:-1]
                mgx, mgy = np.meshgrid(gx, gy)
                grid = np.column_stack([mgx.ravel(), mgy.ravel()]).astype(int)
                interior = [pt for pt in grid if 0 <= pt[1] < mask_full_best.shape[0] and 0 <= pt[0] < mask_full_best.shape[1] and mask_full_best[pt[1], pt[0]] > 0]
                pts_i = np.array(interior) if interior else np.array([]).reshape(0, 2)
                puntos_px_vis = np.vstack([pts_b, pts_i]) if len(pts_i) > 0 else pts_b
                puntos_px_vis = np.unique(puntos_px_vis, axis=0)

                if len(puntos_px_vis) >= 4:
                    if bbox_best is not None:
                        bx1, by1, bx2, by2 = bbox_best
                        bbox_h_px = by2 - by1
                    else:
                        bbox_h_px = ys_on.max() - ys_on.min()
                    if bbox_h_px > 10:
                        esc = cow_height_cm / bbox_h_px
                        puntos_cm_vis = puntos_px_vis.astype(float) * esc
                        puntos_cm_vis[:, 1] = puntos_cm_vis[:, 1].max() - puntos_cm_vis[:, 1]

                        try:
                            tri_vis = Delaunay(puntos_px_vis)
                            tris_list = []
                            for s in tri_vis.simplices:
                                cx, cy = puntos_px_vis[s].mean(axis=0).astype(int)
                                if 0 <= cy < mask_full_best.shape[0] and 0 <= cx < mask_full_best.shape[1] and mask_full_best[cy, cx] > 0:
                                    tris_list.append(s)
                            tris_vis = np.array(tris_list) if tris_list else np.array([]).reshape(0, 3).astype(int)
                        except Exception:
                            tris_vis = np.array([]).reshape(0, 3).astype(int)

                        colores_vis = []
                        is_barril_list = []
                        for pt in puntos_px_vis:
                            py = min(pt[1], frame.shape[0] - 1)
                            px = min(pt[0], frame.shape[1] - 1)
                            b, g, r = frame[py, px]
                            colores_vis.append([int(r), int(g), int(b)])
                            if mask_torso_best is not None:
                                is_barril_list.append(mask_torso_best[min(pt[1], mask_torso_best.shape[0]-1), min(pt[0], mask_torso_best.shape[1]-1)] > 0)
                            else:
                                is_barril_list.append(True)
                        colores_vis = np.array(colores_vis)
                        is_barril_vis = np.array(is_barril_list)

    print(f"  [SfM Real] Resultado: {len(cloud_cm)} pts, vol={vol_litros:.1f}L, peso={peso_kg}kg, "
          f"largo={largo_cm:.1f}cm, alto={alto_cm:.1f}cm, ancho={ancho_cm:.1f}cm")

    return {
        'points_3d': cloud_cm,
        'colors': colors_full if colors_full is not None else colors_raw,
        'triangles': triangles,
        'volumen_cm3': float(round(vol_cm3, 1)),
        'volumen_litros': float(round(vol_litros, 1)),
        'superficie_cm2': float(round(sup_cm2, 1)),
        'peso_kg': float(peso_kg),
        'peso_barril_kg': float(peso_barril_kg),
        'volumen_barril_litros': float(vol_barril_litros),
        'alto_cm': float(round(alto_cm, 1)),
        'largo_cm': float(round(largo_cm, 1)),
        'ancho_cm': float(round(ancho_cm, 1)),
        'num_points': int(len(cloud_cm)),
        'num_pairs': int(n_pares_ok),
        'num_triangles': int(len(triangles)),
        'scale_factor': float(round(scale_factor, 6)),
        'method': 'sfm_real',
        'best_frame_idx': int(best_idx),
        # Visualization data for generar_imagen_resumen
        '_frame': frame,
        '_bbox': bbox_best,
        '_mask': mask_full_best,
        '_mask_torso': mask_torso_best,
        '_puntos_px': puntos_px_vis,
        '_puntos_cm': puntos_cm_vis,
        '_tris': tris_vis,
        '_colores': colores_vis,
        '_is_barril': is_barril_vis,
    }


def main():
    project_dir = Path(__file__).resolve().parents[1]
    fotos_dir = project_dir / "checkpoints" / "dataset" / "modelo" / "vaca1"
    output_dir = project_dir / "reconstruccion_3d_vaca1"
    output_dir.mkdir(exist_ok=True)

    fotos = sorted([f for f in fotos_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    print(f"Fotos: {len(fotos)}\n")

    # Cargar imágenes
    imagenes = []
    for f in fotos:
        img = cv2.imread(str(f))
        if img is not None:
            imagenes.append((f.name, img))
            print(f"  Cargada: {f.name} ({img.shape[1]}x{img.shape[0]})")

    if len(imagenes) < 2:
        print("ERROR: Se necesitan al menos 2 imágenes")
        return

    # Matriz intrínseca estimada (asumimos cámara genérica)
    # focal_length ≈ max(width, height) * 1.2
    h, w = imagenes[0][1].shape[:2]
    focal = max(h, w) * 1.2
    K = np.array([
        [focal, 0, w / 2],
        [0, focal, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    print(f"\n  Focal estimada: {focal:.0f} px")
    print(f"  Centro óptico: ({w/2:.0f}, {h/2:.0f})")

    # ========================================
    # SfM incremental: par a par
    # ========================================
    print(f"\n{'='*60}")
    print("  RECONSTRUCCIÓN 3D")
    print(f"{'='*60}")

    all_points_3d = []
    all_colors = []
    n_pares = 0

    # Cámara 1: identidad
    R_global = np.eye(3)
    t_global = np.zeros((3, 1))

    for i in range(len(imagenes) - 1):
        name1, img1 = imagenes[i]
        name2, img2 = imagenes[i + 1]

        print(f"\n  Par [{i+1}]: {name1} ↔ {name2}")

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        good, pts1, pts2 = detectar_y_matchear(gray1, gray2)
        print(f"    Matches: {len(good)}")

        if len(good) < 20:
            print(f"    SKIP: pocos matches")
            continue

        R, t, mask = estimar_pose(pts1, pts2, K)
        if R is None:
            print(f"    SKIP: no se pudo estimar pose")
            continue

        # Filtrar por mask de pose
        inliers = mask.ravel() > 0
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]
        print(f"    Inliers: {inliers.sum()}")

        if inliers.sum() < 10:
            print(f"    SKIP: pocos inliers")
            continue

        # Triangular
        R2_rel = R
        t2_rel = t

        points_3d = triangular_puntos(K, R_global, t_global, R2_rel, t2_rel, pts1_in, pts2_in)

        # Filtrar puntos con Z positivo (delante de la cámara)
        z_positive = points_3d[:, 2] > 0
        points_3d = points_3d[z_positive]
        pts2_filtered = pts2_in[z_positive]

        if len(points_3d) == 0:
            print(f"    SKIP: no hay puntos con Z>0")
            continue

        # Filtrar outliers
        points_3d = filtrar_puntos(points_3d)
        n_valid = min(len(points_3d), len(pts2_filtered))
        points_3d = points_3d[:n_valid]
        pts2_filtered = pts2_filtered[:n_valid]

        # Colores
        colors = obtener_colores(img2, pts2_filtered, range(len(pts2_filtered)))

        all_points_3d.append(points_3d)
        all_colors.append(colors[:len(points_3d)])
        n_pares += 1

        print(f"    Puntos 3D: {len(points_3d)}")

        # Actualizar pose global
        R_global = R2_rel @ R_global
        t_global = R2_rel @ t_global + t2_rel

    # También hacer matching entre pares no consecutivos (saltar 1)
    print(f"\n  Pares no consecutivos (skip 1):")
    for i in range(len(imagenes) - 2):
        name1, img1 = imagenes[i]
        name2, img2 = imagenes[i + 2]

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        good, pts1, pts2 = detectar_y_matchear(gray1, gray2)

        if len(good) < 20:
            continue

        R, t, mask = estimar_pose(pts1, pts2, K)
        if R is None:
            continue

        inliers = mask.ravel() > 0
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        if inliers.sum() < 10:
            continue

        points_3d = triangular_puntos(K, np.eye(3), np.zeros((3, 1)), R, t, pts1_in, pts2_in)
        z_positive = points_3d[:, 2] > 0
        points_3d = points_3d[z_positive]
        pts2_f = pts2_in[z_positive]

        if len(points_3d) == 0:
            continue

        points_3d = filtrar_puntos(points_3d)
        n_valid = min(len(points_3d), len(pts2_f))
        colors = obtener_colores(img2, pts2_f[:n_valid], range(n_valid))

        all_points_3d.append(points_3d[:n_valid])
        all_colors.append(colors[:n_valid])

        print(f"    {name1} ↔ {name2}: {n_valid} puntos")

    if not all_points_3d:
        print("\nERROR: No se pudieron generar puntos 3D.")
        print("  Las fotos pueden no tener suficiente variación de ángulo.")
        return

    # Combinar toda la nube
    cloud = np.vstack(all_points_3d)
    colors_all = np.vstack(all_colors)
    print(f"\n  NUBE TOTAL: {len(cloud)} puntos 3D")

    # Filtrado final
    cloud = filtrar_puntos(cloud, max_dist=30)
    colors_all = colors_all[:len(cloud)]
    print(f"  Después de filtrar outliers: {len(cloud)} puntos")

    # ========================================
    # Guardar nube de puntos PLY
    # ========================================
    ply_cloud = output_dir / "nube_puntos_vaca1.ply"
    guardar_ply(str(ply_cloud), cloud, colors_all)
    print(f"\n  Nube de puntos: {ply_cloud}")

    # ========================================
    # Generar malla triangulada
    # ========================================
    print(f"\n  Generando malla triangulada...")
    try:
        # Proyectar a 2D para Delaunay (usar X, Y)
        pts_2d = cloud[:, :2]
        tri = Delaunay(pts_2d)

        # Filtrar triángulos muy grandes (outliers)
        triangles = tri.simplices
        valid_tris = []
        for t_idx in triangles:
            p0, p1, p2 = cloud[t_idx]
            edges = [
                np.linalg.norm(p1 - p0),
                np.linalg.norm(p2 - p1),
                np.linalg.norm(p0 - p2),
            ]
            if max(edges) < 5:  # filtro de distancia máxima de arista
                valid_tris.append(t_idx)

        valid_tris = np.array(valid_tris)
        print(f"  Triángulos: {len(valid_tris)} (de {len(triangles)} totales)")

        # Guardar malla
        mesh_ply = output_dir / "malla_vaca1.ply"
        guardar_ply_con_malla(str(mesh_ply), cloud, valid_tris, colors_all)
        print(f"  Malla: {mesh_ply}")

    except Exception as e:
        print(f"  WARN: No se pudo generar malla: {e}")

    # ========================================
    # Volumen (convex hull)
    # ========================================
    volumen, superficie = calcular_volumen_convex_hull(cloud)
    print(f"\n  Volumen (convex hull): {volumen:.2f} unidades³")
    print(f"  Superficie: {superficie:.2f} unidades²")

    # ========================================
    # Visualización 2D del modelo 3D
    # ========================================
    vis_path = output_dir / "modelo_3d_vistas.png"
    visualizar_2d(cloud, str(vis_path), "Vaca 1 - 262 kg")

    # ========================================
    # Resumen JSON
    # ========================================
    resumen = {
        'vaca': 'vaca1',
        'peso_kg': 262,
        'fotos_usadas': len(imagenes),
        'pares_procesados': n_pares,
        'puntos_3d': len(cloud),
        'volumen_convex_hull': round(volumen, 2),
        'superficie_convex_hull': round(superficie, 2),
        'archivos': {
            'nube_puntos': str(ply_cloud),
            'malla': str(output_dir / "malla_vaca1.ply"),
            'visualizacion': str(vis_path),
        }
    }
    json_path = output_dir / "resumen_vaca1.json"
    with open(json_path, 'w') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  LISTO - Modelo 3D de Vaca 1 (262 kg)")
    print(f"{'='*60}")
    print(f"  Archivos en: {output_dir}/")
    print(f"    - nube_puntos_vaca1.ply  (abrir con MeshLab/Blender)")
    print(f"    - malla_vaca1.ply        (malla triangulada)")
    print(f"    - modelo_3d_vistas.png   (preview 2D)")
    print(f"    - resumen_vaca1.json     (métricas)")


if __name__ == '__main__':
    main()
