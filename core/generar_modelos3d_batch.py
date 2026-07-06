"""
Generador batch de modelos 3D para todas las vacas en checkpoints/modelo3d/.
Procesa todas las fotos de cada vaca, calcula volumen por foto,
descarta automáticamente las fotos outliers usando IQR, y genera
el modelo final con la mejor foto válida.
"""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from scipy.spatial import Delaunay, ConvexHull
from pathlib import Path
from core.breed_coefficients import get_estimated_height
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


# ═══════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════
# Escala por defecto: se usa get_estimated_height(category, age) cuando hay datos,
# o este fallback genérico (novillo adulto) cuando no hay info.
ALTO_ESTIMADO_CM = 120.0
MIN_FOTOS = 2  # mínimo de fotos válidas para generar modelo


# ── Detección ──

def detectar_vaca(img, cow_model, coco_model):
    results = cow_model(img, conf=0.15, verbose=False)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        areas = [(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]) for b in boxes]
        return boxes[int(np.argmax(areas))].xyxy[0].cpu().numpy().astype(int)
    results = coco_model(img, conf=0.2, classes=[19], verbose=False)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        areas = [(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]) for b in boxes]
        return boxes[int(np.argmax(areas))].xyxy[0].cpu().numpy().astype(int)
    return None


# ── Segmentación ──

def segmentar(img, bbox):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = 10
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(w, x2+pad), min(h, y2+pad)

    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
    cv2.grabCut(img, mask, (x1,y1,x2-x1,y2-y1), bgd, fgd, 10, cv2.GC_INIT_WITH_RECT)
    mask_fg = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    m = np.zeros_like(mask_fg)
    cv2.drawContours(m, [c], -1, 255, -1)
    return m, c


# ── Recorte de torso (elimina cabeza/cuello y patas) ──

def recortar_torso(mask, bbox):
    """Aísla el torso/barril de la vaca eliminando cabeza/cuello y patas.

    Analiza el ancho de la máscara fila por fila. El torso es la zona
    donde el ancho es >40% del ancho máximo. Cabeza/cuello y patas son
    extensiones finas que se eliminan.

    Returns: (mask_torso, contorno_torso) o (None, None) si falla.
    """
    x1, y1, x2, y2 = bbox
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    if bbox_h < 20 or bbox_w < 20:
        return mask, None

    # ── 1. Análisis de ancho por fila (para cortar patas) ──
    # Para cada fila Y, medir el ancho de la máscara
    ys_range = range(y1, min(y2, mask.shape[0]))
    row_widths = []
    for y in ys_range:
        row = mask[y, x1:x2]
        cols = np.where(row > 0)[0]
        w = (cols[-1] - cols[0] + 1) if len(cols) > 0 else 0
        row_widths.append(w)
    row_widths = np.array(row_widths)

    if len(row_widths) == 0 or row_widths.max() == 0:
        return mask, None

    max_width = row_widths.max()
    threshold = max_width * 0.50  # 50% del ancho máximo = límite torso (patas más altas)

    # Encontrar zona del torso: filas consecutivas con ancho > threshold
    is_body = row_widths > threshold
    # Buscar la secuencia más larga de filas "body"
    body_start = None
    body_end = None
    cur_start = None
    best_len = 0
    for i, b in enumerate(is_body):
        if b:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                length = i - cur_start
                if length > best_len:
                    best_len = length
                    body_start = cur_start
                    body_end = i
                cur_start = None
    if cur_start is not None:
        length = len(is_body) - cur_start
        if length > best_len:
            body_start = cur_start
            body_end = len(is_body)

    if body_start is None:
        return mask, None

    # Convertir a coordenadas absolutas Y
    torso_y_top = y1 + body_start
    torso_y_bottom = y1 + body_end

    # ── 2. Análisis de alto por columna (para cortar cabeza/cuello) ──
    # Para cada columna X, medir el alto de la máscara dentro de la zona torso
    col_heights = []
    for xi in range(x1, min(x2, mask.shape[1])):
        col = mask[torso_y_top:torso_y_bottom, xi]
        rows_on = np.where(col > 0)[0]
        h = (rows_on[-1] - rows_on[0] + 1) if len(rows_on) > 0 else 0
        col_heights.append(h)
    col_heights = np.array(col_heights)

    if len(col_heights) == 0 or col_heights.max() == 0:
        return mask, None

    max_height = col_heights.max()
    col_threshold = max_height * 0.35  # columnas finas = cuello/cabeza

    # Encontrar la secuencia más larga de columnas "body"
    is_body_col = col_heights > col_threshold
    col_start = None
    col_end = None
    cur_start = None
    best_len = 0
    for i, b in enumerate(is_body_col):
        if b:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                length = i - cur_start
                if length > best_len:
                    best_len = length
                    col_start = cur_start
                    col_end = i
                cur_start = None
    if cur_start is not None:
        length = len(is_body_col) - cur_start
        if length > best_len:
            col_start = cur_start
            col_end = len(is_body_col)

    if col_start is None:
        return mask, None

    torso_x_left = x1 + col_start
    torso_x_right = x1 + col_end

    # ── 3. Crear máscara del torso ──
    mask_torso = np.zeros_like(mask)
    mask_torso[torso_y_top:torso_y_bottom, torso_x_left:torso_x_right] = \
        mask[torso_y_top:torso_y_bottom, torso_x_left:torso_x_right]

    # Limpiar con morphología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_torso = cv2.morphologyEx(mask_torso, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_torso, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    c = max(contours, key=cv2.contourArea)
    # Verificar que el torso tiene un área razonable (>30% del original)
    area_torso = cv2.contourArea(c)
    area_original = np.count_nonzero(mask)
    if area_torso < area_original * 0.20:
        return mask, None  # Recorte demasiado agresivo, usar original

    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [c], -1, 255, -1)
    return final_mask, c


# ── Muestreo y triangulación ──

def samplear(contorno, mask, n_borde=80, n_interior=40):
    c = contorno.reshape(-1, 2)
    pts_b = c[::max(1, len(c)//n_borde)]

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return pts_b, np.array([]).reshape(0,2)

    cols = int(np.sqrt(n_interior)*1.5)+2
    rows = int(np.sqrt(n_interior))+2
    gx = np.linspace(xs.min(), xs.max(), cols+2)[1:-1]
    gy = np.linspace(ys.min(), ys.max(), rows+2)[1:-1]
    mx, my = np.meshgrid(gx, gy)
    grid = np.column_stack([mx.ravel(), my.ravel()]).astype(int)
    interior = [pt for pt in grid if 0<=pt[1]<mask.shape[0] and 0<=pt[0]<mask.shape[1] and mask[pt[1],pt[0]]>0]
    return pts_b, np.array(interior) if interior else np.array([]).reshape(0,2)


def triangular(pts_b, pts_i, mask):
    todos = np.vstack([pts_b, pts_i]) if len(pts_i)>0 else pts_b
    todos = np.unique(todos, axis=0)
    if len(todos) < 3:
        return None, None
    tri = Delaunay(todos)
    validos = []
    for s in tri.simplices:
        cx, cy = todos[s].mean(axis=0).astype(int)
        if 0<=cy<mask.shape[0] and 0<=cx<mask.shape[1] and mask[cy,cx]>0:
            validos.append(s)
    return todos, np.array(validos) if validos else np.array([]).reshape(0,3)


# ── PLY export ──

def guardar_ply(path, pts_cm, tris, colores, simetrico=False, escala_info=""):
    if simetrico:
        n = len(pts_cm)
        ys = pts_cm[:, 1]
        y_min, y_max = ys.min(), ys.max()
        y_range = y_max - y_min if y_max > y_min else 1
        y_center = y_min + y_range * 0.4

        depths = []
        for pt in pts_cm:
            d = min(abs(pt[1] - y_center) / (y_range * 0.5), 1.0)
            depths.append(y_range * 0.25 * np.sqrt(max(0, 1 - d**2)))
        depths = np.array(depths)

        pts_r = np.column_stack([pts_cm[:,0], pts_cm[:,1], depths])
        pts_l = np.column_stack([pts_cm[:,0], pts_cm[:,1], -depths])
        all_pts = np.vstack([pts_r, pts_l])
        all_colors = np.vstack([colores, colores]) if colores is not None else None
        tris_r = tris.copy()
        tris_l = tris.copy() + n
        tris_l = tris_l[:, [0,2,1]]
        all_tris = np.vstack([tris_r, tris_l])
    else:
        all_pts = np.column_stack([pts_cm[:,0], pts_cm[:,1], np.zeros(len(pts_cm))])
        all_colors = colores
        all_tris = tris

    # Recorte de cresta del LOMO (mismo criterio que recortar_cresta_ply.py).
    # Solución integral: los modelos nuevos no nacen con cresta. No toca la panza.
    try:
        from core.crest_trim_mesh import trim_top_crest
        all_pts = trim_top_crest(all_pts)
    except Exception as _e:
        print(f"[warn] recorte de cresta omitido: {_e}")

    nv, nf = len(all_pts), len(all_tris)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\n")
        f.write(f"comment Unidades: centimetros\n")
        if escala_info:
            f.write(f"comment {escala_info}\n")
        f.write(f"element vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n")
        for i, pt in enumerate(all_pts):
            r,g,b = (int(all_colors[i][0]), int(all_colors[i][1]), int(all_colors[i][2])) if all_colors is not None and i<len(all_colors) else (139,90,43)
            f.write(f"{pt[0]:.2f} {pt[1]:.2f} {pt[2]:.2f} {r} {g} {b}\n")
        for t in all_tris:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    return all_pts, all_tris


# ── Calcular volumen de un frame en memoria ──

def procesar_frame(img, cow_model, coco_model, cow_height_cm, frame_id="frame"):
    """Procesa una imagen en memoria. Escala por frame = cow_height_cm / bbox_h_px."""
    if img is None:
        return None

    bbox = detectar_vaca(img, cow_model, coco_model)
    if bbox is None:
        return None

    mask_full, contorno_full = segmentar(img, bbox)
    if mask_full is None:
        return None

    area_px_full = cv2.contourArea(contorno_full)
    if area_px_full < 500:
        return None

    x1, y1, x2, y2 = bbox
    bbox_h_px = y2 - y1
    if bbox_h_px < 20:
        return None

    # Recortar torso (sin cabeza/cuello ni patas)
    mask, contorno = recortar_torso(mask_full, bbox)
    if contorno is None:
        mask, contorno = mask_full, contorno_full

    area_px = cv2.contourArea(contorno)
    if area_px < 300:
        return None

    # Escala per-frame: altura calibrada / altura bbox en px
    escala = cow_height_cm / bbox_h_px

    # Triangular
    pts_b, pts_i = samplear(contorno, mask)
    puntos_px, tris = triangular(pts_b, pts_i, mask)
    if puntos_px is None or len(tris) == 0:
        return None

    # Escalar a cm
    puntos_cm = puntos_px.astype(float) * escala
    puntos_cm[:, 1] = puntos_cm[:, 1].max() - puntos_cm[:, 1]

    # Colores
    colores = np.array([
        img[min(pt[1], img.shape[0]-1), min(pt[0], img.shape[1]-1)][::-1]
        for pt in puntos_px
    ])

    # Calcular volumen del modelo simétrico
    n = len(puntos_cm)
    ys = puntos_cm[:, 1]
    y_min, y_max = ys.min(), ys.max()
    y_range = y_max - y_min if y_max > y_min else 1
    y_center = y_min + y_range * 0.4

    depths = []
    for pt in puntos_cm:
        d = min(abs(pt[1] - y_center) / (y_range * 0.5), 1.0)
        depths.append(y_range * 0.25 * np.sqrt(max(0, 1 - d**2)))
    depths = np.array(depths)

    pts_r = np.column_stack([puntos_cm[:,0], puntos_cm[:,1], depths])
    pts_l = np.column_stack([puntos_cm[:,0], puntos_cm[:,1], -depths])
    pts_3d = np.vstack([pts_r, pts_l])

    try:
        hull = ConvexHull(pts_3d)
        vol_cm3 = hull.volume
        vol_litros = vol_cm3 / 1000.0
        sup_cm2 = hull.area
    except:
        vol_cm3 = vol_litros = sup_cm2 = 0

    x_min, x_max = puntos_cm[:, 0].min(), puntos_cm[:, 0].max()
    y_min2, y_max2 = puntos_cm[:, 1].min(), puntos_cm[:, 1].max()
    largo_cm = x_max - x_min
    alto_cm = y_max2 - y_min2
    area_cm2 = area_px * (escala ** 2)

    return {
        'frame_id': frame_id,
        'img': img,
        'bbox': bbox,
        'bbox_h_px': bbox_h_px,
        'mask': mask,
        'contorno': contorno,
        'puntos_px': puntos_px,
        'puntos_cm': puntos_cm,
        'tris': tris,
        'colores': colores,
        'escala': escala,
        'area_px': int(area_px),
        'area_cm2': round(area_cm2, 1),
        'largo_cm': round(largo_cm, 1),
        'alto_cm': round(alto_cm, 1),
        'volumen_cm3': round(vol_cm3, 1),
        'volumen_litros': round(vol_litros, 1),
        'superficie_cm2': round(sup_cm2, 1),
        'num_triangulos': len(tris),
    }


# ── Calcular volumen de una foto ──

def procesar_foto(foto_path, cow_model, coco_model, cow_height_cm=None):
    """Procesa una foto y retorna métricas + datos para modelo, o None si falla."""
    img = cv2.imread(str(foto_path))
    if img is None:
        return None

    bbox = detectar_vaca(img, cow_model, coco_model)
    if bbox is None:
        return None

    mask_full, contorno_full = segmentar(img, bbox)
    if mask_full is None:
        return None

    area_px_full = cv2.contourArea(contorno_full)
    if area_px_full < 500:
        return None

    x1, y1, x2, y2 = bbox
    bbox_h_px = y2 - y1
    if bbox_h_px < 20:
        return None

    # Recortar torso (sin cabeza/cuello ni patas)
    mask, contorno = recortar_torso(mask_full, bbox)
    if contorno is None:
        mask, contorno = mask_full, contorno_full

    area_px = cv2.contourArea(contorno)
    if area_px < 300:
        return None

    # Escala por alto de la vaca (real si se provee, sino estimada)
    alto_usado = cow_height_cm if cow_height_cm is not None else ALTO_ESTIMADO_CM
    escala = alto_usado / bbox_h_px

    # Triangular
    pts_b, pts_i = samplear(contorno, mask)
    puntos_px, tris = triangular(pts_b, pts_i, mask)
    if puntos_px is None or len(tris) == 0:
        return None

    # Escalar a cm
    puntos_cm = puntos_px.astype(float) * escala
    puntos_cm[:, 1] = puntos_cm[:, 1].max() - puntos_cm[:, 1]

    # Colores
    colores = np.array([
        img[min(pt[1], img.shape[0]-1), min(pt[0], img.shape[1]-1)][::-1]
        for pt in puntos_px
    ])

    # Calcular volumen del modelo simétrico (sin guardar PLY)
    n = len(puntos_cm)
    ys = puntos_cm[:, 1]
    y_min, y_max = ys.min(), ys.max()
    y_range = y_max - y_min if y_max > y_min else 1
    y_center = y_min + y_range * 0.4

    depths = []
    for pt in puntos_cm:
        d = min(abs(pt[1] - y_center) / (y_range * 0.5), 1.0)
        depths.append(y_range * 0.25 * np.sqrt(max(0, 1 - d**2)))
    depths = np.array(depths)

    pts_r = np.column_stack([puntos_cm[:,0], puntos_cm[:,1], depths])
    pts_l = np.column_stack([puntos_cm[:,0], puntos_cm[:,1], -depths])
    pts_3d = np.vstack([pts_r, pts_l])

    try:
        hull = ConvexHull(pts_3d)
        vol_cm3 = hull.volume
        vol_litros = vol_cm3 / 1000.0
        sup_cm2 = hull.area
    except:
        vol_cm3 = vol_litros = sup_cm2 = 0

    x_min, x_max = puntos_cm[:, 0].min(), puntos_cm[:, 0].max()
    y_min2, y_max2 = puntos_cm[:, 1].min(), puntos_cm[:, 1].max()
    largo_cm = x_max - x_min
    alto_cm = y_max2 - y_min2
    area_cm2 = area_px * (escala ** 2)

    return {
        'foto': foto_path.name,
        'img': img,
        'bbox': bbox,
        'mask': mask,
        'contorno': contorno,
        'puntos_px': puntos_px,
        'puntos_cm': puntos_cm,
        'tris': tris,
        'colores': colores,
        'escala': escala,
        'area_px': int(area_px),
        'area_cm2': round(area_cm2, 1),
        'largo_cm': round(largo_cm, 1),
        'alto_cm': round(alto_cm, 1),
        'volumen_cm3': round(vol_cm3, 1),
        'volumen_litros': round(vol_litros, 1),
        'superficie_cm2': round(sup_cm2, 1),
        'num_triangulos': len(tris),
    }


# ── Filtro de outliers por IQR ──

def filtrar_outliers(resultados, campo='volumen_litros'):
    """Descarta fotos cuyo valor en `campo` es outlier por IQR."""
    valores = [r[campo] for r in resultados if r[campo] > 0]
    if len(valores) < 3:
        # Con menos de 3 fotos no podemos calcular IQR fiablemente
        return resultados, []

    valores_sorted = sorted(valores)
    q1 = np.percentile(valores_sorted, 25)
    q3 = np.percentile(valores_sorted, 75)
    iqr = q3 - q1

    # Factor 1.5 es estándar para outliers
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    validos = []
    descartados = []
    for r in resultados:
        v = r[campo]
        if v <= 0 or v < lower or v > upper:
            descartados.append(r)
        else:
            validos.append(r)

    return validos, descartados


# ── Visualización por vaca ──

def generar_visualizacion(vaca_name, mejor, puntos_cm, tris, colores, metricas, output_path):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Modelo 3D - {vaca_name}', fontsize=14, fontweight='bold')

    img_rgb = cv2.cvtColor(mejor['img'], cv2.COLOR_BGR2RGB)

    # 1. Original + bbox
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img_rgb)
    x1,y1,x2,y2 = mejor['bbox']
    ax1.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='lime', lw=2))
    ax1.set_title('YOLO Detection')
    ax1.axis('off')

    # 2. Segmentación
    ax2 = fig.add_subplot(2, 3, 2)
    overlay = img_rgb.copy()
    overlay[mejor['mask']>0] = [0,200,0]
    ax2.imshow(cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0))
    ax2.set_title('Segmentacion')
    ax2.axis('off')

    # 3. Malla en px
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(img_rgb, alpha=0.3)
    puntos_px = mejor['puntos_px']
    ax3.triplot(puntos_px[:,0], puntos_px[:,1], tris, 'b-', lw=0.4)
    ax3.plot(puntos_px[:,0], puntos_px[:,1], 'r.', ms=1.5)
    ax3.set_title(f'Malla ({len(tris)} triangulos)')
    ax3.axis('off')

    # 4. Modelo con textura cm
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('black')
    if len(tris) > 0:
        polys = [puntos_cm[t] for t in tris]
        fcolors = [(colores[t]/255.0).mean(axis=0) for t in tris]
        ax4.add_collection(PolyCollection(polys, facecolors=fcolors, edgecolors='none', alpha=0.9))
        x_min, x_max = puntos_cm[:,0].min(), puntos_cm[:,0].max()
        y_min, y_max = puntos_cm[:,1].min(), puntos_cm[:,1].max()
        ax4.set_xlim(x_min-3, x_max+3)
        ax4.set_ylim(y_min-3, y_max+3)
        ax4.annotate('', xy=(x_max, y_min-4), xytext=(x_min, y_min-4),
                     arrowprops=dict(arrowstyle='<->', color='yellow', lw=2))
        ax4.text((x_min+x_max)/2, y_min-7, f'{metricas["largo_cm"]:.0f} cm', color='yellow',
                 ha='center', fontsize=11, fontweight='bold')
        ax4.annotate('', xy=(x_max+4, y_max), xytext=(x_max+4, y_min),
                     arrowprops=dict(arrowstyle='<->', color='cyan', lw=2))
        ax4.text(x_max+6, (y_min+y_max)/2, f'{metricas["alto_cm"]:.0f} cm', color='cyan',
                 ha='left', fontsize=11, fontweight='bold', rotation=90)
    ax4.set_title('Modelo Escalado (cm)')
    ax4.set_aspect('equal')
    ax4.axis('off')

    # 5. Wireframe
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('white')
    ax5.triplot(puntos_cm[:,0], puntos_cm[:,1], tris, color='sienna', linewidth=0.3)
    x_min, x_max = puntos_cm[:,0].min(), puntos_cm[:,0].max()
    y_min, y_max = puntos_cm[:,1].min(), puntos_cm[:,1].max()
    ax5.set_xlim(x_min-3, x_max+3)
    ax5.set_ylim(y_min-3, y_max+3)
    ax5.set_title('Wireframe (cm)')
    ax5.set_aspect('equal')
    ax5.axis('off')

    # 6. Info
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    info = f"""{vaca_name.upper()} - MODELO 3D

Escala estimada:   {metricas['escala']:.4f} cm/px

MEDIDAS:
  Largo:           {metricas['largo_cm']:.1f} cm
  Alto:            {metricas['alto_cm']:.1f} cm
  Area lateral:    {metricas['area_cm2']:.0f} cm2
  Volumen 3D:      {metricas['volumen_cm3']:.0f} cm3
                   {metricas['volumen_litros']:.1f} litros
  Superficie:      {metricas['superficie_cm2']:.0f} cm2

FOTOS:
  Procesadas:      {metricas['fotos_procesadas']}
  Descartadas:     {metricas['fotos_descartadas']}
  Usadas:          {metricas['fotos_validas']}
  Mejor foto:      {metricas['foto_usada']}

Triangulos:        {metricas['num_triangulos']}"""

    ax6.text(0.05, 0.95, info, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def procesar_vaca(vaca_dir, output_dir, cow_model, coco_model, cow_height_cm=None):
    """Procesa todas las fotos de una vaca, descarta outliers, genera modelo final."""
    vaca_name = vaca_dir.name

    fotos = sorted([f for f in vaca_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if len(fotos) == 0:
        return None

    print(f"\n{'='*60}")
    print(f"  {vaca_name.upper()} ({len(fotos)} fotos)")
    print(f"{'='*60}")

    # Procesar todas las fotos
    resultados = []
    for i, foto in enumerate(fotos):
        print(f"  [{i+1:2d}/{len(fotos)}] {foto.name} ... ", end="", flush=True)
        r = procesar_foto(foto, cow_model, coco_model, cow_height_cm=cow_height_cm)
        if r is None:
            print("FALLO (no se detecto vaca o segmentacion fallo)")
            continue
        print(f"Vol: {r['volumen_litros']:.1f} L | Largo: {r['largo_cm']:.0f} cm | Alto: {r['alto_cm']:.0f} cm")
        resultados.append(r)

    if len(resultados) < MIN_FOTOS:
        print(f"\n  SKIP: Solo {len(resultados)} fotos validas (minimo {MIN_FOTOS})")
        return None

    # ── Filtrar outliers por volumen ──
    validos, descartados = filtrar_outliers(resultados, campo='volumen_litros')

    if descartados:
        print(f"\n  OUTLIERS DESCARTADOS ({len(descartados)} fotos):")
        for d in descartados:
            print(f"    - {d['foto']}: {d['volumen_litros']:.1f} L (largo {d['largo_cm']:.0f} cm)")

    if len(validos) == 0:
        print(f"\n  ERROR: Todas las fotos fueron descartadas como outliers")
        return None

    # Estadísticas de las fotos válidas
    vols = [r['volumen_litros'] for r in validos]
    vol_mean = statistics.mean(vols)
    vol_std = statistics.stdev(vols) if len(vols) > 1 else 0

    print(f"\n  FOTOS VALIDAS: {len(validos)}")
    print(f"  Volumen promedio: {vol_mean:.1f} L (std: {vol_std:.1f} L)")

    # Elegir la mejor foto: la que tiene el volumen más cercano a la mediana
    vol_median = statistics.median(vols)
    mejor = min(validos, key=lambda r: abs(r['volumen_litros'] - vol_median))

    print(f"  Volumen mediana: {vol_median:.1f} L")
    print(f"  MEJOR FOTO: {mejor['foto']} (Vol: {mejor['volumen_litros']:.1f} L)")

    # ── Generar modelo final ──
    vaca_output = output_dir / vaca_name
    vaca_output.mkdir(exist_ok=True)

    puntos_cm = mejor['puntos_cm']
    tris = mejor['tris']
    colores = mejor['colores']

    escala_info = f"Escala: {mejor['escala']:.4f} cm/px (estimada por alto={ALTO_ESTIMADO_CM}cm)"

    # PLY lateral
    ply_lat = vaca_output / f"{vaca_name}_lateral.ply"
    guardar_ply(str(ply_lat), puntos_cm, tris, colores, simetrico=False, escala_info=escala_info)

    # PLY 3D simétrico
    ply_3d = vaca_output / f"{vaca_name}_3d.ply"
    pts_3d, tris_3d = guardar_ply(str(ply_3d), puntos_cm, tris, colores, simetrico=True, escala_info=escala_info)

    # Recalcular volumen del modelo final
    try:
        hull = ConvexHull(pts_3d)
        vol_cm3 = hull.volume
        vol_litros = vol_cm3 / 1000.0
        sup_cm2 = hull.area
    except:
        vol_cm3 = vol_litros = sup_cm2 = 0

    x_min, x_max = puntos_cm[:,0].min(), puntos_cm[:,0].max()
    y_min, y_max = puntos_cm[:,1].min(), puntos_cm[:,1].max()
    largo_cm = x_max - x_min
    alto_cm = y_max - y_min
    area_cm2 = mejor['area_px'] * (mejor['escala'] ** 2)

    metricas = {
        'escala': mejor['escala'],
        'largo_cm': round(largo_cm, 1),
        'alto_cm': round(alto_cm, 1),
        'area_cm2': round(area_cm2, 1),
        'volumen_cm3': round(vol_cm3, 1),
        'volumen_litros': round(vol_litros, 1),
        'superficie_cm2': round(sup_cm2, 1),
        'num_triangulos': len(tris),
        'foto_usada': mejor['foto'],
        'fotos_procesadas': len(resultados),
        'fotos_descartadas': len(descartados),
        'fotos_validas': len(validos),
    }

    # Visualización
    vis_path = vaca_output / f"{vaca_name}_modelo.png"
    generar_visualizacion(vaca_name, mejor, puntos_cm, tris, colores, metricas, vis_path)

    # JSON resumen
    resumen = {
        'vaca': vaca_name,
        'escala_cm_px': round(mejor['escala'], 6),
        'largo_cm': metricas['largo_cm'],
        'alto_cm': metricas['alto_cm'],
        'area_lateral_cm2': metricas['area_cm2'],
        'volumen_cm3': metricas['volumen_cm3'],
        'volumen_litros': metricas['volumen_litros'],
        'superficie_cm2': metricas['superficie_cm2'],
        'num_triangulos': metricas['num_triangulos'],
        'foto_usada': metricas['foto_usada'],
        'fotos_procesadas': metricas['fotos_procesadas'],
        'fotos_descartadas': metricas['fotos_descartadas'],
        'fotos_validas': metricas['fotos_validas'],
        'volumen_promedio_litros': round(vol_mean, 1),
        'volumen_std_litros': round(vol_std, 1),
        'volumen_mediana_litros': round(vol_median, 1),
        'fotos_descartadas_detalle': [
            {'foto': d['foto'], 'volumen_litros': d['volumen_litros'],
             'largo_cm': d['largo_cm'], 'razon': 'outlier_iqr'}
            for d in descartados
        ],
        'resultados_por_foto': [
            {'foto': r['foto'], 'volumen_litros': r['volumen_litros'],
             'largo_cm': r['largo_cm'], 'alto_cm': r['alto_cm'],
             'area_cm2': r['area_cm2'], 'descartada': r in descartados}
            for r in resultados
        ],
    }
    with open(vaca_output / f"{vaca_name}_resumen.json", 'w') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\n  RESULTADO {vaca_name.upper()}:")
    print(f"    Volumen:    {vol_litros:.1f} litros ({vol_cm3:.0f} cm3)")
    print(f"    Largo:      {largo_cm:.1f} cm")
    print(f"    Alto:       {alto_cm:.1f} cm")
    print(f"    Superficie: {sup_cm2:.0f} cm2")
    print(f"    Archivos:   {vaca_output}/")

    return resumen


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    modelo3d_dir = Path(args.dataset) if args.dataset else project / "checkpoints" / "modelo3d"
    output_dir = Path(args.output) if args.output else project / "output_modelos3d_batch"
    output_dir.mkdir(exist_ok=True)

    # Cargar alturas reales si existen
    alturas_path = project / "data" / "alturas_individuos.json"
    alturas = {}
    if alturas_path.exists():
        with open(alturas_path) as f:
            data_alturas = json.load(f)
        dataset_name = modelo3d_dir.name
        key = f'alturas_{dataset_name}_cm'
        if key in data_alturas:
            alturas = data_alturas[key]
            print(f"Alturas reales cargadas: {len(alturas)} individuos (clave: {key})")

    print("Cargando modelos YOLO...")
    cow_model = YOLO(str(project / "models" / "cow.pt"))
    coco_model = YOLO(str(project / "yolov8n.pt"))

    # Encontrar todas las carpetas de vacas que tienen fotos
    vaca_dirs = sorted([
        d for d in modelo3d_dir.iterdir()
        if d.is_dir() and any(f.suffix.lower() in ('.png', '.jpg', '.jpeg') for f in d.iterdir())
    ])

    print(f"\nVacas con fotos: {len(vaca_dirs)}")
    for d in vaca_dirs:
        n_fotos = len([f for f in d.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
        print(f"  - {d.name}: {n_fotos} fotos")

    # Procesar cada vaca
    resumen_global = []
    for vaca_dir in vaca_dirs:
        h = alturas.get(vaca_dir.name)
        resumen = procesar_vaca(vaca_dir, output_dir, cow_model, coco_model, cow_height_cm=h)
        if resumen:
            resumen_global.append(resumen)

    # Resumen global
    print(f"\n\n{'#'*60}")
    print(f"  RESUMEN GLOBAL - {len(resumen_global)} vacas procesadas")
    print(f"{'#'*60}")
    print(f"\n  {'Vaca':<20} {'Vol (L)':>10} {'Largo':>10} {'Alto':>10} {'Fotos':>8} {'Desc.':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in resumen_global:
        print(f"  {r['vaca']:<20} {r['volumen_litros']:>10.1f} {r['largo_cm']:>10.1f} {r['alto_cm']:>10.1f} {r['fotos_validas']:>8} {r['fotos_descartadas']:>8}")

    # Guardar resumen global
    with open(output_dir / "resumen_global.json", 'w') as f:
        json.dump(resumen_global, f, indent=2, ensure_ascii=False)
    print(f"\n  Archivos en: {output_dir}/")
    print(f"  Resumen global: {output_dir}/resumen_global.json")


if __name__ == '__main__':
    main()
