"""
Modelo lateral triangulado de vaca.
YOLO detecta la vaca → silueta limpia → triangulación → PLY para MeshLab.
Como la vaca es simétrica, el perfil lateral es suficiente.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import Delaunay, ConvexHull
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Segmentación con YOLO ──

def detectar_vaca_yolo(img, cow_model, coco_model):
    """Detecta la vaca y retorna bounding box."""
    # Intentar con cow.pt primero
    results = cow_model(img, conf=0.15, verbose=False)
    if results and len(results[0].boxes) > 0:
        # Tomar la detección con mayor área
        boxes = results[0].boxes
        areas = [(box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]) for box in boxes]
        best = int(np.argmax(areas))
        box = boxes[best].xyxy[0].cpu().numpy().astype(int)
        return box  # [x1, y1, x2, y2]

    # Fallback: yolov8n con clase "cow" (class 19 en COCO)
    results = coco_model(img, conf=0.2, classes=[19], verbose=False)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        areas = [(box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]) for box in boxes]
        best = int(np.argmax(areas))
        box = boxes[best].xyxy[0].cpu().numpy().astype(int)
        return box

    return None


def segmentar_con_grabcut(img, bbox, iteraciones=10):
    """GrabCut refinado usando el bbox de YOLO como seed."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # Expandir bbox ligeramente
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    rect = (x1, y1, x2 - x1, y2 - y1)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd, fgd, iteraciones, cv2.GC_INIT_WITH_RECT)
    mask_fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    # Quedarse con el contorno más grande dentro del bbox
    contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contorno = max(contours, key=cv2.contourArea)
    mask_clean = np.zeros_like(mask_fg)
    cv2.drawContours(mask_clean, [contorno], -1, 255, -1)

    return mask_clean, contorno


# ── Triangulación ──

def samplear_puntos(contorno, mask, n_borde=80, n_interior=40):
    """Puntos equiespaciados en borde + grid interior."""
    # Borde
    c = contorno.reshape(-1, 2)
    step = max(1, len(c) // n_borde)
    pts_borde = c[::step]

    # Interior: grid regular
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return pts_borde, np.array([]).reshape(0, 2)

    cols = int(np.sqrt(n_interior) * 1.5) + 2
    rows = int(np.sqrt(n_interior)) + 2
    gx = np.linspace(xs.min(), xs.max(), cols + 2)[1:-1]
    gy = np.linspace(ys.min(), ys.max(), rows + 2)[1:-1]
    mx, my = np.meshgrid(gx, gy)
    grid = np.column_stack([mx.ravel(), my.ravel()]).astype(int)

    interior = []
    for pt in grid:
        if 0 <= pt[1] < mask.shape[0] and 0 <= pt[0] < mask.shape[1]:
            if mask[pt[1], pt[0]] > 0:
                interior.append(pt)

    interior = np.array(interior) if interior else np.array([]).reshape(0, 2)
    return pts_borde, interior


def triangular(pts_borde, pts_interior, mask):
    """Delaunay triangulación, filtrando triángulos fuera de la máscara."""
    if len(pts_interior) > 0:
        todos = np.vstack([pts_borde, pts_interior])
    else:
        todos = pts_borde

    todos = np.unique(todos, axis=0)
    if len(todos) < 3:
        return None, None, None

    tri = Delaunay(todos)

    # Filtrar triángulos cuyo centroide está fuera de la máscara
    validos = []
    for simplex in tri.simplices:
        centroid = todos[simplex].mean(axis=0).astype(int)
        cx, cy = centroid[0], centroid[1]
        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
            if mask[cy, cx] > 0:
                validos.append(simplex)

    validos = np.array(validos) if validos else np.array([]).reshape(0, 3)
    return todos, tri, validos


# ── PLY export ──

def guardar_ply_malla(path, puntos, triangulos, colores=None):
    """Guarda malla triangulada como PLY (MeshLab compatible)."""
    nv = len(puntos)
    nf = len(triangulos)

    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i, pt in enumerate(puntos):
            x, y = float(pt[0]), float(pt[1])
            z = 0.0  # modelo lateral plano (z=0)
            if colores is not None and i < len(colores):
                r, g, b = colores[i]
            else:
                r, g, b = 139, 90, 43
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")

        for tri in triangulos:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def guardar_ply_simetrico(path, puntos, triangulos, colores=None):
    """
    Genera modelo simétrico: perfil lateral + espejo.
    Crea un modelo con volumen simulando la simetría del animal.
    El eje Z representa la profundidad (ancho del animal).
    """
    nv_orig = len(puntos)

    # Estimar profundidad basada en la altura local
    # La vaca es más ancha en el centro del cuerpo
    ys = puntos[:, 1]
    y_min, y_max = ys.min(), ys.max()
    y_range = y_max - y_min if y_max > y_min else 1

    # Centro vertical del cuerpo (la parte más ancha)
    y_center = y_min + y_range * 0.4  # un poco arriba del centro

    profundidades = []
    for pt in puntos:
        # Distancia normalizada al centro vertical
        dist_y = abs(pt[1] - y_center) / (y_range * 0.5)
        dist_y = min(dist_y, 1.0)
        # Perfil elíptico: más profundidad en el centro
        depth = y_range * 0.25 * np.sqrt(max(0, 1 - dist_y ** 2))
        profundidades.append(depth)

    profundidades = np.array(profundidades)

    # Lado derecho (z > 0) y lado izquierdo (z < 0)
    puntos_der = np.column_stack([puntos[:, 0], puntos[:, 1], profundidades])
    puntos_izq = np.column_stack([puntos[:, 0], puntos[:, 1], -profundidades])

    all_pts = np.vstack([puntos_der, puntos_izq])

    # Colores duplicados
    if colores is not None:
        all_colors = np.vstack([colores, colores])
    else:
        all_colors = None

    # Triángulos: originales + espejo (con offset de índices)
    tris_der = triangulos.copy()
    tris_izq = triangulos.copy() + nv_orig
    # Invertir winding de los triángulos del espejo para normales correctas
    tris_izq = tris_izq[:, [0, 2, 1]]

    all_tris = np.vstack([tris_der, tris_izq])

    nv = len(all_pts)
    nf = len(all_tris)

    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i, pt in enumerate(all_pts):
            if all_colors is not None and i < len(all_colors):
                r, g, b = all_colors[i]
            else:
                r, g, b = 139, 90, 43
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {int(r)} {int(g)} {int(b)}\n")

        for tri in all_tris:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

    return all_pts, all_tris


# ── Visualización ──

def visualizar(img, mask, contorno, bbox, puntos, tris_validos, colores, output_path, nombre):
    """Genera imagen con 4 paneles."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Modelo Lateral - {nombre}', fontsize=13, fontweight='bold')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Original + bbox YOLO
    axes[0, 0].imshow(img_rgb)
    x1, y1, x2, y2 = bbox
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='lime', linewidth=2)
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('YOLO Detection')
    axes[0, 0].axis('off')

    # 2. Segmentación
    overlay = img_rgb.copy()
    overlay[mask > 0] = [0, 200, 0]
    blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    axes[0, 1].imshow(blended)
    axes[0, 1].set_title('Segmentación (GrabCut + YOLO)')
    axes[0, 1].axis('off')

    # 3. Malla triangulada sobre imagen
    axes[1, 0].imshow(img_rgb, alpha=0.3)
    if len(tris_validos) > 0:
        axes[1, 0].triplot(puntos[:, 0], puntos[:, 1], tris_validos, 'b-', linewidth=0.4, alpha=0.8)
    axes[1, 0].plot(puntos[:, 0], puntos[:, 1], 'r.', markersize=1.5)
    axes[1, 0].set_title(f'Malla ({len(tris_validos)} triángulos)')
    axes[1, 0].axis('off')

    # 4. Modelo wireframe limpio con color
    axes[1, 1].set_facecolor('black')
    if len(tris_validos) > 0 and colores is not None:
        # Colorear triángulos con color promedio
        from matplotlib.collections import PolyCollection
        polygons = []
        face_colors = []
        for tri_idx in tris_validos:
            tri_pts = puntos[tri_idx]
            polygons.append(tri_pts)
            tri_colors = colores[tri_idx] / 255.0
            face_colors.append(tri_colors.mean(axis=0))

        pc = PolyCollection(polygons, facecolors=face_colors, edgecolors='none', alpha=0.9)
        axes[1, 1].add_collection(pc)
        axes[1, 1].set_xlim(puntos[:, 0].min() - 5, puntos[:, 0].max() + 5)
        axes[1, 1].set_ylim(puntos[:, 1].max() + 5, puntos[:, 1].min() - 5)
    axes[1, 1].set_title('Modelo con Textura')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ──

def main():
    project = Path(__file__).parent
    fotos_dir = project / "checkpoints" / "dataset" / "modelo" / "vaca1"
    output_dir = project / "output_modelo_lateral"
    output_dir.mkdir(exist_ok=True)

    print("Cargando modelos YOLO...")
    cow_model = YOLO(str(project / "models_yolo" / "cow.pt"))
    coco_model = YOLO(str(project / "yolov8n.pt"))

    fotos = sorted([f for f in fotos_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    print(f"Fotos: {len(fotos)}\n")

    resultados = []
    mejor_foto = None
    mejor_area = 0

    for i, foto in enumerate(fotos):
        print(f"[{i+1}/{len(fotos)}] {foto.name}")
        img = cv2.imread(str(foto))
        if img is None:
            continue

        # 1. Detectar vaca con YOLO
        bbox = detectar_vaca_yolo(img, cow_model, coco_model)
        if bbox is None:
            print("  No se detectó vaca")
            continue

        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        print(f"  YOLO bbox: [{x1},{y1}]-[{x2},{y2}] ({bbox_area} px²)")

        # 2. Segmentar con GrabCut guiado por YOLO
        mask, contorno = segmentar_con_grabcut(img, bbox)
        if mask is None:
            print("  Segmentación falló")
            continue

        cow_area = cv2.contourArea(contorno)
        print(f"  Área vaca: {cow_area} px²")

        # 3. Triangular
        pts_borde, pts_interior = samplear_puntos(contorno, mask)
        puntos, tri, tris_validos = triangular(pts_borde, pts_interior, mask)

        if puntos is None or len(tris_validos) == 0:
            print("  Triangulación falló")
            continue

        # Colores de la imagen
        colores = np.array([
            img[min(pt[1], img.shape[0]-1), min(pt[0], img.shape[1]-1)][::-1]
            for pt in puntos
        ])

        print(f"  Triángulos: {len(tris_validos)} | Puntos: {len(puntos)}")

        # Guardar visualización por foto
        vis_path = output_dir / f"modelo_{i+1:02d}_{foto.stem}.png"
        visualizar(img, mask, contorno, bbox, puntos, tris_validos, colores, str(vis_path), foto.name)

        # Guardar PLY lateral (plano)
        ply_path = output_dir / f"lateral_{i+1:02d}.ply"
        guardar_ply_malla(str(ply_path), puntos, tris_validos, colores)

        resultado = {
            'foto': foto.name,
            'bbox': bbox.tolist(),
            'area_vaca_px': int(cow_area),
            'num_triangulos': len(tris_validos),
            'num_puntos': len(puntos),
        }
        resultados.append(resultado)

        # Trackear la mejor foto (mayor área de vaca)
        if cow_area > mejor_area:
            mejor_area = cow_area
            mejor_foto = {
                'index': i,
                'foto': foto.name,
                'img': img,
                'mask': mask,
                'contorno': contorno,
                'bbox': bbox,
                'puntos': puntos,
                'tris': tris_validos,
                'colores': colores,
                'area': cow_area,
            }

    # ════════════════════════════════════════
    # MODELO PRINCIPAL: la mejor foto
    # ════════════════════════════════════════
    if mejor_foto:
        print(f"\n{'='*60}")
        print(f"  MEJOR MODELO: {mejor_foto['foto']}")
        print(f"  Área: {mejor_foto['area']} px² | Triángulos: {len(mejor_foto['tris'])}")
        print(f"{'='*60}")

        # PLY lateral (plano, para MeshLab)
        ply_lateral = output_dir / "MODELO_vaca1_lateral.ply"
        guardar_ply_malla(
            str(ply_lateral),
            mejor_foto['puntos'],
            mejor_foto['tris'],
            mejor_foto['colores']
        )
        print(f"\n  PLY lateral: {ply_lateral}")

        # PLY simétrico (con volumen, espejado)
        ply_simetrico = output_dir / "MODELO_vaca1_simetrico.ply"
        pts_3d, tris_3d = guardar_ply_simetrico(
            str(ply_simetrico),
            mejor_foto['puntos'],
            mejor_foto['tris'],
            mejor_foto['colores']
        )
        print(f"  PLY simétrico (3D): {ply_simetrico}")

        # Calcular volumen del modelo simétrico
        try:
            hull = ConvexHull(pts_3d)
            volumen = hull.volume
            superficie = hull.area
            print(f"\n  Volumen estimado: {volumen:.1f} unidades³")
            print(f"  Superficie: {superficie:.1f} unidades²")
        except Exception as e:
            volumen = 0
            superficie = 0
            print(f"  No se pudo calcular volumen: {e}")

    # Resumen
    print(f"\n{'='*60}")
    print(f"  RESUMEN - Vaca 1 (262 kg)")
    print(f"{'='*60}")
    print(f"  Fotos procesadas: {len(resultados)}/{len(fotos)}")
    print(f"  Archivos en: {output_dir}/")
    print(f"    - MODELO_vaca1_lateral.ply     (perfil lateral para MeshLab)")
    print(f"    - MODELO_vaca1_simetrico.ply   (modelo simétrico 3D para MeshLab)")
    print(f"    - modelo_XX_*.png              (visualización por foto)")
    print(f"\n  Abrir con MeshLab: open {output_dir}/MODELO_vaca1_simetrico.ply")

    # JSON
    resumen = {
        'vaca': 'vaca1',
        'peso_kg': 262,
        'fotos_procesadas': len(resultados),
        'mejor_foto': mejor_foto['foto'] if mejor_foto else None,
        'volumen_estimado': round(volumen, 2) if mejor_foto else 0,
        'resultados': resultados,
    }
    with open(output_dir / "resumen.json", 'w') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
