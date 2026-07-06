"""
Modelo lateral triangulado con ESCALA REAL (cm).
YOLO detecta vaca + postes → calcula cm/px → modelo en centímetros → PLY para MeshLab.
"""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from scipy.spatial import Delaunay, ConvexHull
from depth_estimation import DepthEstimator
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def detectar_vaca_yolo(img, cow_model, coco_model):
    """Detecta la vaca y retorna bounding box."""
    results = cow_model(img, conf=0.15, verbose=False)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        areas = [(box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]) for box in boxes]
        best = int(np.argmax(areas))
        return boxes[best].xyxy[0].cpu().numpy().astype(int)

    results = coco_model(img, conf=0.2, classes=[19], verbose=False)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        areas = [(box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]) for box in boxes]
        best = int(np.argmax(areas))
        return boxes[best].xyxy[0].cpu().numpy().astype(int)
    return None


def segmentar_con_grabcut(img, bbox):
    """GrabCut refinado usando el bbox de YOLO."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = 10
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, 10, cv2.GC_INIT_WITH_RECT)
    mask_fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contorno = max(contours, key=cv2.contourArea)
    mask_clean = np.zeros_like(mask_fg)
    cv2.drawContours(mask_clean, [contorno], -1, 255, -1)
    return mask_clean, contorno


def samplear_puntos(contorno, mask, n_borde=80, n_interior=40):
    """Puntos equiespaciados en borde + grid interior."""
    c = contorno.reshape(-1, 2)
    step = max(1, len(c) // n_borde)
    pts_borde = c[::step]

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return pts_borde, np.array([]).reshape(0, 2)

    cols = int(np.sqrt(n_interior) * 1.5) + 2
    rows = int(np.sqrt(n_interior)) + 2
    gx = np.linspace(xs.min(), xs.max(), cols + 2)[1:-1]
    gy = np.linspace(ys.min(), ys.max(), rows + 2)[1:-1]
    mx, my = np.meshgrid(gx, gy)
    grid = np.column_stack([mx.ravel(), my.ravel()]).astype(int)

    interior = [pt for pt in grid
                if 0 <= pt[1] < mask.shape[0] and 0 <= pt[0] < mask.shape[1]
                and mask[pt[1], pt[0]] > 0]
    interior = np.array(interior) if interior else np.array([]).reshape(0, 2)
    return pts_borde, interior


def triangular(pts_borde, pts_interior, mask):
    """Delaunay, filtrando triángulos fuera de la máscara."""
    todos = np.vstack([pts_borde, pts_interior]) if len(pts_interior) > 0 else pts_borde
    todos = np.unique(todos, axis=0)
    if len(todos) < 3:
        return None, None

    tri = Delaunay(todos)
    validos = []
    for simplex in tri.simplices:
        cx, cy = todos[simplex].mean(axis=0).astype(int)
        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx] > 0:
            validos.append(simplex)
    return todos, np.array(validos) if validos else np.array([]).reshape(0, 3)


def guardar_ply_escalado(path, puntos_cm, triangulos, colores, simetrico=False):
    """
    Guarda PLY en centímetros reales.
    Si simetrico=True, espeja para generar volumen.
    """
    if simetrico:
        n_orig = len(puntos_cm)
        # Estimar profundidad (ancho del animal) con perfil elíptico
        ys = puntos_cm[:, 1]
        y_min, y_max = ys.min(), ys.max()
        y_range = y_max - y_min if y_max > y_min else 1
        y_center = y_min + y_range * 0.4

        profundidades = []
        for pt in puntos_cm:
            dist_y = min(abs(pt[1] - y_center) / (y_range * 0.5), 1.0)
            depth = y_range * 0.25 * np.sqrt(max(0, 1 - dist_y ** 2))
            profundidades.append(depth)
        profundidades = np.array(profundidades)

        pts_der = np.column_stack([puntos_cm[:, 0], puntos_cm[:, 1], profundidades])
        pts_izq = np.column_stack([puntos_cm[:, 0], puntos_cm[:, 1], -profundidades])
        all_pts = np.vstack([pts_der, pts_izq])

        all_colors = np.vstack([colores, colores]) if colores is not None else None

        tris_der = triangulos.copy()
        tris_izq = triangulos.copy() + n_orig
        tris_izq = tris_izq[:, [0, 2, 1]]  # invertir winding
        all_tris = np.vstack([tris_der, tris_izq])
    else:
        all_pts = np.column_stack([puntos_cm[:, 0], puntos_cm[:, 1], np.zeros(len(puntos_cm))])
        all_colors = colores
        all_tris = triangulos

    nv, nf = len(all_pts), len(all_tris)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\n")
        f.write(f"comment Unidades: centimetros (cm)\n")
        f.write(f"comment Modelo de vaca - escala real\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n")

        for i, pt in enumerate(all_pts):
            r, g, b = (int(all_colors[i][0]), int(all_colors[i][1]), int(all_colors[i][2])) if all_colors is not None and i < len(all_colors) else (139, 90, 43)
            f.write(f"{pt[0]:.2f} {pt[1]:.2f} {pt[2]:.2f} {r} {g} {b}\n")
        for tri in all_tris:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

    return all_pts, all_tris


def main():
    project = Path(__file__).parent
    fotos_dir = project / "checkpoints" / "dataset" / "modelo" / "vaca1"
    output_dir = project / "output_modelo_escalado"
    output_dir.mkdir(exist_ok=True)

    print("Cargando modelos...")
    cow_model = YOLO(str(project / "models_yolo" / "cow.pt"))
    coco_model = YOLO(str(project / "yolov8n.pt"))
    depth_est = DepthEstimator(
        sticker_model_path=str(project / "models_yolo" / "sticker.pt"),
        poste1_height_cm=50,
        poste2_height_cm=50,
        conf_threshold=0.05
    )

    fotos = sorted([f for f in fotos_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    print(f"Fotos: {len(fotos)}\n")

    mejor = None
    mejor_score = 0

    for i, foto in enumerate(fotos):
        print(f"\n[{i+1}/{len(fotos)}] {foto.name}")
        img = cv2.imread(str(foto))
        if img is None:
            continue

        # Detectar vaca
        bbox = detectar_vaca_yolo(img, cow_model, coco_model)
        if bbox is None:
            print("  No se detectó vaca")
            continue

        # Calcular escala con postes
        escala_result = depth_est.estimar_escala_con_postes(
            img, animal_bbox=bbox.tolist(), debug=True
        )
        escala = escala_result.get('escala')

        if escala and escala > 0:
            print(f"  ESCALA: {escala:.4f} cm/px")
        else:
            print(f"  Sin escala (postes no detectados)")

        # Segmentar
        mask, contorno = segmentar_con_grabcut(img, bbox)
        if mask is None:
            continue

        cow_area_px = cv2.contourArea(contorno)
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)

        # Score: preferir fotos con escala Y buena área
        score = cow_area_px * (10 if escala else 1)

        print(f"  Área vaca: {cow_area_px} px²", end="")
        if escala:
            cow_area_cm2 = cow_area_px * (escala ** 2)
            largo_cm = (x2 - x1) * escala
            alto_cm = (y2 - y1) * escala
            print(f" → {cow_area_cm2:.0f} cm² | Largo: {largo_cm:.0f} cm | Alto: {alto_cm:.0f} cm")
        else:
            print()

        if score > mejor_score:
            mejor_score = score
            mejor = {
                'foto': foto.name,
                'img': img,
                'bbox': bbox,
                'mask': mask,
                'contorno': contorno,
                'escala': escala,
                'area_px': cow_area_px,
                'escala_result': escala_result,
            }

    if not mejor:
        print("\nERROR: No se pudo procesar ninguna foto.")
        return

    print(f"\n{'='*60}")
    print(f"  MEJOR FOTO: {mejor['foto']}")
    escala = mejor['escala']
    if not escala:
        print("  WARN: No se detectaron postes. Usando escala estimada.")
        # Fallback: estimar escala asumiendo que una vaca adulta mide ~130cm de alto
        _, y1, _, y2 = mejor['bbox']
        bbox_h_px = y2 - y1
        escala = 130.0 / bbox_h_px
        print(f"  Escala estimada (130cm alto): {escala:.4f} cm/px")

    print(f"  ESCALA FINAL: {escala:.4f} cm/px")
    print(f"{'='*60}")

    # Triangular la mejor foto
    pts_borde, pts_interior = samplear_puntos(mejor['contorno'], mejor['mask'])
    puntos_px, tris = triangular(pts_borde, pts_interior, mejor['mask'])

    if puntos_px is None or len(tris) == 0:
        print("ERROR: Triangulación falló")
        return

    # Convertir a centímetros
    puntos_cm = puntos_px.astype(float) * escala
    # Invertir Y para que el modelo quede "parado" (Y positivo = arriba)
    puntos_cm[:, 1] = puntos_cm[:, 1].max() - puntos_cm[:, 1]

    # Colores
    colores = np.array([
        mejor['img'][min(pt[1], mejor['img'].shape[0]-1), min(pt[0], mejor['img'].shape[1]-1)][::-1]
        for pt in puntos_px
    ])

    # Métricas en cm
    x_min, x_max = puntos_cm[:, 0].min(), puntos_cm[:, 0].max()
    y_min, y_max = puntos_cm[:, 1].min(), puntos_cm[:, 1].max()
    largo_cm = x_max - x_min
    alto_cm = y_max - y_min
    area_cm2 = mejor['area_px'] * (escala ** 2)

    print(f"\n  MEDIDAS REALES:")
    print(f"    Largo (cabeza-cola): {largo_cm:.1f} cm")
    print(f"    Alto (lomo-pata):    {alto_cm:.1f} cm")
    print(f"    Área lateral:        {area_cm2:.0f} cm²")

    # ── PLY lateral (plano, escala real) ──
    ply_lateral = output_dir / "MODELO_vaca1_lateral_cm.ply"
    guardar_ply_escalado(str(ply_lateral), puntos_cm, tris, colores, simetrico=False)
    print(f"\n  PLY lateral:   {ply_lateral}")

    # ── PLY simétrico (3D, escala real) ──
    ply_sim = output_dir / "MODELO_vaca1_3d_cm.ply"
    pts_3d, tris_3d = guardar_ply_escalado(str(ply_sim), puntos_cm, tris, colores, simetrico=True)
    print(f"  PLY simétrico: {ply_sim}")

    # Volumen
    try:
        hull = ConvexHull(pts_3d)
        volumen_cm3 = hull.volume
        volumen_litros = volumen_cm3 / 1000.0
        superficie_cm2 = hull.area
        print(f"\n  VOLUMEN:")
        print(f"    Convex hull: {volumen_cm3:.0f} cm³ = {volumen_litros:.1f} litros")
        print(f"    Superficie:  {superficie_cm2:.0f} cm²")
        print(f"\n  RELACIÓN PESO/VOLUMEN:")
        print(f"    Peso:     262 kg")
        print(f"    Volumen:  {volumen_litros:.1f} litros")
        print(f"    Densidad: {262 / volumen_litros:.3f} kg/litro")
        print(f"    K (peso/volumen): {262 / volumen_cm3:.6f} kg/cm³")
    except Exception as e:
        volumen_cm3 = 0
        volumen_litros = 0
        superficie_cm2 = 0
        print(f"  No se pudo calcular volumen: {e}")

    # ── Visualización ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Modelo Lateral Escalado - Vaca 1 (262 kg)', fontsize=14, fontweight='bold')

    img_rgb = cv2.cvtColor(mejor['img'], cv2.COLOR_BGR2RGB)

    # 1. Original + bbox + postes
    axes[0, 0].imshow(img_rgb)
    x1, y1, x2, y2 = mejor['bbox']
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
    axes[0, 0].add_patch(rect)
    er = mejor['escala_result']
    if er.get('poste1_bbox'):
        p = er['poste1_bbox']
        r1 = plt.Rectangle((p[0], p[1]), p[2]-p[0], p[3]-p[1], fill=False, edgecolor='red', linewidth=2)
        axes[0, 0].add_patch(r1)
        axes[0, 0].text(p[0], p[1]-5, 'Poste 50cm', color='red', fontsize=8)
    if er.get('poste2_bbox'):
        p = er['poste2_bbox']
        r2 = plt.Rectangle((p[0], p[1]), p[2]-p[0], p[3]-p[1], fill=False, edgecolor='red', linewidth=2)
        axes[0, 0].add_patch(r2)
    axes[0, 0].set_title(f'Detección (escala: {escala:.4f} cm/px)')
    axes[0, 0].axis('off')

    # 2. Malla sobre imagen
    axes[0, 1].imshow(img_rgb, alpha=0.3)
    if len(tris) > 0:
        axes[0, 1].triplot(puntos_px[:, 0], puntos_px[:, 1], tris, 'b-', linewidth=0.4)
    axes[0, 1].plot(puntos_px[:, 0], puntos_px[:, 1], 'r.', markersize=1.5)
    axes[0, 1].set_title(f'Malla ({len(tris)} triángulos)')
    axes[0, 1].axis('off')

    # 3. Modelo con textura + medidas en cm
    axes[1, 0].set_facecolor('black')
    if len(tris) > 0:
        from matplotlib.collections import PolyCollection
        polygons = [puntos_cm[t] for t in tris]
        face_colors = [(colores[t] / 255.0).mean(axis=0) for t in tris]
        pc = PolyCollection(polygons, facecolors=face_colors, edgecolors='none', alpha=0.9)
        axes[1, 0].add_collection(pc)
        axes[1, 0].set_xlim(puntos_cm[:, 0].min() - 2, puntos_cm[:, 0].max() + 2)
        axes[1, 0].set_ylim(puntos_cm[:, 1].min() - 2, puntos_cm[:, 1].max() + 2)
    # Flechas de medida
    axes[1, 0].annotate('', xy=(x_max, y_min - 3), xytext=(x_min, y_min - 3),
                        arrowprops=dict(arrowstyle='<->', color='yellow', lw=1.5))
    axes[1, 0].text((x_min + x_max) / 2, y_min - 5, f'{largo_cm:.0f} cm',
                    color='yellow', ha='center', fontsize=10, fontweight='bold')
    axes[1, 0].annotate('', xy=(x_max + 3, y_max), xytext=(x_max + 3, y_min),
                        arrowprops=dict(arrowstyle='<->', color='cyan', lw=1.5))
    axes[1, 0].text(x_max + 5, (y_min + y_max) / 2, f'{alto_cm:.0f} cm',
                    color='cyan', ha='left', fontsize=10, fontweight='bold', rotation=90)
    axes[1, 0].set_title('Modelo con Textura (cm)')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].axis('off')

    # 4. Info
    axes[1, 1].axis('off')
    info_text = f"""
    VACA 1 - MODELO ESCALADO

    Peso real:         262 kg
    Escala:            {escala:.4f} cm/px

    MEDIDAS:
    Largo:             {largo_cm:.1f} cm
    Alto:              {alto_cm:.1f} cm
    Área lateral:      {area_cm2:.0f} cm²
    Volumen (3D sim.): {volumen_cm3:.0f} cm³
                       {volumen_litros:.1f} litros

    CONSTANTES:
    K = peso/volumen = {262/volumen_cm3:.6f} kg/cm³
    K = peso/área    = {262/area_cm2:.4f} kg/cm²

    Triángulos:        {len(tris)}
    Puntos:            {len(puntos_cm)}
    """
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    vis_path = output_dir / "modelo_escalado_vaca1.png"
    plt.savefig(str(vis_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Visualización: {vis_path}")

    # JSON
    resumen = {
        'vaca': 'vaca1',
        'peso_kg': 262,
        'escala_cm_por_px': round(escala, 6),
        'largo_cm': round(largo_cm, 1),
        'alto_cm': round(alto_cm, 1),
        'area_lateral_cm2': round(area_cm2, 1),
        'volumen_cm3': round(volumen_cm3, 1),
        'volumen_litros': round(volumen_litros, 1),
        'superficie_cm2': round(superficie_cm2, 1),
        'k_peso_volumen': round(262 / volumen_cm3, 6) if volumen_cm3 > 0 else None,
        'k_peso_area': round(262 / area_cm2, 6) if area_cm2 > 0 else None,
        'num_triangulos': len(tris),
        'foto_usada': mejor['foto'],
    }
    with open(output_dir / "resumen_escalado.json", 'w') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\n  Para abrir en MeshLab:")
    print(f"    open \"{ply_sim}\"")


if __name__ == '__main__':
    main()
