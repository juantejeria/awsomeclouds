"""
Modelo triangulado de vaca.
Detecta silueta con GrabCut, samplea puntos, triangula con Delaunay.
"""

import cv2
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pathlib import Path
from scipy.spatial import Delaunay


def obtener_silueta(img_path):
    """Obtiene silueta de la vaca usando GrabCut."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None

    h, w = img.shape[:2]

    # Rectángulo inicial: centro de la imagen con margen
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Máscara: foreground probable + foreground seguro
    mask_fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Limpiar con morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    # Contorno más grande
    contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    contorno = max(contours, key=cv2.contourArea)

    # Verificar que sea suficientemente grande (>5% de la imagen)
    if cv2.contourArea(contorno) < h * w * 0.05:
        return None, None, None

    # Máscara limpia solo con el contorno principal
    mask_clean = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask_clean, [contorno], -1, 255, -1)

    return img, mask_clean, contorno


def samplear_puntos(contorno, mask, n_contorno=60, n_interior=30):
    """
    Samplea puntos sobre el contorno y en el interior.
    - n_contorno: puntos equiespaciados sobre el borde
    - n_interior: puntos aleatorios dentro de la silueta
    """
    # Puntos sobre el contorno (equiespaciados)
    contorno_flat = contorno.reshape(-1, 2)
    total_pts = len(contorno_flat)
    step = max(1, total_pts // n_contorno)
    pts_contorno = contorno_flat[::step]

    # Puntos interiores (grid regular dentro de la máscara)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return pts_contorno, np.array([])

    # Grid regular dentro del bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    grid_x = np.linspace(x_min, x_max, int(np.sqrt(n_interior) * 2) + 2)[1:-1]
    grid_y = np.linspace(y_min, y_max, int(np.sqrt(n_interior)) + 2)[1:-1]
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(int)

    # Filtrar: solo puntos dentro de la máscara
    pts_interior = []
    for pt in grid_pts:
        if 0 <= pt[1] < mask.shape[0] and 0 <= pt[0] < mask.shape[1]:
            if mask[pt[1], pt[0]] > 0:
                pts_interior.append(pt)

    pts_interior = np.array(pts_interior) if pts_interior else np.array([]).reshape(0, 2)

    return pts_contorno, pts_interior


def triangular(pts_contorno, pts_interior):
    """Crea triangulación de Delaunay con todos los puntos."""
    if len(pts_interior) > 0:
        todos_pts = np.vstack([pts_contorno, pts_interior])
    else:
        todos_pts = pts_contorno

    # Eliminar duplicados
    todos_pts = np.unique(todos_pts, axis=0)

    if len(todos_pts) < 3:
        return None, None

    delaunay = Delaunay(todos_pts)
    return todos_pts, delaunay


def calcular_metricas_triangulos(puntos, delaunay):
    """Calcula métricas desde los triángulos."""
    triangulos = delaunay.simplices
    areas = []

    for tri_idx in triangulos:
        p0, p1, p2 = puntos[tri_idx]
        # Área con fórmula del determinante
        area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        areas.append(area)

    areas = np.array(areas)

    return {
        'num_puntos': len(puntos),
        'num_triangulos': len(triangulos),
        'area_total_px': round(float(areas.sum()), 1),
        'area_promedio_tri': round(float(areas.mean()), 1),
        'area_mediana_tri': round(float(np.median(areas)), 1),
    }


def visualizar_modelo(img, puntos, delaunay, contorno, metricas, output_path):
    """Genera visualización del modelo triangulado."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Imagen original con contorno
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    contorno_flat = contorno.reshape(-1, 2)
    axes[0].plot(contorno_flat[:, 0], contorno_flat[:, 1], 'r-', linewidth=1.5)
    axes[0].set_title('Original + Contorno')
    axes[0].axis('off')

    # 2. Modelo triangulado sobre la imagen
    axes[1].imshow(img_rgb, alpha=0.4)
    axes[1].triplot(puntos[:, 0], puntos[:, 1], delaunay.simplices, 'b-', linewidth=0.5)
    axes[1].plot(puntos[:, 0], puntos[:, 1], 'r.', markersize=2)
    axes[1].set_title(f'Modelo Triangulado ({metricas["num_triangulos"]} triángulos)')
    axes[1].axis('off')

    # 3. Solo el modelo (wireframe)
    axes[2].set_facecolor('white')
    axes[2].triplot(puntos[:, 0], puntos[:, 1], delaunay.simplices, 'k-', linewidth=0.5)
    axes[2].plot(puntos[:, 0], puntos[:, 1], 'r.', markersize=2)
    axes[2].set_title(f'Wireframe (área: {metricas["area_total_px"]:.0f} px²)')
    axes[2].set_aspect('equal')
    axes[2].invert_yaxis()
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    base_dir = Path(__file__).parent / "checkpoints" / "dataset" / "modelo" / "vaca1"
    output_dir = Path(__file__).parent / "output_modelo_triangulo"
    output_dir.mkdir(exist_ok=True)

    fotos = sorted([f for f in base_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    print(f"Encontradas {len(fotos)} fotos\n")

    todos_resultados = []

    for i, foto in enumerate(fotos):
        print(f"[{i+1}/{len(fotos)}] {foto.name}")

        img, mask, contorno = obtener_silueta(str(foto))
        if img is None:
            print("  -> No se pudo segmentar, saltando\n")
            continue

        pts_contorno, pts_interior = samplear_puntos(contorno, mask)
        puntos, delaunay = triangular(pts_contorno, pts_interior)

        if puntos is None:
            print("  -> No hay suficientes puntos para triangular\n")
            continue

        metricas = calcular_metricas_triangulos(puntos, delaunay)
        metricas['foto'] = foto.name
        todos_resultados.append(metricas)

        print(f"  Puntos: {metricas['num_puntos']} | Triángulos: {metricas['num_triangulos']} | Área: {metricas['area_total_px']} px²")

        vis_path = output_dir / f"tri_{i+1:02d}_{foto.stem}.png"
        visualizar_modelo(img, puntos, delaunay, contorno, metricas, str(vis_path))
        print(f"  -> {vis_path.name}\n")

    if not todos_resultados:
        print("No se procesó ninguna foto.")
        return

    # Perfil promedio
    print("=" * 60)
    print(f"MODELO TRIANGULADO - VACA 1 ({len(todos_resultados)} fotos)")
    print("=" * 60)

    areas = [r['area_total_px'] for r in todos_resultados]
    print(f"  Área promedio:  {np.mean(areas):.0f} px²")
    print(f"  Área std:       {np.std(areas):.0f} px²")
    print(f"  Área min:       {min(areas):.0f} px²")
    print(f"  Área max:       {max(areas):.0f} px²")
    print(f"  Triángulos avg: {np.mean([r['num_triangulos'] for r in todos_resultados]):.0f}")

    # Guardar JSON
    json_path = output_dir / "modelo_vaca1.json"
    with open(json_path, 'w') as f:
        json.dump({
            'vaca': 'vaca1',
            'fotos_procesadas': len(todos_resultados),
            'area_promedio_px': round(float(np.mean(areas)), 1),
            'area_std_px': round(float(np.std(areas)), 1),
            'resultados': todos_resultados,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResultados: {json_path}")


if __name__ == '__main__':
    main()
