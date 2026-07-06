"""
Modelo lateral triangulado con escala real calibrada por postes.
Escala: 50cm / 32px promedio
"""

import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from scipy.spatial import Delaunay, ConvexHull
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


# ═══════════════════════════════════════
# ESCALA CALIBRADA
# ═══════════════════════════════════════
POSTE_REAL_CM = 50.0
POSTE1_PX = 31
POSTE2_PX = 33
ESCALA_CM_PX = POSTE_REAL_CM / ((POSTE1_PX + POSTE2_PX) / 2.0)  # 1.5625 cm/px
PESO_KG = 262


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


def guardar_ply(path, pts_cm, tris, colores, simetrico=False):
    """Guarda PLY en cm. Si simétrico, espeja con profundidad elíptica."""
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

    nv, nf = len(all_pts), len(all_tris)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\n")
        f.write(f"comment Unidades: centimetros\n")
        f.write(f"comment Escala: {ESCALA_CM_PX:.4f} cm/px (postes {POSTE_REAL_CM}cm: {POSTE1_PX}px y {POSTE2_PX}px)\n")
        f.write(f"comment Peso: {PESO_KG} kg\n")
        f.write(f"element vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n")
        for i, pt in enumerate(all_pts):
            r,g,b = (int(all_colors[i][0]), int(all_colors[i][1]), int(all_colors[i][2])) if all_colors is not None and i<len(all_colors) else (139,90,43)
            f.write(f"{pt[0]:.2f} {pt[1]:.2f} {pt[2]:.2f} {r} {g} {b}\n")
        for t in all_tris:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    return all_pts, all_tris


def main():
    project = Path(__file__).parent
    fotos_dir = project / "checkpoints" / "dataset" / "modelo" / "vaca1"
    output_dir = project / "output_modelo_final"
    output_dir.mkdir(exist_ok=True)

    print(f"ESCALA: {ESCALA_CM_PX:.4f} cm/px")
    print(f"  (Postes: {POSTE1_PX}px y {POSTE2_PX}px = {POSTE_REAL_CM}cm cada uno)\n")

    cow_model = YOLO(str(project / "models_yolo" / "cow.pt"))
    coco_model = YOLO(str(project / "yolov8n.pt"))

    fotos = sorted([f for f in fotos_dir.iterdir() if f.suffix.lower() in ('.png','.jpg','.jpeg')])

    resultados = []
    mejor = None
    mejor_area = 0

    for i, foto in enumerate(fotos):
        img = cv2.imread(str(foto))
        if img is None:
            continue

        bbox = detectar_vaca(img, cow_model, coco_model)
        if bbox is None:
            continue

        mask, contorno = segmentar(img, bbox)
        if mask is None:
            continue

        area_px = cv2.contourArea(contorno)
        x1,y1,x2,y2 = bbox
        largo_cm = (x2-x1) * ESCALA_CM_PX
        alto_cm = (y2-y1) * ESCALA_CM_PX
        area_cm2 = area_px * ESCALA_CM_PX**2

        print(f"[{i+1:2d}] {foto.name}")
        print(f"     Largo: {largo_cm:.0f} cm | Alto: {alto_cm:.0f} cm | Área: {area_cm2:.0f} cm²")

        resultados.append({
            'foto': foto.name,
            'largo_cm': round(largo_cm, 1),
            'alto_cm': round(alto_cm, 1),
            'area_cm2': round(area_cm2, 1),
            'area_px': int(area_px),
        })

        if area_px > mejor_area:
            mejor_area = area_px
            mejor = dict(foto=foto.name, img=img, bbox=bbox, mask=mask,
                        contorno=contorno, area_px=area_px, idx=i)

    if not mejor:
        print("ERROR")
        return

    # ═══════════════════════════════════════
    # MODELO FINAL con la mejor foto
    # ═══════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  MODELO FINAL: {mejor['foto']}")
    print(f"{'='*60}")

    pts_b, pts_i = samplear(mejor['contorno'], mejor['mask'])
    puntos_px, tris = triangular(pts_b, pts_i, mejor['mask'])
    if puntos_px is None:
        return

    # Escalar a cm
    puntos_cm = puntos_px.astype(float) * ESCALA_CM_PX
    puntos_cm[:, 1] = puntos_cm[:, 1].max() - puntos_cm[:, 1]  # Y arriba

    # Colores
    colores = np.array([
        mejor['img'][min(pt[1], mejor['img'].shape[0]-1), min(pt[0], mejor['img'].shape[1]-1)][::-1]
        for pt in puntos_px
    ])

    # Medidas
    x_min, x_max = puntos_cm[:,0].min(), puntos_cm[:,0].max()
    y_min, y_max = puntos_cm[:,1].min(), puntos_cm[:,1].max()
    largo = x_max - x_min
    alto = y_max - y_min
    area_cm2 = mejor['area_px'] * ESCALA_CM_PX**2

    print(f"\n  MEDIDAS REALES (calibradas por postes):")
    print(f"    Largo:  {largo:.1f} cm")
    print(f"    Alto:   {alto:.1f} cm")
    print(f"    Área:   {area_cm2:.0f} cm²")

    # PLY lateral
    ply_lat = output_dir / "vaca1_lateral.ply"
    guardar_ply(str(ply_lat), puntos_cm, tris, colores, simetrico=False)

    # PLY simétrico 3D
    ply_3d = output_dir / "vaca1_3d.ply"
    pts_3d, tris_3d = guardar_ply(str(ply_3d), puntos_cm, tris, colores, simetrico=True)

    # Volumen
    try:
        hull = ConvexHull(pts_3d)
        vol_cm3 = hull.volume
        vol_litros = vol_cm3 / 1000.0
        sup_cm2 = hull.area
    except:
        vol_cm3 = vol_litros = sup_cm2 = 0

    print(f"\n  VOLUMEN (modelo simétrico):")
    print(f"    {vol_cm3:.0f} cm³ = {vol_litros:.1f} litros")
    print(f"    Superficie: {sup_cm2:.0f} cm²")

    print(f"\n  CONSTANTES (Vaca 1 = {PESO_KG} kg):")
    k_vol = PESO_KG / vol_cm3 if vol_cm3 > 0 else 0
    k_area = PESO_KG / area_cm2 if area_cm2 > 0 else 0
    k_litro = PESO_KG / vol_litros if vol_litros > 0 else 0
    print(f"    K peso/volumen:    {k_vol:.6f} kg/cm³")
    print(f"    K peso/litros:     {k_litro:.4f} kg/litro")
    print(f"    K peso/área:       {k_area:.4f} kg/cm²")

    # ═══════════════════════════════════════
    # VISUALIZACIÓN
    # ═══════════════════════════════════════
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'MODELO FINAL - Vaca 1 ({PESO_KG} kg) - Escala: {ESCALA_CM_PX:.4f} cm/px',
                 fontsize=14, fontweight='bold')

    # 1. Original
    ax1 = fig.add_subplot(2, 3, 1)
    img_rgb = cv2.cvtColor(mejor['img'], cv2.COLOR_BGR2RGB)
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
    ax2.set_title('Segmentación')
    ax2.axis('off')

    # 3. Malla en px
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(img_rgb, alpha=0.3)
    ax3.triplot(puntos_px[:,0], puntos_px[:,1], tris, 'b-', lw=0.4)
    ax3.plot(puntos_px[:,0], puntos_px[:,1], 'r.', ms=1.5)
    ax3.set_title(f'Malla ({len(tris)} triángulos)')
    ax3.axis('off')

    # 4. Modelo con textura + medidas cm
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('black')
    polys = [puntos_cm[t] for t in tris]
    fcolors = [(colores[t]/255.0).mean(axis=0) for t in tris]
    ax4.add_collection(PolyCollection(polys, facecolors=fcolors, edgecolors='none', alpha=0.9))
    ax4.set_xlim(x_min-3, x_max+3)
    ax4.set_ylim(y_min-3, y_max+3)
    # Flechas medida
    ax4.annotate('', xy=(x_max, y_min-4), xytext=(x_min, y_min-4),
                 arrowprops=dict(arrowstyle='<->', color='yellow', lw=2))
    ax4.text((x_min+x_max)/2, y_min-7, f'{largo:.0f} cm', color='yellow',
             ha='center', fontsize=11, fontweight='bold')
    ax4.annotate('', xy=(x_max+4, y_max), xytext=(x_max+4, y_min),
                 arrowprops=dict(arrowstyle='<->', color='cyan', lw=2))
    ax4.text(x_max+6, (y_min+y_max)/2, f'{alto:.0f} cm', color='cyan',
             ha='left', fontsize=11, fontweight='bold', rotation=90)
    ax4.set_title('Modelo Escalado (cm)')
    ax4.set_aspect('equal')
    ax4.axis('off')

    # 5. Wireframe limpio
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('white')
    ax5.triplot(puntos_cm[:,0], puntos_cm[:,1], tris, color='sienna', linewidth=0.3)
    ax5.set_xlim(x_min-3, x_max+3)
    ax5.set_ylim(y_min-3, y_max+3)
    ax5.set_title('Wireframe (cm)')
    ax5.set_aspect('equal')
    ax5.axis('off')

    # 6. Datos
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    info = f"""VACA 1 - MODELO CALIBRADO

Peso real:        {PESO_KG} kg
Escala:           {ESCALA_CM_PX:.4f} cm/px
  Poste 1:        {POSTE1_PX} px → {POSTE_REAL_CM} cm
  Poste 2:        {POSTE2_PX} px → {POSTE_REAL_CM} cm

MEDIDAS:
  Largo:          {largo:.1f} cm
  Alto:           {alto:.1f} cm
  Área lateral:   {area_cm2:.0f} cm²
  Volumen 3D:     {vol_cm3:.0f} cm³
                  {vol_litros:.1f} litros

CONSTANTES:
  K = peso/vol  = {k_litro:.4f} kg/litro
  K = peso/área = {k_area:.4f} kg/cm²

Triángulos:       {len(tris)}
Puntos:           {len(puntos_cm)}"""

    ax6.text(0.05, 0.95, info, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    vis_path = output_dir / "MODELO_FINAL_vaca1.png"
    plt.savefig(str(vis_path), dpi=150, bbox_inches='tight')
    plt.close()

    # JSON
    with open(output_dir / "modelo_final.json", 'w') as f:
        json.dump({
            'vaca': 'vaca1', 'peso_kg': PESO_KG,
            'escala_cm_px': round(ESCALA_CM_PX, 4),
            'largo_cm': round(largo, 1), 'alto_cm': round(alto, 1),
            'area_lateral_cm2': round(area_cm2, 1),
            'volumen_cm3': round(vol_cm3, 1), 'volumen_litros': round(vol_litros, 1),
            'k_peso_volumen_kg_litro': round(k_litro, 4),
            'k_peso_area_kg_cm2': round(k_area, 4),
            'resultados_por_foto': resultados,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Archivos generados:")
    print(f"    {ply_lat.name}       - perfil lateral (cm)")
    print(f"    {ply_3d.name}            - modelo 3D simétrico (cm)")
    print(f"    MODELO_FINAL_vaca1.png   - visualización completa")
    print(f"    modelo_final.json        - datos")

    # Abrir en MeshLab
    os.system(f'open "/Applications/MeshLab2025.07.app" --args "{ply_3d}"')
    print(f"\n  Abriendo modelo 3D en MeshLab...")


if __name__ == '__main__':
    main()
