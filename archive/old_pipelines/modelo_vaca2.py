"""
Modelo lateral triangulado - Vaca 2.
Escala calibrada por altura conocida del animal: 112 cm.
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
# DATOS DEL INDIVIDUO
# ═══════════════════════════════════════
VACA_ID = 'vaca2'
ALTO_REAL_CM = 112.0  # Altura conocida del animal
PESO_KG = 462


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

    # Recortar al bbox original (sin padding) para respetar la altura real
    bbox_mask = np.zeros_like(mask_fg)
    bx1, by1, bx2, by2 = bbox
    bbox_mask[by1:by2, bx1:bx2] = 255
    mask_fg = cv2.bitwise_and(mask_fg, bbox_mask)

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


def guardar_ply(path, pts_cm, tris, colores, simetrico=False, escala=1.0):
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
        f.write(f"comment Animal: {VACA_ID} | Alto real: {ALTO_REAL_CM} cm | Escala: {escala:.4f} cm/px\n")
        f.write(f"element vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n")
        for i, pt in enumerate(all_pts):
            r,g,b = (int(all_colors[i][0]), int(all_colors[i][1]), int(all_colors[i][2])) if all_colors is not None and i<len(all_colors) else (100,50,30)
            f.write(f"{pt[0]:.2f} {pt[1]:.2f} {pt[2]:.2f} {r} {g} {b}\n")
        for t in all_tris:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    return all_pts, all_tris


def main():
    project = Path(__file__).parent
    fotos_dir = project / "checkpoints" / "dataset" / "modelo" / VACA_ID
    output_dir = project / f"output_modelo_{VACA_ID}"
    output_dir.mkdir(exist_ok=True)

    print(f"ANIMAL: {VACA_ID}")
    print(f"ALTO REAL: {ALTO_REAL_CM} cm\n")

    cow_model = YOLO(str(project / "models_yolo" / "cow.pt"))
    coco_model = YOLO(str(project / "yolov8n.pt"))

    fotos = sorted([f for f in fotos_dir.iterdir() if f.suffix.lower() in ('.png','.jpg','.jpeg')])
    print(f"Fotos: {len(fotos)}\n")

    resultados = []
    mejor = None
    mejor_area = 0

    for i, foto in enumerate(fotos):
        img = cv2.imread(str(foto))
        if img is None:
            continue

        bbox = detectar_vaca(img, cow_model, coco_model)
        if bbox is None:
            print(f"[{i+1:2d}] {foto.name} - No detectada")
            continue

        mask, contorno = segmentar(img, bbox)
        if mask is None:
            print(f"[{i+1:2d}] {foto.name} - Segmentación falló")
            continue

        area_px = cv2.contourArea(contorno)
        x1,y1,x2,y2 = bbox
        alto_px = y2 - y1
        escala = ALTO_REAL_CM / alto_px  # cm/px calibrado por altura del animal
        largo_cm = (x2 - x1) * escala
        alto_cm = alto_px * escala  # debería ser ~112
        area_cm2 = area_px * escala**2

        print(f"[{i+1:2d}] {foto.name}")
        print(f"     bbox: {alto_px}px alto → escala {escala:.4f} cm/px")
        print(f"     Largo: {largo_cm:.0f} cm | Alto: {alto_cm:.0f} cm | Área: {area_cm2:.0f} cm²")

        resultados.append({
            'foto': foto.name,
            'escala_cm_px': round(escala, 4),
            'largo_cm': round(largo_cm, 1),
            'alto_cm': round(alto_cm, 1),
            'area_cm2': round(area_cm2, 1),
            'area_px': int(area_px),
            'alto_bbox_px': int(alto_px),
        })

        if area_px > mejor_area:
            mejor_area = area_px
            mejor = dict(foto=foto.name, img=img, bbox=bbox, mask=mask,
                        contorno=contorno, area_px=area_px, escala=escala, idx=i)

    if not resultados:
        print("ERROR: no se procesó ninguna foto")
        return

    # ═══════════════════════════════════════
    # MODELO POR CADA FOTO
    # ═══════════════════════════════════════
    modelos = []

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

        x1,y1,x2,y2 = bbox
        alto_px = y2 - y1
        escala = ALTO_REAL_CM / alto_px
        area_px = cv2.contourArea(contorno)

        pts_b, pts_i = samplear(contorno, mask)
        puntos_px, tris = triangular(pts_b, pts_i, mask)
        if puntos_px is None or len(tris) == 0:
            continue

        puntos_cm = puntos_px.astype(float) * escala
        puntos_cm[:, 1] = puntos_cm[:, 1].max() - puntos_cm[:, 1]

        colores = np.array([
            img[min(pt[1], img.shape[0]-1), min(pt[0], img.shape[1]-1)][::-1]
            for pt in puntos_px
        ])

        xn, xx = puntos_cm[:,0].min(), puntos_cm[:,0].max()
        yn, yx = puntos_cm[:,1].min(), puntos_cm[:,1].max()
        largo = xx - xn
        alto = yx - yn
        area_cm2 = area_px * escala**2

        # PLY lateral
        ply_lat = output_dir / f"foto_{i+1:02d}_lateral.ply"
        guardar_ply(str(ply_lat), puntos_cm, tris, colores, simetrico=False, escala=escala)

        # PLY 3D simétrico
        ply_3d = output_dir / f"foto_{i+1:02d}_3d.ply"
        pts_3d, _ = guardar_ply(str(ply_3d), puntos_cm, tris, colores, simetrico=True, escala=escala)

        try:
            hull = ConvexHull(pts_3d)
            vol_cm3 = hull.volume
            vol_litros = vol_cm3 / 1000.0
        except:
            vol_cm3 = vol_litros = 0

        k_litro = PESO_KG / vol_litros if PESO_KG and vol_litros > 0 else 0
        k_area = PESO_KG / area_cm2 if PESO_KG and area_cm2 > 0 else 0

        modelos.append({
            'idx': i+1, 'foto': foto.name, 'img': img, 'bbox': bbox,
            'mask': mask, 'puntos_px': puntos_px, 'puntos_cm': puntos_cm,
            'tris': tris, 'colores': colores, 'escala': escala,
            'largo': largo, 'alto': alto, 'area_cm2': area_cm2,
            'vol_litros': vol_litros, 'k_litro': k_litro, 'k_area': k_area,
        })

        print(f"  Foto {i+1:2d}: Largo={largo:.0f}cm Alto={alto:.0f}cm Área={area_cm2:.0f}cm² Vol={vol_litros:.0f}L K={k_litro:.4f}")

    # ═══════════════════════════════════════
    # VISUALIZACIÓN: GALERÍA DE TODOS LOS MODELOS
    # ═══════════════════════════════════════
    n = len(modelos)
    cols = 4
    rows = (n + cols - 1) // cols

    # Panel 1: Modelos con textura
    fig1, axes1 = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    peso_str = f"{PESO_KG} kg" if PESO_KG else "peso pendiente"
    fig1.suptitle(f'{VACA_ID.upper()} ({peso_str}, {ALTO_REAL_CM}cm) - Modelos con Textura', fontsize=14, fontweight='bold')
    axes1 = axes1.flatten() if n > cols else (axes1 if n > 1 else [axes1])

    for j, m in enumerate(modelos):
        ax = axes1[j]
        ax.set_facecolor('black')
        polys = [m['puntos_cm'][t] for t in m['tris']]
        fcolors = [(m['colores'][t]/255.0).mean(axis=0) for t in m['tris']]
        ax.add_collection(PolyCollection(polys, facecolors=fcolors, edgecolors='none', alpha=0.9))
        xn, xx = m['puntos_cm'][:,0].min(), m['puntos_cm'][:,0].max()
        yn, yx = m['puntos_cm'][:,1].min(), m['puntos_cm'][:,1].max()
        ax.set_xlim(xn-2, xx+2)
        ax.set_ylim(yn-2, yx+2)
        ax.set_title(f"#{m['idx']} L={m['largo']:.0f} A={m['alto']:.0f} K={m['k_litro']:.3f}", fontsize=9)
        ax.set_aspect('equal')
        ax.axis('off')

    for j in range(n, len(axes1)):
        axes1[j].axis('off')

    plt.tight_layout()
    fig1.savefig(str(output_dir / "GALERIA_texturas.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Panel 2: Wireframes
    fig2, axes2 = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    fig2.suptitle(f'{VACA_ID.upper()} ({peso_str}, {ALTO_REAL_CM}cm) - Wireframes', fontsize=14, fontweight='bold')
    axes2 = axes2.flatten() if n > cols else (axes2 if n > 1 else [axes2])

    for j, m in enumerate(modelos):
        ax = axes2[j]
        ax.triplot(m['puntos_cm'][:,0], m['puntos_cm'][:,1], m['tris'], color='saddlebrown', linewidth=0.3)
        xn, xx = m['puntos_cm'][:,0].min(), m['puntos_cm'][:,0].max()
        yn, yx = m['puntos_cm'][:,1].min(), m['puntos_cm'][:,1].max()
        ax.set_xlim(xn-2, xx+2)
        ax.set_ylim(yn-2, yx+2)
        ax.set_title(f"#{m['idx']} {m['foto'][-8:-4]} | {len(m['tris'])} tri", fontsize=9)
        ax.set_aspect('equal')
        ax.axis('off')

    for j in range(n, len(axes2)):
        axes2[j].axis('off')

    plt.tight_layout()
    fig2.savefig(str(output_dir / "GALERIA_wireframes.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # ═══════════════════════════════════════
    # RESUMEN ESTADÍSTICO
    # ═══════════════════════════════════════
    largos = [m['largo'] for m in modelos]
    altos = [m['alto'] for m in modelos]
    areas = [m['area_cm2'] for m in modelos]
    vols = [m['vol_litros'] for m in modelos]
    ks_litro = [m['k_litro'] for m in modelos if m['k_litro'] > 0]
    ks_area = [m['k_area'] for m in modelos if m['k_area'] > 0]

    print(f"\n{'='*60}")
    print(f"  RESUMEN {VACA_ID.upper()} - {len(modelos)} modelos")
    print(f"{'='*60}")
    print(f"  {'Métrica':<20} {'Promedio':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*60}")
    for nombre, vals in [('Largo (cm)', largos), ('Alto (cm)', altos),
                          ('Área (cm²)', areas), ('Volumen (L)', vols),
                          ('K peso/litro', ks_litro), ('K peso/área', ks_area)]:
        v = np.array(vals)
        print(f"  {nombre:<20} {v.mean():>10.1f} {v.std():>10.1f} {v.min():>10.1f} {v.max():>10.1f}")

    # JSON
    resumen_json = {
        'vaca': VACA_ID, 'peso_kg': PESO_KG, 'alto_real_cm': ALTO_REAL_CM,
        'num_modelos': len(modelos),
        'promedios': {
            'largo_cm': round(float(np.mean(largos)), 1),
            'alto_cm': round(float(np.mean(altos)), 1),
            'area_cm2': round(float(np.mean(areas)), 1),
            'volumen_litros': round(float(np.mean(vols)), 1),
            'k_peso_litro': round(float(np.mean(ks_litro)), 4) if ks_litro else 0,
            'k_peso_area': round(float(np.mean(ks_area)), 4) if ks_area else 0,
        },
        'modelos': [{
            'foto': m['foto'], 'largo_cm': round(m['largo'], 1),
            'alto_cm': round(m['alto'], 1), 'area_cm2': round(m['area_cm2'], 1),
            'vol_litros': round(m['vol_litros'], 1),
            'k_litro': round(m['k_litro'], 4), 'k_area': round(m['k_area'], 4),
            'num_triangulos': len(m['tris']),
        } for m in modelos],
    }
    with open(output_dir / f"todos_modelos_{VACA_ID}.json", 'w') as f:
        json.dump(resumen_json, f, indent=2, ensure_ascii=False)

    print(f"\n  Archivos en: {output_dir}/")
    print(f"    foto_XX_lateral.ply       - {len(modelos)} perfiles laterales")
    print(f"    foto_XX_3d.ply            - {len(modelos)} modelos 3D")
    print(f"    GALERIA_texturas.png      - todos los modelos con textura")
    print(f"    GALERIA_wireframes.png    - todos los wireframes")
    print(f"    todos_modelos_{VACA_ID}.json  - datos completos")


if __name__ == '__main__':
    main()
