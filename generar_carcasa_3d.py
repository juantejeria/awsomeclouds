"""
Generador de modelos 3D de CARCASA (barril) desde video.

Pipeline:
  1. Extraer frames del video (cada N frames)
  2. Detectar vaca → filtrar solo frames con animal completo
  3. Segmentar barril con barril_seg.pt (modelo entrenado)
  4. Fusionar máscaras de barril de todos los frames válidos
  5. Generar modelo 3D (PLY) a resolución nativa
  6. Escala: altura real medida del individuo
"""

import cv2
import numpy as np
import sys
import os
import json
import statistics
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from scipy.spatial import Delaunay, ConvexHull
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


PROJECT = Path(__file__).parent


# ═══════════════════════════════════════
# Funciones base
# ═══════════════════════════════════════

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


def animal_completo(bbox, img_w, img_h, margin=10):
    """Verifica que el bbox no toque los bordes de la imagen."""
    x1, y1, x2, y2 = bbox
    return x1 > margin and y1 > margin and x2 < img_w - margin and y2 < img_h - margin


def segmentar_barril(img, bbox, barril_model):
    """Segmenta el barril usando barril_seg.pt. Cropea al bbox para mantener resolución."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    margin = 20
    cx1 = max(0, x1 - margin)
    cy1 = max(0, y1 - margin)
    cx2 = min(w, x2 + margin)
    cy2 = min(h, y2 + margin)
    crop = img[cy1:cy2, cx1:cx2]

    results = barril_model(crop, conf=0.25, verbose=False)
    if not results or len(results[0].boxes) == 0 or results[0].masks is None:
        return None

    masks = results[0].masks.data.cpu().numpy()
    crop_h, crop_w = crop.shape[:2]

    best_mask = None
    best_area = 0
    for m in masks:
        m_resized = cv2.resize(m, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        m_bin = (m_resized > 0.5).astype(np.uint8)
        area = np.sum(m_bin)
        if area > best_area:
            best_area = area
            best_mask = m_bin

    if best_mask is None or best_area < 100:
        return None

    # Pegar en imagen completa sin perder resolución
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[cy1:cy2, cx1:cx2] = best_mask * 255

    # Limpieza mínima
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask_full)
    cv2.drawContours(clean, [c], -1, 255, -1)

    return clean


def extraer_frames(video_path, step=5):
    """Extrae frames del video cada `step` frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append((idx, frame))
        idx += 1
    cap.release()
    return frames


# ═══════════════════════════════════════
# Sampleo y triangulación (resolución nativa)
# ═══════════════════════════════════════

def samplear(contorno, mask, n_borde=120, n_interior=80):
    pts = contorno.reshape(-1, 2)
    step = max(1, len(pts) // n_borde)
    border = pts[::step]

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return border, np.array([]).reshape(0, 2)

    cols = int(np.sqrt(n_interior) * 1.5) + 2
    rows = int(np.sqrt(n_interior)) + 2
    gx = np.linspace(xs.min(), xs.max(), cols + 2)[1:-1]
    gy = np.linspace(ys.min(), ys.max(), rows + 2)[1:-1]
    mgx, mgy = np.meshgrid(gx, gy)
    grid = np.column_stack([mgx.ravel(), mgy.ravel()]).astype(int)

    interior = []
    for pt in grid:
        if 0 <= pt[1] < mask.shape[0] and 0 <= pt[0] < mask.shape[1] and mask[pt[1], pt[0]] > 0:
            interior.append(pt)
    interior = np.array(interior) if interior else np.array([]).reshape(0, 2)

    return border, interior


def triangular(pts_borde, pts_interior, mask):
    if len(pts_borde) < 3:
        return None, []
    all_pts = np.vstack([pts_borde, pts_interior]) if len(pts_interior) > 0 else pts_borde
    all_pts = np.unique(all_pts, axis=0)
    if len(all_pts) < 4:
        return None, []

    tri = Delaunay(all_pts)
    valid = []
    for s in tri.simplices:
        cx, cy = all_pts[s].mean(axis=0).astype(int)
        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx] > 0:
            valid.append(s)
    return all_pts, np.array(valid) if valid else np.array([])


def profundidad_eliptica(puntos_cm):
    y_min, y_max = puntos_cm[:, 1].min(), puntos_cm[:, 1].max()
    alto = y_max - y_min
    if alto < 1:
        return np.zeros(len(puntos_cm))
    t = (puntos_cm[:, 1] - y_min) / alto
    perfil = np.sqrt(np.clip(t * (1 - t), 0, None)) * 2
    max_depth = alto * 0.45
    return perfil * max_depth


def volumen_por_rebanadas(mask, escala, n_slices=100):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 0, 0
    y_min, y_max = ys.min(), ys.max()
    total_vol = 0
    total_sup = 0
    for y in np.linspace(y_min, y_max, n_slices):
        y_int = int(y)
        if y_int >= mask.shape[0]:
            continue
        row = mask[y_int, :]
        cols = np.where(row > 0)[0]
        if len(cols) == 0:
            continue
        w_px = cols[-1] - cols[0] + 1
        w_cm = w_px * escala
        r_cm = w_cm / 2
        slice_area = np.pi * r_cm ** 2
        dy_cm = (y_max - y_min) / n_slices * escala
        total_vol += slice_area * dy_cm
        total_sup += 2 * np.pi * r_cm * dy_cm
    return total_vol / 1000.0, total_sup


def guardar_ply(path, puntos, tris, colores, simetrico=True):
    if simetrico:
        depths = profundidad_eliptica(puntos)
        pts_r = np.column_stack([puntos[:, 0], puntos[:, 1], depths])
        pts_l = np.column_stack([puntos[:, 0], puntos[:, 1], -depths])
        n = len(puntos)
        all_pts = np.vstack([pts_r, pts_l])
        all_colors = np.vstack([colores, colores])
        tris_r = tris
        tris_l = tris.copy()
        tris_l = n + tris_l[:, ::-1]
        all_tris = np.vstack([tris_r, tris_l])
    else:
        all_pts = np.column_stack([puntos[:, 0], puntos[:, 1], np.zeros(len(puntos))])
        all_colors = colores
        all_tris = tris

    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(all_tris)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for pt, c in zip(all_pts, all_colors):
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for t in all_tris:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")

    return all_pts, all_tris


# ═══════════════════════════════════════
# Fusión de múltiples frames → un modelo 3D
# ═══════════════════════════════════════

def fusionar_barriles(frames_data, cow_height_cm):
    """Fusiona máscaras de barril de múltiples frames en un solo modelo 3D.
    Usa el bbox como referencia de escala (cow_height_cm = alto del bbox).
    Mantiene la resolución nativa del video.
    """
    n = len(frames_data)
    if n == 0:
        return None

    # Encontrar la resolución del canvas: usar el bbox más grande
    max_bbox_h = max(fd['bbox'][3] - fd['bbox'][1] for fd in frames_data)
    max_bbox_w = max(fd['bbox'][2] - fd['bbox'][0] for fd in frames_data)

    # Canvas al tamaño del bbox más grande (sin perder resolución)
    CANVAS_H = max_bbox_h
    CANVAS_W = max_bbox_w

    # Acumular máscaras normalizadas al canvas
    acumulador = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    masks_ok = 0

    for fd in frames_data:
        x1, y1, x2, y2 = fd['bbox']
        mask = fd['mask_barril']
        # Crop de la máscara al bbox
        crop = mask[y1:y2, x1:x2]
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        # Resize al canvas (mantiene proporciones del bbox)
        resized = cv2.resize(crop, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_NEAREST)
        acumulador += (resized > 0).astype(np.float32)
        masks_ok += 1

    if masks_ok == 0:
        return None

    # Umbral: aparecer en al menos 30% de los frames
    umbral = max(1, masks_ok * 0.3)
    mask_fusion = (acumulador >= umbral).astype(np.uint8) * 255

    # Limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_fusion = cv2.morphologyEx(mask_fusion, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_fusion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contorno = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contorno) < 200:
        return None

    mask_clean = np.zeros_like(mask_fusion)
    cv2.drawContours(mask_clean, [contorno], -1, 255, -1)

    # Escala: alto del canvas = alto del bbox = cow_height_cm
    escala = cow_height_cm / CANVAS_H

    # Samplear y triangular
    pts_b, pts_i = samplear(contorno, mask_clean, n_borde=150, n_interior=100)
    puntos_canvas, tris = triangular(pts_b, pts_i, mask_clean)
    if puntos_canvas is None or len(tris) == 0:
        return None

    # Escalar a cm
    puntos_cm = puntos_canvas.astype(float) * escala
    puntos_cm[:, 1] = puntos_cm[:, 1].max() - puntos_cm[:, 1]

    # Colores de la mejor foto (la del bbox más grande)
    mejor = max(frames_data, key=lambda fd: (fd['bbox'][2]-fd['bbox'][0]) * (fd['bbox'][3]-fd['bbox'][1]))
    img_ref = mejor['img']
    bx1, by1, bx2, by2 = mejor['bbox']

    colores = []
    for pt in puntos_canvas:
        img_x = int(bx1 + pt[0] * (bx2 - bx1) / CANVAS_W)
        img_y = int(by1 + pt[1] * (by2 - by1) / CANVAS_H)
        img_x = max(0, min(img_x, img_ref.shape[1] - 1))
        img_y = max(0, min(img_y, img_ref.shape[0] - 1))
        b, g, r = img_ref[img_y, img_x]
        colores.append([int(r), int(g), int(b)])
    colores = np.array(colores)

    # Volumen
    vol_litros, sup_cm2 = volumen_por_rebanadas(mask_clean, escala)

    x_min, x_max = puntos_cm[:, 0].min(), puntos_cm[:, 0].max()
    y_min, y_max = puntos_cm[:, 1].min(), puntos_cm[:, 1].max()

    return {
        'puntos_cm': puntos_cm,
        'tris': tris,
        'colores': colores,
        'mask_fusion': mask_clean,
        'acumulador': acumulador,
        'escala': escala,
        'canvas_w': CANVAS_W,
        'canvas_h': CANVAS_H,
        'largo_cm': round(x_max - x_min, 1),
        'alto_cm': round(y_max - y_min, 1),
        'vol_litros': round(vol_litros, 1),
        'superficie_cm2': round(sup_cm2, 1),
        'num_triangulos': len(tris),
        'frames_usados': masks_ok,
    }


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    dataset_dir = PROJECT / 'checkpoints' / 'Dataset Modelo 3d "grandes" '
    output_dir = PROJECT / "output_carcasa_3d"
    output_dir.mkdir(exist_ok=True)

    # Alturas reales
    alturas = {}
    alturas_path = PROJECT / "alturas_individuos.json"
    if alturas_path.exists():
        with open(alturas_path) as f:
            alturas = json.load(f).get('alturas_cm', {})
        print(f"Alturas reales: {alturas}")

    # Modelos
    print("\nCargando modelos...")
    cow_model = YOLO(str(PROJECT / "models_yolo" / "cow.pt"))
    coco_model = YOLO(str(PROJECT / "yolov8n.pt"))
    barril_model = YOLO(str(PROJECT / "barril_seg.pt"))
    print("  cow.pt + barril_seg.pt cargados\n")

    # Buscar videos en carpetas 3D_modelo_*
    resumen_global = []

    for ind_dir in sorted(dataset_dir.iterdir()):
        if not ind_dir.is_dir():
            continue
        nombre = ind_dir.name

        # Buscar video en subcarpeta 3D_modelo_*
        video_path = None
        for sub in ind_dir.iterdir():
            if sub.is_dir() and sub.name.lower().startswith('3d_modelo'):
                for f in sub.iterdir():
                    if f.suffix.lower() in ('.mov', '.mp4'):
                        video_path = f
                        break
            if video_path:
                break

        if not video_path:
            continue

        # Parsear peso del nombre
        parts = nombre.split('_')
        peso_real = int(parts[1]) if len(parts) >= 2 else 0
        cow_height_cm = alturas.get(nombre, 120)

        print(f"{'='*60}")
        print(f"  {nombre.upper()}")
        print(f"  Peso real: {peso_real} kg | Altura real: {cow_height_cm} cm")
        print(f"  Video: {video_path.name}")
        print(f"{'='*60}")

        # Extraer frames (cada 5 frames del video)
        all_frames = extraer_frames(video_path, step=5)
        print(f"  Frames extraídos: {len(all_frames)}")

        # Directorio para debug de frames
        ind_output = output_dir / nombre
        ind_output.mkdir(exist_ok=True)
        debug_dir = ind_output / "frames_barril"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Procesar cada frame
        frames_validos = []
        frames_incompletos = 0
        frames_sin_barril = 0

        for frame_idx, img in all_frames:
            bbox = detectar_vaca(img, cow_model, coco_model)
            if bbox is None:
                continue

            img_h, img_w = img.shape[:2]
            if not animal_completo(bbox, img_w, img_h):
                frames_incompletos += 1
                continue

            # Filtrar frames donde el animal está muy lejos (bbox < 150px alto)
            bbox_h = bbox[3] - bbox[1]
            if bbox_h < 150:
                frames_incompletos += 1
                continue

            mask_barril = segmentar_barril(img, bbox, barril_model)
            if mask_barril is None:
                frames_sin_barril += 1
                continue

            x1, y1, x2, y2 = bbox
            bbox_h_px = y2 - y1
            escala_frame = cow_height_cm / bbox_h_px if bbox_h_px > 0 else 0
            vol_frame, _ = volumen_por_rebanadas(mask_barril, escala_frame)

            # Guardar imagen debug: foto + barril overlay + info
            vis = img.copy()
            overlay = np.zeros_like(img)
            overlay[mask_barril > 0] = [0, 200, 0]
            vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
            ct, _ = cv2.findContours(mask_barril, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, ct, -1, (0, 255, 0), 2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # Info
            barril_area = np.count_nonzero(mask_barril)
            full_area = (x2-x1) * (y2-y1)
            info = f'F{frame_idx} | bbox:{bbox_h_px}px | esc:{escala_frame:.3f} | vol:{vol_frame:.1f}L | {vol_frame*1.03:.0f}kg'
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(debug_dir / f"frame_{frame_idx:05d}.jpg"), vis)

            frames_validos.append({
                'frame_idx': frame_idx,
                'img': img,
                'bbox': bbox,
                'mask_barril': mask_barril,
                'escala': escala_frame,
                'vol_litros': vol_frame,
            })

        print(f"  Frames válidos: {len(frames_validos)} | Incompletos: {frames_incompletos} | Sin barril: {frames_sin_barril}")
        print(f"  Debug frames en: {debug_dir}/")

        if len(frames_validos) > 0:
            vols = [fd['vol_litros'] for fd in frames_validos]
            escalas = [fd['escala'] for fd in frames_validos]
            print(f"\n  Volúmenes por frame:")
            print(f"    Min: {min(vols):.1f}L | Max: {max(vols):.1f}L | Mean: {statistics.mean(vols):.1f}L | Median: {statistics.median(vols):.1f}L")
            if len(vols) > 1:
                print(f"    Std: {statistics.stdev(vols):.1f}L")
            print(f"  Escalas por frame:")
            print(f"    Min: {min(escalas):.4f} | Max: {max(escalas):.4f} | Mean: {statistics.mean(escalas):.4f}")
            # Top 5 y bottom 5 por volumen
            sorted_frames = sorted(frames_validos, key=lambda f: f['vol_litros'])
            bottom5 = [f"F{f['frame_idx']}={f['vol_litros']:.0f}L(esc={f['escala']:.3f})" for f in sorted_frames[:5]]
            top5 = [f"F{f['frame_idx']}={f['vol_litros']:.0f}L(esc={f['escala']:.3f})" for f in sorted_frames[-5:]]
            print(f"  5 menores vol: {bottom5}")
            print(f"  5 mayores vol: {top5}")

        if len(frames_validos) == 0:
            print(f"  SKIP: ningún frame válido\n")
            continue

        # Fusionar todos los barriles en un modelo 3D
        fusion = fusionar_barriles(frames_validos, cow_height_cm)
        if fusion is None:
            print(f"  ERROR: fusión falló\n")
            continue

        # Guardar
        ind_output = output_dir / nombre
        ind_output.mkdir(exist_ok=True)

        # PLY lateral
        ply_lat = ind_output / f"{nombre}_carcasa_lateral.ply"
        guardar_ply(str(ply_lat), fusion['puntos_cm'], fusion['tris'], fusion['colores'], simetrico=False)

        # PLY 3D
        ply_3d = ind_output / f"{nombre}_carcasa_3d.ply"
        pts_3d, _ = guardar_ply(str(ply_3d), fusion['puntos_cm'], fusion['tris'], fusion['colores'], simetrico=True)

        # Volumen del PLY 3D
        try:
            hull = ConvexHull(pts_3d)
            vol_3d = hull.volume / 1000.0
        except Exception:
            vol_3d = fusion['vol_litros']

        peso_carcasa = vol_3d * 1.03

        resumen = {
            'individuo': nombre,
            'peso_real_kg': peso_real,
            'altura_real_cm': cow_height_cm,
            'frames_video': len(all_frames),
            'frames_validos': len(frames_validos),
            'frames_incompletos': frames_incompletos,
            'vol_carcasa_litros': round(vol_3d, 1),
            'peso_carcasa_kg': round(peso_carcasa, 1),
            'largo_cm': fusion['largo_cm'],
            'escala_cm_px': round(fusion['escala'], 6),
            'canvas': f"{fusion['canvas_w']}x{fusion['canvas_h']}",
        }
        resumen_global.append(resumen)

        with open(ind_output / f"{nombre}_resumen.json", 'w') as f:
            json.dump(resumen, f, indent=2, ensure_ascii=False)

        error_50 = (peso_carcasa - peso_real * 0.5) / peso_real * 100
        print(f"\n  RESULTADO {nombre.upper()}:")
        print(f"    Frames usados:  {len(frames_validos)}")
        print(f"    Vol carcasa:    {vol_3d:.1f} L → {peso_carcasa:.1f} kg")
        print(f"    Target 50%:     {peso_real*0.5:.0f} kg (error: {error_50:+.1f}%)")
        print(f"    Largo:          {fusion['largo_cm']:.1f} cm")
        print(f"    PLY:            {ply_3d}\n")

    # Resumen global
    print(f"\n{'#'*60}")
    print(f"  RESUMEN GLOBAL - {len(resumen_global)} individuos")
    print(f"{'#'*60}")
    print(f"\n  {'Individuo':<15} {'Peso Real':>10} {'Frames':>7} {'Vol(L)':>8} {'Peso Carc':>10} {'Target':>8} {'Error':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*7} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for r in resumen_global:
        target = r['peso_real_kg'] * 0.5
        error = (r['peso_carcasa_kg'] - target) / r['peso_real_kg'] * 100
        print(f"  {r['individuo']:<15} {r['peso_real_kg']:>8} kg {r['frames_validos']:>7} {r['vol_carcasa_litros']:>8.1f} {r['peso_carcasa_kg']:>8.1f} kg {target:>6.0f} kg {error:>+7.1f}%")

    with open(output_dir / "resumen_global.json", 'w') as f:
        json.dump(resumen_global, f, indent=2, ensure_ascii=False)
    print(f"\n  Archivos en: {output_dir}/")


if __name__ == '__main__':
    main()
