"""
Procesa videos del dataset "grandes" para generar modelos 3D.
Extrae frames donde el animal se mueve/gira, usa YOLO-seg para segmentar,
y corre SfM para reconstruir el volumen 3D.
"""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from pathlib import Path
from reconstruccion_3d import sfm_real_desde_frames
from generar_modelos3d_grandes import (
    detectar_vaca, segmentar_yolo_seg, recortar_torso,
    volumen_por_rebanadas, parsear_nombre
)
import json
import matplotlib
matplotlib.use('Agg')


def detectar_vaca_video(img, seg_model, coco_model):
    """Detecta vaca en frame de video usando seg_model o coco_model como fallback.
    cow.pt no funciona bien en videos 1080p, así que usamos COCO.
    Retorna bbox [x1,y1,x2,y2] o None.
    """
    # Intentar con seg_model (también da bbox)
    results = seg_model(img, conf=0.15, classes=[19], verbose=False)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        areas = [(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]) for b in boxes]
        return boxes[int(np.argmax(areas))].xyxy[0].cpu().numpy().astype(int)

    # Fallback a coco
    results = coco_model(img, conf=0.15, classes=[19], verbose=False)
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        areas = [(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]) for b in boxes]
        return boxes[int(np.argmax(areas))].xyxy[0].cpu().numpy().astype(int)

    return None


def extraer_frames_video(video_path, coco_model, seg_model, max_frames=12, sample_interval=None):
    """Extrae frames del video donde la vaca está bien detectada y con variación angular.

    Selecciona frames espaciados y con diferencias en la silueta (animal girando).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

    # Intervalo de sampleo: ~2 frames por segundo para captar movimiento
    if sample_interval is None:
        sample_interval = max(1, int(fps / 2))

    # Primera pasada: extraer todos los frames candidatos
    candidatos = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            # Detectar vaca
            bbox = detectar_vaca_video(frame, seg_model, coco_model)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)

                # Segmentar con YOLO-seg
                mask, contorno = segmentar_yolo_seg(frame, bbox, seg_model)
                if mask is not None and contorno is not None:
                    mask_area = np.count_nonzero(mask)
                    if mask_area > 5000:  # mínimo de área
                        candidatos.append({
                            'frame_idx': frame_idx,
                            'frame': frame.copy(),
                            'bbox': bbox,
                            'mask': mask,
                            'contorno': contorno,
                            'mask_area': mask_area,
                            'bbox_area': area,
                        })

        frame_idx += 1

    cap.release()

    if len(candidatos) == 0:
        return []

    print(f"    {len(candidatos)} frames candidatos de {total_frames} totales")

    # Segunda pasada: seleccionar frames con mayor variación de silueta
    if len(candidatos) <= max_frames:
        return candidatos

    # Calcular "diferencia de silueta" entre frames consecutivos
    # Frames con más diferencia = animal giró/se movió
    diffs = [0]
    for i in range(1, len(candidatos)):
        m1 = cv2.resize(candidatos[i-1]['mask'], (200, 150))
        m2 = cv2.resize(candidatos[i]['mask'], (200, 150))
        diff = np.sum(np.abs(m1.astype(float) - m2.astype(float))) / m1.size
        diffs.append(diff)

    # Seleccionar: primer frame + frames con mayor diferencia + último frame
    indices = [0]  # siempre incluir primero

    # Ordenar por diferencia, tomar los que más cambiaron
    ranked = sorted(range(1, len(diffs)), key=lambda i: diffs[i], reverse=True)
    for idx in ranked:
        if len(indices) >= max_frames - 1:
            break
        # No tomar frames muy cercanos entre sí
        if all(abs(idx - sel) >= 2 for sel in indices):
            indices.append(idx)

    indices.append(len(candidatos) - 1)  # siempre incluir último
    indices = sorted(set(indices))[:max_frames]

    selected = [candidatos[i] for i in indices]
    print(f"    Seleccionados {len(selected)} frames (de {len(candidatos)} candidatos)")

    return selected


def procesar_video_individuo(video_path, coco_model, seg_model, cow_height_cm, nombre):
    """Procesa un video de un individuo y retorna resultado SfM + volumen por rebanadas."""

    print(f"\n  Procesando video: {video_path.name}")

    # Extraer frames
    frames_data = extraer_frames_video(video_path, coco_model, seg_model)
    if len(frames_data) < 3:
        print(f"    Solo {len(frames_data)} frames válidos, insuficiente")
        return None

    frames = [f['frame'] for f in frames_data]
    masks_full = [f['mask'] for f in frames_data]
    bboxes = [f['bbox'] for f in frames_data]

    # Masks de torso para barril
    masks_torso = []
    for fd in frames_data:
        mt, _ = recortar_torso(fd['mask'], fd['bbox'])
        masks_torso.append(mt)

    # ── SfM ──
    print(f"    Corriendo SfM con {len(frames)} frames...")
    sfm_result = sfm_real_desde_frames(
        frames, masks_torso, cow_height_cm,
        bboxes=bboxes, masks_full=masks_full
    )

    # ── Volumen por rebanadas (promedio de todos los frames) ──
    vols_rebanadas = []
    vols_barril_reb = []
    for fd in frames_data:
        bbox = fd['bbox']
        bbox_h = bbox[3] - bbox[1]
        if bbox_h < 20:
            continue
        escala = cow_height_cm / bbox_h
        try:
            vol, _ = volumen_por_rebanadas(fd['mask'], escala)
            vols_rebanadas.append(vol)
        except:
            pass
        mt, _ = recortar_torso(fd['mask'], bbox)
        if mt is not None:
            try:
                vol_b, _ = volumen_por_rebanadas(mt, escala)
                vols_barril_reb.append(vol_b)
            except:
                pass

    vol_reb_avg = np.mean(vols_rebanadas) if vols_rebanadas else 0
    vol_reb_std = np.std(vols_rebanadas) if len(vols_rebanadas) > 1 else 0
    vol_bar_avg = np.mean(vols_barril_reb) if vols_barril_reb else 0

    return {
        'sfm': sfm_result,
        'n_frames': len(frames),
        'vol_rebanadas_avg': round(vol_reb_avg, 1),
        'vol_rebanadas_std': round(vol_reb_std, 1),
        'vol_barril_reb_avg': round(vol_bar_avg, 1),
        'n_mediciones_reb': len(vols_rebanadas),
        'vols_individuales': [round(v, 1) for v in vols_rebanadas],
    }


def main():
    project = Path(__file__).parent
    dataset_dir = project / 'checkpoints' / 'Dataset Modelo 3d "grandes" '
    output_dir = project / "output_modelos3d_grandes" / "_video_sfm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar alturas
    with open(project / "alturas_individuos.json") as f:
        alturas = json.load(f)['alturas_cm']

    print("Cargando modelos...")
    coco_model = YOLO(str(project / "yolov8n.pt"))
    seg_model = YOLO(str(project / "yolov8n-seg.pt"))

    # Buscar individuos con videos en 3d_modelo_*
    individuos = []
    for ind_dir in sorted(dataset_dir.iterdir()):
        if not ind_dir.is_dir() or ind_dir.name not in alturas:
            continue
        for sub in ind_dir.iterdir():
            if sub.is_dir() and sub.name.lower().startswith('3d_modelo'):
                videos = sorted([f for f in sub.iterdir() if f.suffix.lower() in ('.mov', '.mp4')])
                if videos:
                    individuos.append((ind_dir, sub, videos))
                break

    print(f"\nIndividuos con videos: {len(individuos)}")

    resultados = []
    for ind_dir, fotos_dir, videos in individuos:
        nombre = ind_dir.name
        categoria, peso_real, meses = parsear_nombre(nombre)
        cow_height = alturas[nombre]

        print(f"\n{'='*60}")
        print(f"  {nombre.upper()} | Peso real: {peso_real} kg | Alto: {cow_height} cm")
        print(f"  Videos: {len(videos)}")
        print(f"{'='*60}")

        best_result = None
        for video in videos:
            result = procesar_video_individuo(video, coco_model, seg_model, cow_height, nombre)
            if result is not None:
                if best_result is None or result['n_frames'] > best_result['n_frames']:
                    best_result = result

        if best_result is None:
            print(f"  SKIP: no se pudo procesar ningún video")
            continue

        sfm = best_result['sfm']
        sfm_vol = sfm.get('volumen_litros', 0) if sfm else 0
        sfm_peso = sfm.get('peso_kg', 0) if sfm else 0
        sfm_ancho = sfm.get('ancho_cm', 0) if sfm else 0
        sfm_pts = sfm.get('num_points', 0) if sfm else 0
        sfm_pairs = sfm.get('num_pairs', 0) if sfm else 0
        sfm_method = sfm.get('method', '?') if sfm else '?'

        r = {
            'individuo': nombre,
            'peso_real_kg': peso_real,
            'altura_cm': cow_height,
            'n_frames': best_result['n_frames'],
            'sfm_method': sfm_method,
            'sfm_vol_litros': sfm_vol,
            'sfm_peso_kg': sfm_peso,
            'sfm_ancho_cm': sfm_ancho,
            'sfm_puntos_3d': sfm_pts,
            'sfm_pares': sfm_pairs,
            'vol_rebanadas_avg': best_result['vol_rebanadas_avg'],
            'vol_rebanadas_std': best_result['vol_rebanadas_std'],
            'peso_rebanadas_kg': round(best_result['vol_rebanadas_avg'] * 1.03, 1),
            'vol_barril_reb_avg': best_result['vol_barril_reb_avg'],
            'peso_barril_reb_kg': round(best_result['vol_barril_reb_avg'] * 1.03, 1),
        }
        resultados.append(r)

        print(f"\n  RESULTADO {nombre.upper()}:")
        print(f"    SfM:       {sfm_method} | {sfm_pts} pts, {sfm_pairs} pares | vol={sfm_vol:.1f}L → {sfm_peso:.1f}kg")
        if sfm_ancho > 0:
            print(f"    Ancho 3D:  {sfm_ancho:.1f} cm")
        print(f"    Rebanadas: {best_result['vol_rebanadas_avg']:.1f}L (±{best_result['vol_rebanadas_std']:.1f}) → {r['peso_rebanadas_kg']:.1f}kg")
        print(f"    Barril:    {best_result['vol_barril_reb_avg']:.1f}L → {r['peso_barril_reb_kg']:.1f}kg")

    # Tabla resumen
    print(f"\n\n{'#'*70}")
    print(f"  RESUMEN - VIDEO SfM vs REBANADAS ({len(resultados)} individuos)")
    print(f"{'#'*70}")
    print(f"\n  {'Vaca':<12} {'Real':>6} {'SfM Vol':>8} {'SfM kg':>7} {'Err%':>6} {'Reb Vol':>8} {'Reb kg':>7} {'Err%':>6} {'Pts3D':>6}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*7} {'-'*6} {'-'*8} {'-'*7} {'-'*6} {'-'*6}")
    for r in resultados:
        peso = r['peso_real_kg']
        sfm_err = (r['sfm_peso_kg'] - peso) / peso * 100 if r['sfm_peso_kg'] > 0 else 0
        reb_err = (r['peso_rebanadas_kg'] - peso) / peso * 100
        name = r['individuo'].replace('vaca_','').replace('_36','')
        print(f"  {name:<12} {peso:>6} {r['sfm_vol_litros']:>8.1f} {r['sfm_peso_kg']:>7.1f} {sfm_err:>+5.0f}% "
              f"{r['vol_rebanadas_avg']:>8.1f} {r['peso_rebanadas_kg']:>7.1f} {reb_err:>+5.0f}% {r['sfm_puntos_3d']:>6}")

    # Guardar JSON
    with open(output_dir / "resumen_video_sfm.json", 'w') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\n  Guardado en: {output_dir}/resumen_video_sfm.json")


if __name__ == '__main__':
    main()
