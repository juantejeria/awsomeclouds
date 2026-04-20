"""
Genera modelos 3D desde video usando SfM multi-frame.
Usa muchos frames del video para reconstruir una nube de puntos 3D del animal.

Pipeline:
1. Extraer frames del video (cada N frames)
2. YOLO-seg para segmentar la vaca en cada frame
3. SIFT matching entre frames consecutivos (imagen completa, incluye fondo)
4. Estimar poses de cámara relativas
5. Triangular puntos 3D que caen dentro de la máscara de la vaca
6. Escalar con altura conocida
7. Calcular volumen y generar modelo
"""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from scipy.spatial import Delaunay, ConvexHull
from pathlib import Path
from generar_modelos3d_grandes import parsear_nombre, volumen_por_rebanadas, recortar_torso
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def extraer_frames(video_path, seg_model, frame_interval=5, max_frames=60):
    """Extrae frames del video con detección de vaca."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    masks = []
    bboxes = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            r = seg_model(frame, conf=0.15, classes=[19], verbose=False)
            if r and len(r[0].boxes) > 0 and r[0].masks is not None:
                boxes = r[0].boxes
                ms = r[0].masks.data.cpu().numpy()
                areas = [(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]) for b in boxes]
                best = int(np.argmax(areas))
                bbox = boxes[best].xyxy[0].cpu().numpy().astype(int)

                h, w = frame.shape[:2]
                m = cv2.resize(ms[best], (w, h))
                mask = (m > 0.5).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

                frames.append(frame)
                masks.append(mask)
                bboxes.append(bbox)

                if len(frames) >= max_frames:
                    break
        frame_idx += 1

    cap.release()
    print(f"    Extraidos {len(frames)} frames de {total} ({fps:.0f}fps, intervalo={frame_interval})")
    return frames, masks, bboxes


def sfm_multiframe(frames, masks, bboxes, cow_height_cm):
    """SfM real usando matching de imagen completa + filtrado por máscara de vaca."""
    n = len(frames)
    if n < 3:
        print("    SfM: muy pocos frames")
        return None

    h, w = frames[0].shape[:2]
    focal = max(h, w) * 1.2
    K = np.array([[focal, 0, w/2.0], [0, focal, h/2.0], [0, 0, 1]], dtype=np.float64)

    sift = cv2.SIFT_create(nfeatures=3000)
    bf = cv2.BFMatcher()

    # ── Paso 1: Match consecutivos y triangular ──
    all_points_3d = []
    all_colors = []
    all_is_cow = []
    n_pairs_ok = 0

    # Procesar pares con diferentes separaciones: consecutivos, skip 1, skip 2
    pair_configs = []
    for skip in [1, 2, 3]:
        for i in range(n - skip):
            pair_configs.append((i, i + skip))

    print(f"    SfM: procesando {len(pair_configs)} pares de frames...")

    for idx1, idx2 in pair_configs:
        gray1 = cv2.cvtColor(frames[idx1], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[idx2], cv2.COLOR_BGR2GRAY)

        kp1, d1 = sift.detectAndCompute(gray1, None)
        kp2, d2 = sift.detectAndCompute(gray2, None)
        if d1 is None or d2 is None:
            continue

        matches = bf.knnMatch(d1, d2, k=2)
        good = [m for m, nn in matches if m.distance < 0.7 * nn.distance]

        if len(good) < 20:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Essential matrix
        E, mask_e = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask_e)
        inliers = mask_pose.ravel() > 0
        pts1_in = pts1[inliers]
        pts2_in = pts2[inliers]

        if inliers.sum() < 10:
            continue

        # Triangulate
        P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
        P2 = K @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        # Filtrar puntos detrás de la cámara
        z_ok = pts3d[:, 2] > 0
        pts3d = pts3d[z_ok]
        pts1_filt = pts1_in[z_ok]
        pts2_filt = pts2_in[z_ok]

        if len(pts3d) == 0:
            continue

        # Filtrar outliers (distancia al centroide)
        centroid = np.median(pts3d, axis=0)
        dists = np.linalg.norm(pts3d - centroid, axis=1)
        p95 = np.percentile(dists, 95)
        keep = dists < p95 * 2
        pts3d = pts3d[keep]
        pts1_filt = pts1_filt[keep]
        pts2_filt = pts2_filt[keep]

        if len(pts3d) < 5:
            continue

        # Marcar cuáles caen dentro de la máscara de la vaca
        m1 = masks[idx1]
        m2 = masks[idx2]
        for j in range(len(pts3d)):
            px1, py1 = int(pts1_filt[j][0]), int(pts1_filt[j][1])
            px2, py2 = int(pts2_filt[j][0]), int(pts2_filt[j][1])
            in_cow1 = (0 <= py1 < h and 0 <= px1 < w and m1[py1, px1] > 0)
            in_cow2 = (0 <= py2 < h and 0 <= px2 < w and m2[py2, px2] > 0)

            all_points_3d.append(pts3d[j])
            # Color from frame 2
            if 0 <= py2 < h and 0 <= px2 < w:
                b, g, r = frames[idx2][py2, px2]
                all_colors.append([int(r), int(g), int(b)])
            else:
                all_colors.append([128, 128, 128])
            all_is_cow.append(in_cow1 or in_cow2)

        n_pairs_ok += 1

    if n_pairs_ok == 0 or len(all_points_3d) == 0:
        print("    SfM: no se pudo reconstruir")
        return None

    all_points_3d = np.array(all_points_3d)
    all_colors = np.array(all_colors)
    all_is_cow = np.array(all_is_cow)

    print(f"    SfM: {n_pairs_ok} pares OK, {len(all_points_3d)} puntos total, {all_is_cow.sum()} en vaca")

    # ── Paso 2: Filtrar solo puntos de la vaca ──
    cow_pts = all_points_3d[all_is_cow]
    cow_colors = all_colors[all_is_cow]

    if len(cow_pts) < 20:
        print(f"    SfM: muy pocos puntos en la vaca ({len(cow_pts)})")
        return None

    # Filtrar outliers finales
    centroid = np.median(cow_pts, axis=0)
    dists = np.linalg.norm(cow_pts - centroid, axis=1)
    p90 = np.percentile(dists, 90)
    keep = dists < p90 * 1.5
    cow_pts = cow_pts[keep]
    cow_colors = cow_colors[keep]

    print(f"    SfM: {len(cow_pts)} puntos de vaca después de filtro")

    # ── Paso 3: Escalar a cm ──
    # Usar la extensión en el eje de mayor varianza como "largo" o "alto"
    centered = cow_pts - cow_pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigenvalues sorted ascending: [smallest, middle, largest]
    # Asumir: smallest = ancho, middle = alto, largest = largo
    extents = []
    for i in range(3):
        proj = centered @ eigenvectors[:, i]
        extents.append(proj.max() - proj.min())

    # El "alto" es la extensión media (ni la más larga ni la más corta)
    sorted_extents = sorted(enumerate(extents), key=lambda x: x[1])
    alto_idx = sorted_extents[1][0]
    alto_sfm = sorted_extents[1][1]

    if alto_sfm < 0.001:
        print("    SfM: extensión de altura ≈ 0")
        return None

    scale = cow_height_cm / alto_sfm
    cow_pts_cm = cow_pts * scale

    # Recalcular dimensiones
    centered_cm = cow_pts_cm - cow_pts_cm.mean(axis=0)
    extents_cm = []
    for i in range(3):
        proj = centered_cm @ eigenvectors[:, i]
        extents_cm.append(proj.max() - proj.min())
    sorted_cm = sorted(extents_cm)
    ancho_cm = sorted_cm[0]
    alto_cm = sorted_cm[1]
    largo_cm = sorted_cm[2]

    # ── Paso 4: Volumen ──
    vol_litros = 0
    sup_cm2 = 0
    try:
        hull = ConvexHull(cow_pts_cm)
        vol_litros = hull.volume / 1000.0
        sup_cm2 = hull.area
    except:
        pass

    print(f"    SfM: largo={largo_cm:.0f}cm alto={alto_cm:.0f}cm ancho={ancho_cm:.0f}cm vol={vol_litros:.0f}L")

    return {
        'points_cm': cow_pts_cm,
        'colors': cow_colors,
        'n_points': len(cow_pts_cm),
        'n_pairs': n_pairs_ok,
        'largo_cm': round(largo_cm, 1),
        'alto_cm': round(alto_cm, 1),
        'ancho_cm': round(ancho_cm, 1),
        'vol_litros': round(vol_litros, 1),
        'sup_cm2': round(sup_cm2, 1),
        'scale': scale,
        'eigenvectors': eigenvectors,
        # Para visualización: usar el mejor frame
        'frames': frames,
        'masks': masks,
        'bboxes': bboxes,
    }


def generar_visualizacion_sfm(nombre, peso_real, altura_cm, result, vol_reb, output_path):
    """Genera imagen de resultado del modelo SfM."""
    fig = plt.figure(figsize=(24, 10))
    peso_str = f'{peso_real} kg' if peso_real > 0 else 'desconocido'
    fig.suptitle(f'MODELO 3D VIDEO - {nombre.upper()} ({peso_str}) | {result["n_points"]} pts, {result["n_pairs"]} pares',
                 fontsize=13, fontweight='bold')

    pts_cm = result['points_cm']
    colors = result['colors']
    ev = result['eigenvectors']

    # Mejor frame (mayor área de máscara)
    best_idx = max(range(len(result['masks'])), key=lambda i: np.count_nonzero(result['masks'][i]))
    best_frame = result['frames'][best_idx]
    best_mask = result['masks'][best_idx]
    best_bbox = result['bboxes'][best_idx]
    img_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)

    # 1. Frame + segmentación
    ax1 = fig.add_subplot(2, 4, 1)
    overlay = img_rgb.copy()
    overlay[best_mask > 0] = [0, 220, 0]
    ax1.imshow(cv2.addWeighted(img_rgb, 0.5, overlay, 0.5, 0))
    x1, y1, x2, y2 = best_bbox
    ax1.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='lime', lw=1.5))
    ax1.set_title(f'Mejor frame (de {len(result["frames"])})')
    ax1.axis('off')

    # 2. Nube de puntos 3D - vista lateral (XY)
    ax2 = fig.add_subplot(2, 4, 2)
    # Proyectar a los dos ejes principales
    centered = pts_cm - pts_cm.mean(axis=0)
    proj_xy = centered @ ev[:, 1:]  # middle and largest = alto y largo
    c_norm = colors / 255.0
    ax2.scatter(proj_xy[:, 1], proj_xy[:, 0], c=c_norm, s=2, alpha=0.7)
    ax2.set_aspect('equal')
    ax2.set_title(f'Vista lateral ({len(pts_cm)} pts)')
    ax2.set_xlabel('Largo (cm)')
    ax2.set_ylabel('Alto (cm)')

    # 3. Nube de puntos 3D - vista frontal (ancho x alto)
    ax3 = fig.add_subplot(2, 4, 3)
    proj_za = centered @ ev[:, :2]  # smallest and middle = ancho y alto
    ax3.scatter(proj_za[:, 0], proj_za[:, 1], c=c_norm, s=2, alpha=0.7)
    ax3.set_aspect('equal')
    ax3.set_title(f'Vista frontal (ancho={result["ancho_cm"]:.0f}cm)')
    ax3.set_xlabel('Ancho (cm)')
    ax3.set_ylabel('Alto (cm)')

    # 4. Vista superior (largo x ancho)
    ax4 = fig.add_subplot(2, 4, 4)
    proj_top = centered @ ev[:, [0, 2]]  # smallest and largest = ancho y largo
    ax4.scatter(proj_top[:, 1], proj_top[:, 0], c=c_norm, s=2, alpha=0.7)
    ax4.set_aspect('equal')
    ax4.set_title('Vista superior')
    ax4.set_xlabel('Largo (cm)')
    ax4.set_ylabel('Ancho (cm)')

    # 5. Modelo 2D del mejor frame (malla triangulada)
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.set_facecolor('black')
    # Triangular la máscara del mejor frame
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        ct = max(contours, key=cv2.contourArea)
        ct_pts = ct.reshape(-1, 2)
        pts_b = ct_pts[::max(1, len(ct_pts)//80)]
        ys_on, xs_on = np.where(best_mask > 0)
        if len(xs_on) > 0:
            cg = int(np.sqrt(60)*1.5)+2
            rg = int(np.sqrt(60))+2
            gx = np.linspace(xs_on.min(), xs_on.max(), cg+2)[1:-1]
            gy = np.linspace(ys_on.min(), ys_on.max(), rg+2)[1:-1]
            mgx, mgy = np.meshgrid(gx, gy)
            grid = np.column_stack([mgx.ravel(), mgy.ravel()]).astype(int)
            interior = [pt for pt in grid if 0<=pt[1]<best_mask.shape[0] and 0<=pt[0]<best_mask.shape[1] and best_mask[pt[1],pt[0]]>0]
            pts_i = np.array(interior) if interior else np.array([]).reshape(0,2)
            ppx = np.vstack([pts_b, pts_i]) if len(pts_i) > 0 else pts_b
            ppx = np.unique(ppx, axis=0)
            if len(ppx) >= 4:
                bbox_h = best_bbox[3] - best_bbox[1]
                esc = altura_cm / bbox_h
                pcm = ppx.astype(float) * esc
                pcm[:, 1] = pcm[:, 1].max() - pcm[:, 1]
                try:
                    tri = Delaunay(ppx)
                    valid = [s for s in tri.simplices
                             if best_mask[int(ppx[s].mean(axis=0)[1]), int(ppx[s].mean(axis=0)[0])] > 0]
                    if valid:
                        tris = np.array(valid)
                        col = np.array([best_frame[min(pt[1],best_frame.shape[0]-1), min(pt[0],best_frame.shape[1]-1)][::-1] for pt in ppx])
                        polys = [pcm[t] for t in tris]
                        fcolors = [(col[t]/255.0).mean(axis=0) for t in tris]
                        ax5.add_collection(PolyCollection(polys, facecolors=fcolors, edgecolors='none', alpha=0.9))
                        ax5.set_xlim(pcm[:,0].min()-3, pcm[:,0].max()+3)
                        ax5.set_ylim(pcm[:,1].min()-3, pcm[:,1].max()+3)
                except:
                    pass
    ax5.set_title('Modelo lateral (mejor frame)')
    ax5.set_aspect('equal')
    ax5.axis('off')

    # 6. Barril sobre foto
    ax6 = fig.add_subplot(2, 4, 6)
    mask_torso, _ = recortar_torso(best_mask, best_bbox)
    if mask_torso is not None:
        dark = (img_rgb * 0.3).astype(np.uint8)
        dark[mask_torso > 0] = img_rgb[mask_torso > 0]
        cts, _ = cv2.findContours(mask_torso, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dark_bgr = cv2.cvtColor(dark, cv2.COLOR_RGB2BGR)
        cv2.drawContours(dark_bgr, cts, -1, (255,255,0), 2)
        ax6.imshow(cv2.cvtColor(dark_bgr, cv2.COLOR_BGR2RGB))
    else:
        ax6.imshow(img_rgb)
    ax6.set_title('Barril')
    ax6.axis('off')

    # 7. Info
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.axis('off')

    vol_reb_str = f"{vol_reb['avg']:.1f}" if vol_reb else "-"
    peso_reb_str = f"{vol_reb['avg']*1.03:.1f}" if vol_reb else "-"
    vol_bar_str = f"{vol_reb['barril_avg']:.1f}" if vol_reb else "-"

    info = f"""{nombre.upper()} - MODELO 3D VIDEO

Peso real:       {peso_str}
Altura:          {altura_cm} cm

SfM ({result['n_points']} pts, {result['n_pairs']} pares):
  Largo:         {result['largo_cm']:.1f} cm
  Alto:          {result['alto_cm']:.1f} cm
  Ancho:         {result['ancho_cm']:.1f} cm
  Vol ConvHull:  {result['vol_litros']:.1f} L
  Peso x1.03:   {result['vol_litros']*1.03:.1f} kg

Rebanadas (promedio {len(result['frames'])} frames):
  Vol total:     {vol_reb_str} L
  Peso x1.03:   {peso_reb_str} kg
  Vol barril:    {vol_bar_str} L

Frames usados:   {len(result['frames'])}"""

    ax7.text(0.05, 0.95, info, transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 8. Histograma de anchos 3D
    ax8 = fig.add_subplot(2, 4, 8)
    # Ancho de la nube en rebanadas a lo largo del eje principal
    centered = pts_cm - pts_cm.mean(axis=0)
    proj_largo = centered @ ev[:, 2]  # proyección sobre eje más largo
    proj_ancho = centered @ ev[:, 0]  # proyección sobre eje más angosto
    n_slices = 20
    largo_min, largo_max = proj_largo.min(), proj_largo.max()
    slice_edges = np.linspace(largo_min, largo_max, n_slices + 1)
    slice_widths = []
    slice_centers = []
    for si in range(n_slices):
        in_slice = (proj_largo >= slice_edges[si]) & (proj_largo < slice_edges[si+1])
        if in_slice.sum() > 2:
            w = proj_ancho[in_slice].max() - proj_ancho[in_slice].min()
            slice_widths.append(w)
            slice_centers.append((slice_edges[si] + slice_edges[si+1]) / 2)
    if slice_widths:
        ax8.bar(slice_centers, slice_widths, width=(largo_max-largo_min)/n_slices*0.8, color='cyan', alpha=0.7)
        ax8.set_xlabel('Posicion a lo largo del cuerpo (cm)')
        ax8.set_ylabel('Ancho 3D (cm)')
        ax8.set_title('Perfil de ancho 3D por rebanada')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    project = Path(__file__).parent
    dataset = project / 'checkpoints' / 'Dataset Modelo 3d "grandes" '
    output = project / 'output_modelos3d_grandes' / '_video_3d'
    output.mkdir(parents=True, exist_ok=True)

    with open(project / 'alturas_individuos.json') as f:
        alturas = json.load(f)['alturas_cm']

    print("Cargando modelos...")
    seg_model = YOLO(str(project / "yolov8n-seg.pt"))

    resultados = []

    for ind_dir in sorted(dataset.iterdir()):
        if not ind_dir.is_dir() or ind_dir.name not in alturas:
            continue

        nombre = ind_dir.name
        categoria, peso_real, meses = parsear_nombre(nombre)
        altura_cm = alturas[nombre]

        # Buscar videos en 3d_modelo_*
        videos = []
        for sub in ind_dir.iterdir():
            if sub.is_dir() and sub.name.lower().startswith('3d_modelo'):
                videos = sorted([f for f in sub.iterdir() if f.suffix.lower() in ('.mov', '.mp4')])
                break

        if not videos:
            continue

        print(f"\n{'='*60}")
        print(f"  {nombre.upper()} | Peso: {peso_real} kg | Alto: {altura_cm} cm")
        print(f"{'='*60}")

        best_result = None
        for video in videos:
            print(f"\n  Video: {video.name}")

            # Extraer frames (cada 5 frames ≈ 6-12 fps)
            frames, masks, bboxes = extraer_frames(video, seg_model, frame_interval=5, max_frames=50)

            if len(frames) < 5:
                print(f"    Solo {len(frames)} frames, saltando")
                continue

            # SfM
            result = sfm_multiframe(frames, masks, bboxes, altura_cm)

            if result is not None:
                if best_result is None or result['n_points'] > best_result['n_points']:
                    best_result = result

        if best_result is None:
            print(f"  SfM falló para {nombre}")
            continue

        # Volumen por rebanadas (promedio de frames)
        vols = []
        vols_barril = []
        for i in range(len(best_result['frames'])):
            bbox = best_result['bboxes'][i]
            mask = best_result['masks'][i]
            bbox_h = bbox[3] - bbox[1]
            if bbox_h < 20:
                continue
            esc = altura_cm / bbox_h
            try:
                v, _ = volumen_por_rebanadas(mask, esc)
                vols.append(v)
            except:
                pass
            mt, _ = recortar_torso(mask, bbox)
            if mt is not None:
                try:
                    vb, _ = volumen_por_rebanadas(mt, esc)
                    vols_barril.append(vb)
                except:
                    pass

        vol_reb = None
        if vols:
            vol_reb = {
                'avg': np.mean(vols),
                'std': np.std(vols),
                'barril_avg': np.mean(vols_barril) if vols_barril else 0,
            }

        # ── Guardar PLY en carpeta del individuo (para visor 3D de la app) ──
        ind_output = project / 'output_modelos3d_grandes' / nombre
        ind_output.mkdir(exist_ok=True)
        pts_cm = best_result['points_cm']
        colors_3d = best_result['colors']

        # PLY con nube de puntos + malla (Delaunay sobre proyección 2D)
        ply_path = ind_output / f"{nombre}_video_3d.ply"
        try:
            # Triangular proyectando a 2D (vista lateral)
            ev = best_result['eigenvectors']
            centered = pts_cm - pts_cm.mean(axis=0)
            proj_2d = centered @ ev[:, 1:]  # alto y largo
            tri = Delaunay(proj_2d)
            # Filtrar triángulos con aristas muy largas
            valid_tris = []
            edges = []
            for t in tri.simplices:
                e = [np.linalg.norm(pts_cm[t[0]]-pts_cm[t[1]]),
                     np.linalg.norm(pts_cm[t[1]]-pts_cm[t[2]]),
                     np.linalg.norm(pts_cm[t[0]]-pts_cm[t[2]])]
                edges.extend(e)
            if edges:
                max_edge = np.percentile(edges, 90) * 2
                for t in tri.simplices:
                    e = [np.linalg.norm(pts_cm[t[0]]-pts_cm[t[1]]),
                         np.linalg.norm(pts_cm[t[1]]-pts_cm[t[2]]),
                         np.linalg.norm(pts_cm[t[0]]-pts_cm[t[2]])]
                    if max(e) < max_edge:
                        valid_tris.append(t)
            tris_3d = np.array(valid_tris) if valid_tris else np.array([]).reshape(0,3).astype(int)
        except:
            tris_3d = np.array([]).reshape(0,3).astype(int)

        nv = len(pts_cm)
        nf = len(tris_3d)
        with open(ply_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"comment Modelo 3D desde video SfM - {nombre}\n")
            f.write(f"comment Puntos: {nv} | Pares: {best_result['n_pairs']} | Peso real: {peso_real} kg\n")
            f.write(f"element vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write(f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n")
            for i, pt in enumerate(pts_cm):
                r_c, g_c, b_c = int(colors_3d[i][0]), int(colors_3d[i][1]), int(colors_3d[i][2])
                f.write(f"{pt[0]:.2f} {pt[1]:.2f} {pt[2]:.2f} {r_c} {g_c} {b_c}\n")
            for t in tris_3d:
                f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
        print(f"    PLY guardado: {ply_path} ({nv} verts, {nf} faces)")

        # Resumen JSON compatible con la app
        resumen_video = {
            'individuo': nombre,
            'peso_real_kg': peso_real,
            'altura_real_cm': altura_cm,
            'metodo': 'video_sfm',
            'sfm_puntos': best_result['n_points'],
            'sfm_pares': best_result['n_pairs'],
            'largo_cm': best_result['largo_cm'],
            'alto_cm': best_result['alto_cm'],
            'ancho_cm': best_result['ancho_cm'],
            'vol_total_litros': best_result['vol_litros'],
            'superficie_cm2': best_result['sup_cm2'],
            'n_frames': len(best_result['frames']),
        }
        with open(ind_output / f"{nombre}_video_resumen.json", 'w') as f:
            json.dump(resumen_video, f, indent=2, ensure_ascii=False)

        # Visualización
        vis_path = output / f"{nombre}_modelo3d_video.png"
        generar_visualizacion_sfm(nombre, peso_real, altura_cm, best_result, vol_reb, vis_path)

        r = {
            'individuo': nombre,
            'peso_real_kg': peso_real,
            'altura_cm': altura_cm,
            'sfm_puntos': best_result['n_points'],
            'sfm_pares': best_result['n_pairs'],
            'sfm_largo_cm': best_result['largo_cm'],
            'sfm_alto_cm': best_result['alto_cm'],
            'sfm_ancho_cm': best_result['ancho_cm'],
            'sfm_vol_litros': best_result['vol_litros'],
            'sfm_peso_kg': round(best_result['vol_litros'] * 1.03, 1),
            'reb_vol_avg': round(vol_reb['avg'], 1) if vol_reb else 0,
            'reb_barril_avg': round(vol_reb['barril_avg'], 1) if vol_reb else 0,
            'n_frames': len(best_result['frames']),
        }
        resultados.append(r)

        print(f"\n  RESULTADO {nombre.upper()}:")
        print(f"    SfM: {best_result['n_points']} pts | {best_result['largo_cm']:.0f}x{best_result['alto_cm']:.0f}x{best_result['ancho_cm']:.0f} cm")
        print(f"    Vol SfM: {best_result['vol_litros']:.0f}L → {best_result['vol_litros']*1.03:.0f}kg (real: {peso_real}kg)")

    # Tabla resumen
    print(f"\n\n{'#'*70}")
    print(f"  RESUMEN - MODELO 3D VIDEO ({len(resultados)} individuos)")
    print(f"{'#'*70}")
    print(f"\n  {'Vaca':<12} {'Real':>6} {'Pts3D':>6} {'Largo':>6} {'Alto':>6} {'Ancho':>6} {'Vol':>7} {'Peso':>7} {'Err':>6}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*6}")
    for r in resultados:
        err = (r['sfm_peso_kg'] - r['peso_real_kg']) / r['peso_real_kg'] * 100
        name = r['individuo'].replace('vaca_','').replace('_36','')
        print(f"  {name:<12} {r['peso_real_kg']:>6} {r['sfm_puntos']:>6} {r['sfm_largo_cm']:>6.0f} {r['sfm_alto_cm']:>6.0f} "
              f"{r['sfm_ancho_cm']:>6.0f} {r['sfm_vol_litros']:>7.0f} {r['sfm_peso_kg']:>7.0f} {err:>+5.0f}%")

    with open(output / 'resumen_video_3d.json', 'w') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\n  Guardado en: {output}/")


if __name__ == '__main__':
    main()
