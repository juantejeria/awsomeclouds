"""
Generador de modelos 3D para el dataset "grandes".
Estructura esperada:
  Dataset Modelo 3d "grandes"/
    vaca_370_36/
      3d_modelo_370/
        *.png
    vaca_382_36/
      3D_modelo_382/
        *.png

Nombre carpeta individuo: {categoria}_{peso}_{meses}
Carpeta de fotos: 3d_modelo_{peso} (case insensitive)

Pipeline por foto:
  1. YOLO detect → bbox
  2. GrabCut → silueta COMPLETA (con patas, cabeza, todo)
  3. Triangulación Delaunay → malla
  4. Escala a cm (altura REAL medida del individuo, desde alturas_individuos.json)
  5. Modelo simétrico 3D (profundidad elíptica)
  6. Volumen total + volumen solo-barril (recorte torso)
"""

import cv2
import numpy as np
import sys
import os
import re
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from scipy.spatial import Delaunay, ConvexHull
from pathlib import Path
from core.breed_coefficients import get_estimated_height
from core.reconstruccion_3d import sfm_real_desde_frames, guardar_ply_con_malla
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


# ═══════════════════════════════════════
# Funciones reutilizadas de generar_modelos3d_batch
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


def refinar_patas(mask, img, bbox):
    """Elimina sombra/pasto entre y alrededor de las patas.

    Enfoque geométrico: las patas son estructuras angostas verticales.
    Todo lo demás debajo de la línea ventral es sombra/suelo.

    1. Detecta la línea ventral (belly line) con el perfil de anchos
    2. Proyección vertical: suma de pixels por columna en la zona de patas
    3. Detecta picos en la proyección = posiciones de las patas
    4. Mantiene solo los picos (±ancho máximo de pata)
    5. Color: elimina pasto verde
    """
    x1, y1, x2, y2 = bbox
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    if bbox_h < 40 or bbox_w < 40:
        return mask

    h_img, w_img = mask.shape[:2]
    crop_w = min(x2, w_img) - x1

    # ── Perfil de anchos por fila ──
    row_widths = np.zeros(bbox_h, dtype=int)
    for i, y in enumerate(range(y1, min(y2, h_img))):
        row = mask[y, x1:min(x2, w_img)]
        cols = np.where(row > 0)[0]
        if len(cols) > 0:
            row_widths[i] = cols[-1] - cols[0] + 1

    if row_widths.max() == 0:
        return mask

    max_width = row_widths.max()

    # ── Línea ventral: última fila (desde abajo) donde ancho >= 50% del máximo ──
    # PERO: con sombra, la silueta puede ser ancha hasta abajo, empujando
    # la belly line al fondo. Forzar máximo en 65% del bbox para que siempre
    # quede una zona de patas razonable.
    body_threshold = max_width * 0.50
    max_belly_idx = int(len(row_widths) * 0.65)
    belly_idx = max_belly_idx  # default
    for i in range(len(row_widths) - 1, len(row_widths) // 3, -1):
        if row_widths[i] >= body_threshold:
            belly_idx = min(i, max_belly_idx)  # nunca más abajo del 65%
            break

    belly_y = y1 + belly_idx
    leg_zone_top = belly_y
    leg_zone_bottom = min(y2, h_img)
    leg_zone_h = leg_zone_bottom - leg_zone_top

    if leg_zone_h < 8:
        return mask

    # Ancho del cuerpo en zona torso
    body_top_idx = max(0, int(len(row_widths) * 0.25))
    body_bot_idx = min(belly_idx, int(len(row_widths) * 0.65))
    body_ws = row_widths[body_top_idx:body_bot_idx]
    body_width = float(np.median(body_ws[body_ws > 0])) if np.any(body_ws > 0) else float(bbox_w)

    # Verificar relleno de zona de patas
    leg_crop = mask[leg_zone_top:leg_zone_bottom, x1:min(x2, w_img)]
    if leg_crop.size == 0:
        return mask
    leg_fill = np.count_nonzero(leg_crop) / leg_crop.size

    result = mask.copy()

    # ── Estrategia 1: Color (eliminar pasto verde) ──
    body_top = y1 + body_top_idx
    body_bot = y1 + body_bot_idx
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    body_region = mask[body_top:min(body_bot, h_img), x1:min(x2, w_img)]
    body_hsv = hsv[body_top:min(body_bot, h_img), x1:min(x2, w_img)]
    body_pixels_hsv = body_hsv[body_region > 0]

    if len(body_pixels_hsv) >= 20:
        body_h_mean = np.mean(body_pixels_hsv[:, 0])
        body_h_std = np.std(body_pixels_hsv[:, 0])
        body_v_mean = np.mean(body_pixels_hsv[:, 2])
        body_v_std = np.std(body_pixels_hsv[:, 2])

        leg_hsv = hsv[leg_zone_top:leg_zone_bottom, x1:min(x2, w_img)]
        leg_region = mask[leg_zone_top:leg_zone_bottom, x1:min(x2, w_img)]

        h_ch = leg_hsv[:, :, 0].astype(np.float32)
        s_ch = leg_hsv[:, :, 1].astype(np.float32)
        v_ch = leg_hsv[:, :, 2].astype(np.float32)

        is_green = (h_ch >= 25) & (h_ch <= 90) & (s_ch > 20) & (v_ch > 25)
        h_diff = np.abs(h_ch - body_h_mean)
        h_diff = np.minimum(h_diff, 180 - h_diff)
        v_diff = np.abs(v_ch - body_v_mean)
        is_different = (h_diff > max(20, body_h_std * 2)) | (v_diff > max(35, body_v_std * 2))

        remove = (is_green | is_different) & (leg_region > 0)
        result[leg_zone_top:leg_zone_bottom, x1:min(x2, w_img)][remove] = 0

    # ── Estrategia 2: Apertura morfológica vertical SIEMPRE ──
    # Esto separa las patas (verticales) de la sombra (horizontal).
    # DEBE ir antes de la proyección vertical para que los picos sean detectables.
    # Se aplica siempre porque la sombra puede hacer que leg_fill parezca bajo
    # cuando en realidad la zona tiene mucha sombra unida al cuerpo.
    leg_crop_pre = result[leg_zone_top:leg_zone_bottom, x1:min(x2, w_img)].copy()
    leg_fill_pre = np.count_nonzero(leg_crop_pre) / leg_crop_pre.size if leg_crop_pre.size > 0 else 0

    if leg_fill_pre > 0.05:  # solo skip si la zona está prácticamente vacía
        # Apertura vertical: mantiene estructuras verticales, rompe puentes horizontales
        vert_h = max(10, int(leg_zone_h * 0.40))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_h))
        leg_opened = cv2.morphologyEx(leg_crop_pre, cv2.MORPH_OPEN, vert_kernel)

        # Recuperar ancho natural de pata con dilatación horizontal suave
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        leg_opened = cv2.dilate(leg_opened, horiz_kernel, iterations=2)

        remaining = np.count_nonzero(leg_opened)
        if remaining > np.count_nonzero(leg_crop_pre) * 0.05:
            result[leg_zone_top:leg_zone_bottom, x1:min(x2, w_img)] = leg_opened

    # ── Estrategia 3: Proyección vertical — encontrar patas como picos ──
    # Ahora que la apertura separó las patas, la proyección puede detectar picos.
    leg_crop_after = result[leg_zone_top:leg_zone_bottom, x1:min(x2, w_img)]
    col_sums = np.sum(leg_crop_after > 0, axis=0).astype(np.float32)

    if col_sums.max() == 0:
        return result

    # Suavizar la proyección para encontrar picos claros
    kernel_size = max(3, int(crop_w * 0.04)) | 1  # impar
    col_smooth = cv2.blur(col_sums.reshape(1, -1), (1, kernel_size)).ravel()

    # Ancho máximo de una pata: ~15% del ancho del cuerpo
    max_leg_w = max(int(body_width * 0.15), 12)
    min_peak_h = leg_zone_h * 0.15  # una pata debe cubrir al menos 15% de la zona

    # Encontrar picos (posiciones de patas)
    peaks = []
    for c in range(1, len(col_smooth) - 1):
        if col_smooth[c] >= min_peak_h:
            window_left = max(0, c - max_leg_w)
            window_right = min(len(col_smooth), c + max_leg_w)
            if col_smooth[c] >= col_smooth[window_left:window_right].max() * 0.90:
                if not peaks or (c - peaks[-1]) > max_leg_w:
                    peaks.append(c)

    if len(peaks) >= 2:
        # Crear máscara de columnas a mantener: ±max_leg_w alrededor de cada pico
        keep_cols = np.zeros(crop_w, dtype=bool)
        for peak in peaks:
            half = max_leg_w
            start = max(0, peak - half)
            end = min(crop_w, peak + half)
            keep_cols[start:end] = True

        # Remover pixels fuera de las columnas de patas
        leg_zone = result[leg_zone_top:leg_zone_bottom, x1:min(x2, w_img)]
        for local_x in range(crop_w):
            if not keep_cols[local_x]:
                leg_zone[:, local_x] = 0

    # ── Limpieza final ──
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    min_area = np.count_nonzero(mask) * 0.05
    clean = np.zeros_like(result)
    total_kept = 0
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean, [cnt], -1, 255, -1)
            total_kept += cv2.contourArea(cnt)

    if total_kept < np.count_nonzero(mask) * 0.30:
        return mask

    removed_total = np.count_nonzero(mask) - np.count_nonzero(clean)
    if removed_total > 0:
        print(f"    [Patas] Removidos {removed_total} px (belly_y={belly_y}, "
              f"leg_fill={leg_fill:.2f}, peaks={len(peaks)}, body_w={body_width:.0f})")

    return clean


def eliminar_sombra(mask, img, bbox):
    """Elimina sombra del animal de la máscara segmentada.

    La sombra se proyecta sobre el suelo debajo/alrededor del animal y GrabCut
    la incluye como foreground, especialmente en vacas negras donde el color es
    similar.  Usa señales complementarias:

    1. Luminosidad LAB: sombra más oscura que el cuerpo.
    2. Textura local: sombra uniforme vs pelaje con varianza.
    3. Cromaticidad LAB (a,b): sombra en pasto retiene tinte verde/marrón del
       suelo, distinto de la cromaticidad neutra del animal (clave para vacas
       negras donde L no discrimina bien).
    4. Geometría: extensión lateral y vertical más allá del cuerpo esperado.
    """
    x1, y1, x2, y2 = bbox
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    if bbox_h < 40 or bbox_w < 40:
        return mask

    h_img, w_img = mask.shape[:2]
    result = mask.copy()

    # ── Referencia: color y textura del cuerpo central (zona segura) ──
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a_ch = lab[:, :, 1].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_local = cv2.blur(gray, (9, 9))
    var_local = cv2.blur(gray ** 2, (9, 9)) - mean_local ** 2
    var_local = np.maximum(var_local, 0)

    # Zona "core" del cuerpo: 20-50% vertical, 25-75% horizontal
    core_y1 = int(y1 + bbox_h * 0.20)
    core_y2 = int(y1 + bbox_h * 0.50)
    core_x1 = int(x1 + bbox_w * 0.25)
    core_x2 = int(x1 + bbox_w * 0.75)

    core_mask = mask[core_y1:core_y2, core_x1:core_x2]
    core_L = L[core_y1:core_y2, core_x1:core_x2]
    core_L_vals = core_L[core_mask > 0]
    core_var = var_local[core_y1:core_y2, core_x1:core_x2]
    core_var_vals = core_var[core_mask > 0]

    # Cromaticidad del cuerpo
    core_a = a_ch[core_y1:core_y2, core_x1:core_x2]
    core_b = b_ch[core_y1:core_y2, core_x1:core_x2]
    body_a_vals = core_a[core_mask > 0]
    body_b_vals = core_b[core_mask > 0]

    if len(core_L_vals) < 20:
        return mask

    body_L_median = np.median(core_L_vals)
    body_L_p10 = np.percentile(core_L_vals, 10)
    body_var_median = np.median(core_var_vals) if len(core_var_vals) > 0 else 100
    body_a_mean = np.mean(body_a_vals)
    body_a_std = np.std(body_a_vals)
    body_b_mean = np.mean(body_b_vals)
    body_b_std = np.std(body_b_vals)

    # ── Muestrear color del suelo ──
    # Estrategia doble: pixels NO enmascarados en la zona inferior (mejor señal)
    # + justo debajo del bbox como fallback.
    has_ground_ref = False
    ground_a_mean = ground_b_mean = 128.0

    # Opción 1: pixels de fondo (no-máscara) en la zona inferior del bbox
    bg_zone_top = int(y1 + bbox_h * 0.60)
    bg_zone_bot = min(y2, h_img)
    bg_zone_left = max(0, x1)
    bg_zone_right = min(x2, w_img)
    bg_mask_zone = mask[bg_zone_top:bg_zone_bot, bg_zone_left:bg_zone_right]
    bg_a_zone = a_ch[bg_zone_top:bg_zone_bot, bg_zone_left:bg_zone_right]
    bg_b_zone = b_ch[bg_zone_top:bg_zone_bot, bg_zone_left:bg_zone_right]
    bg_pixels = bg_mask_zone == 0  # pixels de fondo
    bg_a_vals = bg_a_zone[bg_pixels]
    bg_b_vals = bg_b_zone[bg_pixels]

    if len(bg_a_vals) > 50:
        ground_a_mean = np.mean(bg_a_vals)
        ground_b_mean = np.mean(bg_b_vals)
        has_ground_ref = True
    else:
        # Fallback: justo debajo del bbox
        ground_y1 = min(y2, h_img - 2)
        ground_y2 = min(y2 + max(25, int(bbox_h * 0.08)), h_img)
        if ground_y2 > ground_y1 + 3:
            ground_a = a_ch[ground_y1:ground_y2, x1:x2]
            ground_b = b_ch[ground_y1:ground_y2, x1:x2]
            if ground_a.size > 20:
                ground_a_mean = np.mean(ground_a)
                ground_b_mean = np.mean(ground_b)
                has_ground_ref = True

    # ── Umbral de sombra adaptativo ──
    if body_L_median < 80:
        L_thresh = max(body_L_p10 - 12, body_L_median * 0.55)
    else:
        L_thresh = body_L_median * 0.45

    var_thresh = max(body_var_median * 0.25, 8.0)

    # ── Zona inferior: debajo del 55% del bbox ──
    shadow_top = int(y1 + bbox_h * 0.55)
    zone_mask = np.zeros((h_img, w_img), dtype=bool)
    zone_mask[shadow_top:min(y2 + 20, h_img), max(0, x1):min(x2, w_img)] = True
    candidates = (result > 0) & zone_mask

    # Señal 1: oscuro + liso (funciona bien para vacas claras)
    is_dark = L < L_thresh
    is_smooth = var_local < var_thresh
    shadow_by_lum = candidates & is_dark & is_smooth
    result[shadow_by_lum] = 0

    # ── Señal 2: Cromaticidad — pixel más parecido al suelo que al cuerpo ──
    # Clave para vacas negras donde L no discrimina: la sombra en pasto retiene
    # el tinte verde/marrón del suelo, distinto de la cromaticidad neutra del animal.
    # Se exige que el pixel además sea oscuro (no más brillante que el cuerpo)
    # para no remover variaciones legítimas del pelaje (panza, ubre, etc).
    if has_ground_ref:
        dist_body = np.sqrt((a_ch - body_a_mean)**2 + (b_ch - body_b_mean)**2)
        dist_ground = np.sqrt((a_ch - ground_a_mean)**2 + (b_ch - ground_b_mean)**2)

        chroma_diff = np.sqrt((body_a_mean - ground_a_mean)**2 +
                              (body_b_mean - ground_b_mean)**2)

        if chroma_diff > 12:  # Diferencia cromática suficiente cuerpo vs suelo
            # Pixel claramente más parecido al suelo que al cuerpo
            is_ground_like = dist_ground < dist_body * 0.5

            # Combinación conservadora: tipo suelo Y no más brillante que cuerpo
            # Y (oscuro O liso) — necesita al menos una señal extra
            is_not_brighter = L <= body_L_median + 10
            is_dark_or_smooth = (L < body_L_median) | (var_local < var_thresh * 2)
            shadow_by_chroma = candidates & is_ground_like & is_not_brighter & is_dark_or_smooth
            result[shadow_by_chroma] = 0

            print(f"    [Sombra-Chroma] chroma_diff={chroma_diff:.1f}, "
                  f"body_ab=({body_a_mean:.0f},{body_b_mean:.0f}), "
                  f"ground_ab=({ground_a_mean:.0f},{ground_b_mean:.0f})")

    # ── Señal 3: Recorte lateral en TODA la altura ──
    # La sombra puede extenderse a la misma altura del cuerpo (no solo debajo).
    # Si una fila es mucho más ancha que el torso, los bordes son sombra/suelo.
    # Se opera en TODA la máscara, usando cromaticidad como señal principal.
    body_top = int(y1 + bbox_h * 0.25)
    body_bot = int(y1 + bbox_h * 0.55)
    body_widths = []
    body_centers = []
    for y in range(body_top, min(body_bot, h_img)):
        row = result[y, x1:x2]
        cols = np.where(row > 0)[0]
        if len(cols) > 0:
            body_widths.append(cols[-1] - cols[0] + 1)
            body_centers.append((cols[0] + cols[-1]) / 2 + x1)
    if body_widths:
        expected_width = np.median(body_widths)
        expected_center = np.median(body_centers)

        # Recorrer TODAS las filas del mask (no solo zona inferior)
        for y in range(y1, min(y2 + 10, h_img)):
            row = result[y, :]
            cols = np.where(row > 0)[0]
            if len(cols) == 0:
                continue
            row_left = cols[0]
            row_right = cols[-1]
            row_width = row_right - row_left + 1

            if row_width <= expected_width * 1.15:
                continue  # fila tiene ancho razonable

            # Fila más ancha de lo esperado: recortar bordes que sean suelo
            # Más agresivo cuanto más ancha sea la fila respecto al cuerpo
            trim_left = int(expected_center - expected_width * 0.55)
            trim_right = int(expected_center + expected_width * 0.55)

            # Borde izquierdo: recortar pixels que sean suelo/sombra
            for xx in range(row_left, min(trim_left, w_img)):
                if result[y, xx] > 0:
                    # Usar cromaticidad si disponible, sino L/var
                    if has_ground_ref:
                        d_body = np.sqrt((a_ch[y, xx] - body_a_mean)**2 +
                                         (b_ch[y, xx] - body_b_mean)**2)
                        d_ground = np.sqrt((a_ch[y, xx] - ground_a_mean)**2 +
                                           (b_ch[y, xx] - ground_b_mean)**2)
                        if d_ground < d_body * 0.8:
                            result[y, xx] = 0
                            continue
                    pL = L[y, xx]
                    pVar = var_local[y, xx]
                    if pL < L_thresh * 1.5 or pVar < var_thresh * 2:
                        result[y, xx] = 0

            # Borde derecho: idem
            for xx in range(max(trim_right, 0), min(row_right + 1, w_img)):
                if result[y, xx] > 0:
                    if has_ground_ref:
                        d_body = np.sqrt((a_ch[y, xx] - body_a_mean)**2 +
                                         (b_ch[y, xx] - body_b_mean)**2)
                        d_ground = np.sqrt((a_ch[y, xx] - ground_a_mean)**2 +
                                           (b_ch[y, xx] - ground_b_mean)**2)
                        if d_ground < d_body * 0.8:
                            result[y, xx] = 0
                            continue
                    pL = L[y, xx]
                    pVar = var_local[y, xx]
                    if pL < L_thresh * 1.5 or pVar < var_thresh * 2:
                        result[y, xx] = 0

    # ── Limpiar fragmentos pequeños resultantes ──
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    min_area = np.count_nonzero(mask) * 0.05
    clean = np.zeros_like(result)
    total_kept = 0
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean, [cnt], -1, 255, -1)
            total_kept += cv2.contourArea(cnt)

    # Si eliminamos demasiado, la detección fue muy agresiva → devolver original
    if total_kept < np.count_nonzero(mask) * 0.40:
        return mask

    removed = np.count_nonzero(mask) - np.count_nonzero(clean)
    if removed > 0:
        print(f"    [Sombra] Eliminados {removed} px sombra "
              f"(L_thresh={L_thresh:.0f}, var_thresh={var_thresh:.0f}, "
              f"body_L={body_L_median:.0f})")

    return clean


_diag_counter = [0]  # mutable counter for diagnostic filenames
_diag_output_dir = [None]


def guardar_diagnostico(img, bbox, mask_grabcut, mask_sombra, mask_patas, nombre=""):
    """Guarda imagen diagnóstico de 4 paneles mostrando cada etapa de segmentación."""
    if _diag_output_dir[0] is None:
        return
    out_dir = Path(_diag_output_dir[0])
    out_dir.mkdir(exist_ok=True)

    _diag_counter[0] += 1
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = bbox

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: Original + bbox
    axes[0].imshow(img_rgb)
    axes[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                     fill=False, edgecolor='lime', lw=2))
    axes[0].set_title('1. Original + YOLO bbox')
    axes[0].axis('off')

    # Panel 2: Máscara GrabCut
    overlay_gc = img_rgb.copy()
    overlay_gc[mask_grabcut > 0] = [255, 0, 0]
    blended_gc = cv2.addWeighted(img_rgb, 0.5, overlay_gc, 0.5, 0)
    axes[1].imshow(blended_gc)
    axes[1].set_title(f'2. GrabCut ({np.count_nonzero(mask_grabcut)} px)')
    axes[1].axis('off')

    # Panel 3: Después de eliminar_sombra
    overlay_sh = img_rgb.copy()
    overlay_sh[mask_sombra > 0] = [0, 255, 0]
    # Marcar en rojo lo que se eliminó
    removed_shadow = (mask_grabcut > 0) & (mask_sombra == 0)
    overlay_sh[removed_shadow] = [255, 0, 0]
    blended_sh = cv2.addWeighted(img_rgb, 0.5, overlay_sh, 0.5, 0)
    axes[2].imshow(blended_sh)
    removed_sh_count = np.count_nonzero(removed_shadow)
    axes[2].set_title(f'3. Post-sombra (rojo={removed_sh_count} removidos)')
    axes[2].axis('off')

    # Panel 4: Después de refinar_patas (final)
    overlay_final = img_rgb.copy()
    overlay_final[mask_patas > 0] = [0, 200, 255]
    # Marcar en rojo lo eliminado en refinar_patas
    removed_patas = (mask_sombra > 0) & (mask_patas == 0)
    overlay_final[removed_patas] = [255, 100, 0]
    # Marcar en rojo oscuro lo eliminado en sombra
    overlay_final[removed_shadow] = [180, 0, 0]
    blended_final = cv2.addWeighted(img_rgb, 0.4, overlay_final, 0.6, 0)
    axes[3].imshow(blended_final)
    removed_patas_count = np.count_nonzero(removed_patas)
    axes[3].set_title(f'4. Final (cyan=animal, naranja={removed_patas_count} patas)')
    axes[3].axis('off')

    fig.suptitle(f'Diagnóstico segmentación: {nombre}', fontsize=12)
    plt.tight_layout()
    fname = out_dir / f"diag_{_diag_counter[0]:03d}_{nombre.replace(' ', '_')}.png"
    plt.savefig(str(fname), dpi=120, bbox_inches='tight')
    plt.close()


def segmentar(img, bbox, nombre_foto=""):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = 10
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(w, x2+pad), min(h, y2+pad)
    bbox_h = y2 - y1
    bbox_w = x2 - x1

    # ── Pase 1: GrabCut con rectangulo ──
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, (x1, y1, x2-x1, y2-y1), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    # ── Pase 2: Re-run con semillas (FG seguro = centro cuerpo, BG seguro = suelo) ──
    fg_y1 = int(y1 + bbox_h * 0.25)
    fg_y2 = int(y1 + bbox_h * 0.55)
    fg_x1 = int(x1 + bbox_w * 0.25)
    fg_x2 = int(x1 + bbox_w * 0.75)
    mask[fg_y1:fg_y2, fg_x1:fg_x2] = cv2.GC_FGD

    if y2 + 5 < h:
        mask[y2:min(y2 + 30, h), x1:x2] = cv2.GC_BGD
    border = max(3, int(bbox_w * 0.05))
    mask[int(y1 + bbox_h * 0.7):y2, x1:x1+border] = cv2.GC_BGD
    mask[int(y1 + bbox_h * 0.7):y2, x2-border:x2] = cv2.GC_BGD

    leg_y_top = int(y1 + bbox_h * 0.75)
    leg_x_center1 = int(x1 + bbox_w * 0.30)
    leg_x_center2 = int(x1 + bbox_w * 0.70)
    center_leg = mask[leg_y_top:y2, leg_x_center1:leg_x_center2]
    center_leg[(center_leg != cv2.GC_FGD)] = cv2.GC_PR_BGD

    bgd2, fgd2 = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgd2, fgd2, 5, cv2.GC_INIT_WITH_MASK)

    mask_fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    m = np.zeros_like(mask_fg)
    cv2.drawContours(m, [c], -1, 255, -1)

    mask_grabcut = m.copy()  # guardar para diagnóstico

    # ── Evaluar calidad del GrabCut ──
    # Si la zona de patas ya tiene fill bajo (patas separadas, sin sombra),
    # el GrabCut ya es bueno → no tocar para no empeorar.
    # Solo aplicar limpieza agresiva cuando el GrabCut falló (fill alto = sombra).
    leg_eval_top = int(y1 + bbox_h * 0.65)
    leg_eval_bot = min(y2, h)
    leg_eval_left = x1
    leg_eval_right = min(x2, w)
    leg_eval_zone = m[leg_eval_top:leg_eval_bot, leg_eval_left:leg_eval_right]
    leg_eval_fill = np.count_nonzero(leg_eval_zone) / leg_eval_zone.size if leg_eval_zone.size > 0 else 0

    # Fill < 25% = patas ya separadas, GrabCut bueno → skip post-procesamiento
    needs_cleanup = leg_eval_fill >= 0.25
    if needs_cleanup:
        print(f"    [GrabCut] leg_fill={leg_eval_fill:.0%} → aplicando limpieza sombra+patas")
        m = eliminar_sombra(m, img, np.array([x1, y1, x2, y2]))
        mask_post_sombra = m.copy()
        m = refinar_patas(m, img, np.array([x1, y1, x2, y2]))
    else:
        print(f"    [GrabCut] leg_fill={leg_eval_fill:.0%} → contorno OK, sin post-proceso")
        mask_post_sombra = m.copy()

    # Diagnóstico visual
    guardar_diagnostico(img, np.array([x1, y1, x2, y2]),
                        mask_grabcut, mask_post_sombra, m, nombre_foto)

    # Recalcular contorno
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)

    return m, c


def recortar_torso(mask, bbox):
    """Aísla el torso/barril eliminando cabeza/cuello y patas.
    Solo para cálculo de peso del barril, NO para el modelo 3D visual.
    """
    x1, y1, x2, y2 = bbox
    bbox_h = y2 - y1
    bbox_w = x2 - x1
    if bbox_h < 20 or bbox_w < 20:
        return mask, None

    # Análisis de ancho por fila (para cortar patas)
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
    threshold = max_width * 0.50

    is_body = row_widths > threshold
    body_start = body_end = cur_start = None
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

    torso_y_top = y1 + body_start
    torso_y_bottom = y1 + body_end

    # Análisis de alto por columna (para cortar cabeza/cuello)
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
    col_threshold = max_height * 0.35

    is_body_col = col_heights > col_threshold
    col_start = col_end = cur_start = None
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

    mask_torso = np.zeros_like(mask)
    mask_torso[torso_y_top:torso_y_bottom, torso_x_left:torso_x_right] = \
        mask[torso_y_top:torso_y_bottom, torso_x_left:torso_x_right]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_torso = cv2.morphologyEx(mask_torso, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_torso, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    c = max(contours, key=cv2.contourArea)
    area_torso = cv2.contourArea(c)
    area_original = np.count_nonzero(mask)
    if area_torso < area_original * 0.20:
        return mask, None

    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [c], -1, 255, -1)
    return final_mask, c


# ═══════════════════════════════════════
# Muestreo + Triangulación + PLY
# ═══════════════════════════════════════

def samplear(contorno, mask, n_borde=100, n_interior=60):
    c = contorno.reshape(-1, 2)
    pts_b = c[::max(1, len(c) // n_borde)]

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return pts_b, np.array([]).reshape(0, 2)

    cols = int(np.sqrt(n_interior) * 1.5) + 2
    rows = int(np.sqrt(n_interior)) + 2
    gx = np.linspace(xs.min(), xs.max(), cols + 2)[1:-1]
    gy = np.linspace(ys.min(), ys.max(), rows + 2)[1:-1]
    mx, my = np.meshgrid(gx, gy)
    grid = np.column_stack([mx.ravel(), my.ravel()]).astype(int)
    interior = [pt for pt in grid
                if 0 <= pt[1] < mask.shape[0] and 0 <= pt[0] < mask.shape[1]
                and mask[pt[1], pt[0]] > 0]
    return pts_b, np.array(interior) if interior else np.array([]).reshape(0, 2)


def triangular(pts_b, pts_i, mask):
    todos = np.vstack([pts_b, pts_i]) if len(pts_i) > 0 else pts_b
    todos = np.unique(todos, axis=0)
    if len(todos) < 3:
        return None, None
    tri = Delaunay(todos)
    validos = []
    for s in tri.simplices:
        cx, cy = todos[s].mean(axis=0).astype(int)
        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx] > 0:
            validos.append(s)
    return todos, np.array(validos) if validos else np.array([]).reshape(0, 3)


def profundidad_eliptica(puntos_cm):
    """Calcula profundidad elíptica para cada punto (modelo simétrico)."""
    ys = puntos_cm[:, 1]
    y_min, y_max = ys.min(), ys.max()
    y_range = y_max - y_min if y_max > y_min else 1
    y_center = y_min + y_range * 0.4

    depths = []
    for pt in puntos_cm:
        d = min(abs(pt[1] - y_center) / (y_range * 0.5), 1.0)
        depths.append(y_range * 0.25 * np.sqrt(max(0, 1 - d ** 2)))
    return np.array(depths)


def volumen_por_rebanadas(mask, escala):
    """Calcula volumen integrando rebanadas verticales (columnas X) sobre la máscara.

    Para cada columna X de la máscara (= corte transversal del cuerpo):
      1. Mide la altura en píxeles (rango Y donde mask > 0)
      2. Convierte a cm con la escala
      3. Asume sección transversal elíptica:
         semi-eje vertical = h/2, semi-eje profundidad = h * K_DEPTH
      4. Área de la elipse = π * (h/2) * (h * K_DEPTH)
      5. Suma: vol = Σ área(x) * dx
    """
    K_DEPTH = 0.25  # ratio profundidad/altura del corte

    h_img, w_img = mask.shape[:2]
    dx_cm = escala

    vol_cm3 = 0.0
    sup_cm2 = 0.0

    for x in range(w_img):
        col = mask[:, x]
        rows = np.where(col > 0)[0]
        if len(rows) == 0:
            continue
        h_px = rows[-1] - rows[0] + 1
        h_cm = h_px * escala

        a = h_cm / 2.0        # semi-eje vertical
        b = h_cm * K_DEPTH     # semi-eje profundidad
        area = np.pi * a * b
        vol_cm3 += area * dx_cm

        perimetro = np.pi * (3*(a+b) - np.sqrt((3*a+b)*(a+3*b)))
        sup_cm2 += perimetro * dx_cm

    vol_litros = vol_cm3 / 1000.0
    return vol_litros, sup_cm2


def guardar_ply(path, pts_cm, tris, colores, simetrico=False, escala_info=""):
    if simetrico:
        n = len(pts_cm)
        depths = profundidad_eliptica(pts_cm)
        # Forzar depth=0 en los vertices del RIM (lazo de borde 2D). Asi z+
        # y z- COINCIDEN en posicion en el rim, sin hueco visual al renderizar
        # solid. NO se sueldan indices: cada mitad sigue siendo una malla
        # independiente — las "lineas" estan solo dentro de cada lado, nunca
        # cruzan al espejo.
        from collections import Counter as _Counter
        _ec = _Counter()
        for a, b, c in tris:
            for e in [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))]:
                _ec[e] += 1
        rim_set = set()
        for (a, b), k in _ec.items():
            if k == 1:
                rim_set.add(a)
                rim_set.add(b)
        is_rim = np.array([i in rim_set for i in range(n)])
        depths_r = depths.copy()
        depths_r[is_rim] = 0.0
        pts_r = np.column_stack([pts_cm[:, 0], pts_cm[:, 1], depths_r])
        pts_l = np.column_stack([pts_cm[:, 0], pts_cm[:, 1], -depths_r])
        all_pts = np.vstack([pts_r, pts_l])
        all_colors = np.vstack([colores, colores]) if colores is not None else None
        tris_r = tris.copy()
        tris_l = tris.copy() + n
        tris_l = tris_l[:, [0, 2, 1]]
        all_tris = np.vstack([tris_r, tris_l])
    else:
        all_pts = np.column_stack([pts_cm[:, 0], pts_cm[:, 1], np.zeros(len(pts_cm))])
        all_colors = colores
        all_tris = tris

    # Recorte de cresta del LOMO: baja el filo superior (ancho z~0) hasta el lomo
    # real, igual que el post-proceso recortar_cresta_ply.py. Solución integral
    # para que los modelos NUEVOS no nazcan con cresta. En el caso plano (z=0 en
    # todo) no hace nada. No toca la panza.
    try:
        from core.crest_trim_mesh import trim_top_crest
        all_pts = trim_top_crest(all_pts)
    except Exception as _e:
        print(f"[warn] recorte de cresta omitido: {_e}")

    nv, nf = len(all_pts), len(all_tris)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("comment Unidades: centimetros\n")
        if escala_info:
            f.write(f"comment {escala_info}\n")
        f.write(f"element vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\nproperty list uchar int vertex_indices\nend_header\n")
        for i, pt in enumerate(all_pts):
            r, g, b = (int(all_colors[i][0]), int(all_colors[i][1]), int(all_colors[i][2])) \
                if all_colors is not None and i < len(all_colors) else (139, 90, 43)
            f.write(f"{pt[0]:.2f} {pt[1]:.2f} {pt[2]:.2f} {r} {g} {b}\n")
        for t in all_tris:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    return all_pts, all_tris


def volumen_malla_cerrada(pts, tris):
    """Volumen encerrado (en litros) de una malla triangular cerrada.

    Usa el teorema de la divergencia: el volumen con signo de cada
    tetraedro (origen, v0, v1, v2) es (v0 · (v1 × v2)) / 6, y la suma
    sobre todas las caras da el volumen encerrado. Se toma el valor
    absoluto para ser robusto al sentido de las normales.

    Pensado para la malla `_3d.ply` (silueta espejada en Z con
    profundidad elíptica), que es cerrada por construcción. Puntos en
    cm → cm³; se reporta en litros (cm³ / 1000).
    """
    pts = np.asarray(pts, dtype=float)
    tris = np.asarray(tris, dtype=int)
    if len(tris) == 0 or len(pts) < 4:
        return 0.0
    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    vol_cm3 = np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum() / 6.0
    return round(abs(vol_cm3) / 1000.0, 1)


def volumen_ply_cerrado(path):
    """Lee un PLY ASCII (formato de guardar_ply) y devuelve el volumen
    encerrado en litros vía volumen_malla_cerrada().

    Sirve para recalcular el volumen del `_3d.ply` ya escrito en disco
    sin reconstruir la malla en memoria.
    """
    from pathlib import Path as _Path
    lines = _Path(path).read_text().splitlines()
    n_vertex = n_face = 0
    header_end = 0
    for i, ln in enumerate(lines):
        if ln.startswith('element vertex'):
            n_vertex = int(ln.split()[-1])
        elif ln.startswith('element face'):
            n_face = int(ln.split()[-1])
        elif ln.strip() == 'end_header':
            header_end = i + 1
            break

    pts = []
    for ln in lines[header_end:header_end + n_vertex]:
        p = ln.split()
        if len(p) >= 3:
            pts.append((float(p[0]), float(p[1]), float(p[2])))

    tris = []
    for ln in lines[header_end + n_vertex:header_end + n_vertex + n_face]:
        p = ln.split()
        if not p:
            continue
        k = int(p[0])
        idx = [int(v) for v in p[1:1 + k]]
        # Triangular en abanico cualquier cara con >3 vértices
        for j in range(1, k - 1):
            tris.append((idx[0], idx[j], idx[j + 1]))

    if not pts or not tris:
        return 0.0
    return volumen_malla_cerrada(np.array(pts, dtype=float), np.array(tris, dtype=int))


# ═══════════════════════════════════════
# Procesar una foto (silueta completa + barril separado)
# ═══════════════════════════════════════

def segmentar_yolo_seg(img, bbox, seg_model):
    """Segmenta la vaca usando YOLO-seg (yolov8s-seg, clase cow=19).
    Retorna (mask, contorno) o (None, None) si falla.
    """
    h, w = img.shape[:2]
    results = seg_model(img, conf=0.15, classes=[19], verbose=False)
    if not results or len(results[0].boxes) == 0 or results[0].masks is None:
        return None, None

    masks = results[0].masks.data.cpu().numpy()
    # Seleccionar la máscara con mayor overlap con el bbox de cow.pt
    x1, y1, x2, y2 = bbox
    best_iou = -1
    best_mask = None
    for m in masks:
        m_resized = cv2.resize(m, (w, h))
        m_bin = (m_resized > 0.5).astype(np.uint8)
        # IoU con bbox
        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 1
        inter = np.sum(m_bin & bbox_mask)
        union = np.sum(m_bin | bbox_mask)
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_mask = m_bin

    if best_mask is None or best_iou < 0.1:
        return None, None

    mask_fg = (best_mask * 255).astype(np.uint8)

    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    m = np.zeros_like(mask_fg)
    cv2.drawContours(m, [c], -1, 255, -1)

    return m, c


def segmentar_silueta_custom(img, bbox, silueta_model):
    """Segmenta la silueta completa usando silueta_seg.pt (clase 0 = silueta).
    Retorna (mask, contorno) o (None, None) si falla.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    results = silueta_model(img, conf=0.25, verbose=False)
    if not results or len(results[0].boxes) == 0 or results[0].masks is None:
        return None, None

    masks = results[0].masks.data.cpu().numpy()

    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_mask[y1:y2, x1:x2] = 1

    best_mask = None
    best_iou = -1
    for m in masks:
        m_resized = cv2.resize(m, (w, h))
        m_bin = (m_resized > 0.5).astype(np.uint8)
        inter = np.sum(m_bin & bbox_mask)
        union = np.sum(m_bin | bbox_mask)
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_mask = m_bin

    if best_mask is None or best_iou < 0.1:
        return None, None

    mask_fg = (best_mask * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    m = np.zeros_like(mask_fg)
    cv2.drawContours(m, [c], -1, 255, -1)

    return m, c


def segmentar_barril(img, bbox, barril_model):
    """Segmenta el BARRIL de la vaca usando barril_seg.pt (clase 0 = barril).
    Corre sobre la imagen COMPLETA (sin cropear al bbox) para no cortar el barril.
    Usa el bbox solo para seleccionar la detección más cercana al animal.
    Retorna (mask, contorno) o (None, None) si falla.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    results = barril_model(img, conf=0.25, verbose=False)
    if not results or len(results[0].boxes) == 0 or results[0].masks is None:
        return None, None

    masks = results[0].masks.data.cpu().numpy()

    # Seleccionar la máscara con mayor overlap con el bbox de detección
    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_mask[y1:y2, x1:x2] = 1

    best_mask = None
    best_iou = -1
    for m in masks:
        m_resized = cv2.resize(m, (w, h))
        m_bin = (m_resized > 0.5).astype(np.uint8)
        inter = np.sum(m_bin & bbox_mask)
        union = np.sum(m_bin | bbox_mask)
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_mask = m_bin

    if best_mask is None or best_iou < 0.05:
        return None, None

    mask_fg = (best_mask * 255).astype(np.uint8)

    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    m = np.zeros_like(mask_fg)
    cv2.drawContours(m, [c], -1, 255, -1)

    return m, c


def postprocesar_barril(mask_pred, mask_full, bbox, head_x=None):
    """Post-procesa la predicción de barril_seg para ajustarla al patrón de anotación:
    - INCLUIR el pecho/cuello
    - CORTAR las patas (detectar donde el ancho cae debajo de la panza)
    - RECORTAR la cola (usar head_x de cow.pt keypoints para saber el lado)

    head_x: coordenada X de la cabeza (keypoint 0 de cow.pt). Si se proporciona,
    la cola se recorta del lado OPUESTO a la cabeza.
    """
    x1, y1, x2, y2 = bbox
    h, w = mask_full.shape[:2]

    ys_full, xs_full = np.where(mask_full > 0)
    if len(ys_full) == 0:
        return mask_pred, None

    full_top = int(ys_full.min())
    full_bottom = int(ys_full.max())
    full_left = int(xs_full.min())
    full_right = int(xs_full.max())
    full_h = full_bottom - full_top
    full_w = full_right - full_left

    if full_h < 20 or full_w < 20:
        return mask_pred, None

    # ── 1. CORTAR PATAS: perfil de anchos por fila ──
    row_widths = np.zeros(full_h + 1, dtype=int)
    for i, y in enumerate(range(full_top, full_bottom + 1)):
        if y >= h:
            break
        row = mask_full[y, full_left:full_right + 1]
        cols = np.where(row > 0)[0]
        row_widths[i] = (cols[-1] - cols[0] + 1) if len(cols) > 0 else 0

    max_w = row_widths.max()
    if max_w == 0:
        return mask_pred, None

    # Encontrar la panza: la zona del máximo ancho (parte media-baja del cuerpo)
    # Suavizar para evitar ruido
    kernel_size = max(3, len(row_widths) // 20)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(row_widths, np.ones(kernel_size) / kernel_size, mode='same')

    panza_idx = int(np.argmax(smoothed))
    panza_width = smoothed[panza_idx]

    # Desde la panza hacia abajo, buscar donde el ancho cae al 85% del ancho de panza
    # Esto detecta la transición cuerpo → patas incluso cuando las patas son anchas
    drop_threshold = panza_width * 0.85
    body_end_idx = panza_idx
    for i in range(panza_idx, len(smoothed)):
        if smoothed[i] >= drop_threshold:
            body_end_idx = i
        else:
            break

    body_bottom_y = full_top + body_end_idx

    # ── 2. CORTAR COLA (pero mantener pecho/cuello) ──
    # Análisis de alto por columna para detectar extensiones finas
    col_heights = np.zeros(full_w + 1, dtype=int)
    for i, x in enumerate(range(full_left, full_right + 1)):
        if x >= w:
            break
        col = mask_full[full_top:body_bottom_y + 1, x]
        rows = np.where(col > 0)[0]
        col_heights[i] = (rows[-1] - rows[0] + 1) if len(rows) > 0 else 0

    max_col_h = col_heights.max()
    if max_col_h == 0:
        return mask_pred, None

    col_threshold = max_col_h * 0.45

    body_x_left = full_left
    body_x_right = full_right

    # Recortar del lado de la CABEZA (quitar cabeza/cuello, mantener pecho).
    # La cola se mantiene (no se recorta).
    # Usamos keypoint 0 de cow.pt para saber de qué lado está la cabeza.
    if head_x is not None:
        full_center_x = (full_left + full_right) / 2
        head_is_right = head_x > full_center_x

        if head_is_right:
            # Cabeza a la derecha → recortar derecha
            for i in range(len(col_heights) - 1, -1, -1):
                if col_heights[i] >= col_threshold:
                    body_x_right = full_left + i
                    break
        else:
            # Cabeza a la izquierda → recortar izquierda
            for i in range(len(col_heights)):
                if col_heights[i] >= col_threshold:
                    body_x_left = full_left + i
                    break

    # ── 3. Construir máscara resultado ──
    result = np.zeros_like(mask_full)
    result[full_top:body_bottom_y + 1, body_x_left:body_x_right] = \
        mask_full[full_top:body_bottom_y + 1, body_x_left:body_x_right]

    # ── 4. Limpieza ──
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_morph, iterations=2)

    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_pred, None

    c = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(result)
    cv2.drawContours(clean, [c], -1, 255, -1)

    return clean, c


def procesar_foto(img, cow_model, coco_model, cow_height_cm, nombre="foto", seg_model=None, barril_model=None, silueta_model=None):
    """Procesa una foto. Modelo 3D construido desde el BARRIL (si barril_model disponible)."""

    bbox = detectar_vaca(img, cow_model, coco_model)
    if bbox is None:
        print(f"    {nombre}: no se detectó vaca")
        return None

    # Prioridad: silueta_seg.pt → yolov8s-seg → GrabCut
    mask_full, contorno_full = None, None
    seg_method = "grabcut"
    if silueta_model is not None:
        mask_full, contorno_full = segmentar_silueta_custom(img, bbox, silueta_model)
        if mask_full is not None:
            seg_method = "silueta-seg"

    if mask_full is None and seg_model is not None:
        mask_full, contorno_full = segmentar_yolo_seg(img, bbox, seg_model)
        if mask_full is not None:
            seg_method = "yolo-seg"

    if mask_full is None:
        mask_full, contorno_full = segmentar(img, bbox, nombre_foto=nombre)

    if mask_full is None:
        print(f"    {nombre}: segmentación falló")
        return None

    area_full_px = cv2.contourArea(contorno_full)
    if area_full_px < 500:
        print(f"    {nombre}: área muy pequeña ({area_full_px})")
        return None

    x1, y1, x2, y2 = bbox
    bbox_h_px = y2 - y1
    if bbox_h_px < 20:
        return None

    escala = cow_height_cm / bbox_h_px

    # ── Segmentar barril: solo el modelo entrenado, sin post-proceso ──
    mask_barril, contorno_barril = None, None
    barril_method = "heuristica"
    if barril_model is not None:
        mask_barril, contorno_barril = segmentar_barril(img, bbox, barril_model)
        if contorno_barril is not None:
            barril_method = "barril-seg"

    # ── Clampear barril a la silueta: el barril no puede salir de la silueta completa ──
    if mask_barril is not None and contorno_barril is not None:
        mask_barril = cv2.bitwise_and(mask_barril, mask_full)
        contours_clamp, _ = cv2.findContours(mask_barril, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_clamp:
            contorno_barril = max(contours_clamp, key=cv2.contourArea)
        else:
            # El clamp dejó el barril vacío → no hay barril válido
            mask_barril, contorno_barril = None, None
            barril_method = "sin-barril"
    else:
        barril_method = "sin-barril"

    print(f"    [{seg_method}+{barril_method}] ", end="")

    # ── Modelo 3D construido desde la SILUETA COMPLETA ──
    pts_b, pts_i = samplear(contorno_full, mask_full)
    puntos_px, tris = triangular(pts_b, pts_i, mask_full)
    if puntos_px is None or len(tris) == 0:
        print(f"    {nombre}: triangulación falló")
        return None

    puntos_cm = puntos_px.astype(float) * escala
    puntos_cm[:, 1] = puntos_cm[:, 1].max() - puntos_cm[:, 1]

    colores = np.array([
        img[min(pt[1], img.shape[0]-1), min(pt[0], img.shape[1]-1)][::-1]
        for pt in puntos_px
    ])

    # Malla del barril para visualización (separada de la silueta)
    puntos_barril_px = puntos_barril_cm = tris_barril = colores_barril = None
    if mask_barril is not None and contorno_barril is not None:
        pts_bb, pts_bi = samplear(contorno_barril, mask_barril)
        puntos_barril_px, tris_barril = triangular(pts_bb, pts_bi, mask_barril)
        if puntos_barril_px is not None and len(tris_barril) > 0:
            puntos_barril_cm = puntos_barril_px.astype(float) * escala
            puntos_barril_cm[:, 1] = puntos_barril_cm[:, 1].max() - puntos_barril_cm[:, 1]
            colores_barril = np.array([
                img[min(pt[1], img.shape[0]-1), min(pt[0], img.shape[1]-1)][::-1]
                for pt in puntos_barril_px
            ])

    # Volumen del barril (solo si hay barril válido)
    vol_barril_litros = None
    sup_barril_cm2 = None
    if mask_barril is not None:
        try:
            vol_barril_litros, sup_barril_cm2 = volumen_por_rebanadas(mask_barril, escala)
            vol_barril_litros = round(vol_barril_litros, 1)
            sup_barril_cm2 = round(sup_barril_cm2, 1)
        except Exception:
            vol_barril_litros = sup_barril_cm2 = None

    # Volumen silueta completa
    try:
        vol_total_litros, sup_cm2 = volumen_por_rebanadas(mask_full, escala)
    except Exception:
        vol_total_litros = sup_cm2 = 0

    x_min, x_max = puntos_cm[:, 0].min(), puntos_cm[:, 0].max()
    y_min, y_max = puntos_cm[:, 1].min(), puntos_cm[:, 1].max()

    return {
        'nombre': nombre,
        'img': img,
        'bbox': bbox,
        'mask_full': mask_full,
        'mask_barril': mask_barril,
        'contorno_full': contorno_full,
        'contorno_barril': contorno_barril,
        'puntos_px': puntos_px,
        'puntos_cm': puntos_cm,
        'tris': tris,
        'colores': colores,
        'puntos_barril_px': puntos_barril_px,
        'puntos_barril_cm': puntos_barril_cm,
        'tris_barril': tris_barril,
        'colores_barril': colores_barril,
        'escala': escala,
        'largo_cm': round(x_max - x_min, 1),
        'alto_cm': round(y_max - y_min, 1),
        'vol_total_litros': round(vol_total_litros, 1),
        'vol_barril_litros': vol_barril_litros,
        'superficie_barril_cm2': sup_barril_cm2,
        'superficie_cm2': round(sup_cm2, 1),
        'num_triangulos': len(tris),
        'barril_method': barril_method,
    }


# ═══════════════════════════════════════
# Filtro IQR
# ═══════════════════════════════════════

def filtrar_outliers(resultados, campo='vol_total_litros'):
    valores = [r[campo] for r in resultados if r[campo] > 0]
    if len(valores) < 3:
        return resultados, []
    q1 = np.percentile(valores, 25)
    q3 = np.percentile(valores, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    validos = [r for r in resultados if lower <= r[campo] <= upper and r[campo] > 0]
    descartados = [r for r in resultados if r not in validos]
    return validos, descartados


# ═══════════════════════════════════════
# Visualización
# ═══════════════════════════════════════

def generar_visualizacion(nombre_vaca, peso_real, mejor, metricas, output_path):
    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(f'MODELO - {nombre_vaca.upper()} ({peso_real} kg) \u00b7 Altura real: {metricas["altura_real_cm"]:.0f} cm \u00b7 Escala: {metricas["escala"]:.4f} cm/px',
                 fontsize=14, fontweight='bold')

    img_rgb = cv2.cvtColor(mejor['img'], cv2.COLOR_BGR2RGB)
    puntos_cm = mejor['puntos_cm']
    puntos_px = mejor['puntos_px']
    tris = mejor['tris']
    colores = mejor['colores']

    # 1. Original + bbox
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(img_rgb)
    x1, y1, x2, y2 = mejor['bbox']
    ax1.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', lw=2))
    ax1.set_title('YOLO Detection')
    ax1.axis('off')

    # 2. Segmentación completa + contorno barril
    ax2 = fig.add_subplot(2, 4, 2)
    overlay = img_rgb.copy()
    overlay[mejor['mask_full'] > 0] = [0, 200, 0]
    blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    if mejor['mask_barril'] is not None:
        contours_t, _ = cv2.findContours(mejor['mask_barril'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_t:
            pts = cnt.reshape(-1, 2)
            ax2.plot(pts[:, 0], pts[:, 1], 'c-', linewidth=1.5, alpha=0.8)
    ax2.imshow(blended)
    ax2.set_title('Silueta completa (verde) + barril (cyan)')
    ax2.axis('off')

    # 3. Barril: overlay sobre foto (solo barril resaltado)
    ax3 = fig.add_subplot(2, 4, 3)
    if mejor['mask_barril'] is not None:
        # Fondo oscurecido, barril en color original
        dark = (img_rgb * 0.3).astype(np.uint8)
        barril_vis = dark.copy()
        barril_vis[mejor['mask_barril'] > 0] = img_rgb[mejor['mask_barril'] > 0]
        # Borde cyan del barril
        contours_b, _ = cv2.findContours(mejor['mask_barril'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        barril_vis_bgr = cv2.cvtColor(barril_vis, cv2.COLOR_RGB2BGR)
        cv2.drawContours(barril_vis_bgr, contours_b, -1, (255, 255, 0), 2)
        barril_vis = cv2.cvtColor(barril_vis_bgr, cv2.COLOR_BGR2RGB)
        ax3.imshow(barril_vis)
    else:
        ax3.imshow(img_rgb, alpha=0.3)
    vbl = metricas["vol_barril_litros"]
    ax3.set_title(f'Barril (vol={vbl:.0f}L)' if vbl is not None else 'Barril (N/A)')
    ax3.axis('off')

    # 4. Malla en px
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(img_rgb, alpha=0.3)
    if len(tris) > 0:
        ax4.triplot(puntos_px[:, 0], puntos_px[:, 1], tris, 'b-', lw=0.4)
    ax4.plot(puntos_px[:, 0], puntos_px[:, 1], 'r.', ms=1.5)
    ax4.set_title(f'Malla ({len(tris)} triangulos)')
    ax4.axis('off')

    # 5. Modelo con textura cm
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.set_facecolor('black')
    if len(tris) > 0:
        polys = [puntos_cm[t] for t in tris]
        fcolors = [(colores[t] / 255.0).mean(axis=0) for t in tris]
        ax5.add_collection(PolyCollection(polys, facecolors=fcolors, edgecolors='none', alpha=0.9))
        x_min, x_max = puntos_cm[:, 0].min(), puntos_cm[:, 0].max()
        y_min, y_max = puntos_cm[:, 1].min(), puntos_cm[:, 1].max()
        ax5.set_xlim(x_min - 3, x_max + 3)
        ax5.set_ylim(y_min - 3, y_max + 3)
        ax5.annotate('', xy=(x_max, y_min-4), xytext=(x_min, y_min-4),
                     arrowprops=dict(arrowstyle='<->', color='yellow', lw=2))
        ax5.text((x_min+x_max)/2, y_min-7, f'{metricas["largo_cm"]:.0f} cm', color='yellow',
                 ha='center', fontsize=11, fontweight='bold')
        ax5.annotate('', xy=(x_max+4, y_max), xytext=(x_max+4, y_min),
                     arrowprops=dict(arrowstyle='<->', color='cyan', lw=2))
        ax5.text(x_max+6, (y_min+y_max)/2, f'Altura real: {metricas["altura_real_cm"]:.0f} cm', color='cyan',
                 ha='left', fontsize=11, fontweight='bold', rotation=90)
    ax5.set_title('Modelo Escalado (cm)')
    ax5.set_aspect('equal')
    ax5.axis('off')

    # 6. Modelo barril (malla propia del barril, no la silueta completa)
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.set_facecolor('black')
    pb_cm = mejor.get('puntos_barril_cm')
    tb = mejor.get('tris_barril')
    cb = mejor.get('colores_barril')
    if pb_cm is not None and tb is not None and len(tb) > 0:
        polys_b = [pb_cm[t] for t in tb]
        fcolors_b = [(cb[t] / 255.0).mean(axis=0) for t in tb]
        ax6.add_collection(PolyCollection(polys_b, facecolors=fcolors_b, edgecolors='none', alpha=0.9))
        ax6.set_xlim(pb_cm[:, 0].min()-3, pb_cm[:, 0].max()+3)
        ax6.set_ylim(pb_cm[:, 1].min()-3, pb_cm[:, 1].max()+3)
    ax6.set_title(f'Barril 3D ({vbl:.0f}L → {vbl*1.03:.0f}kg)' if vbl is not None else 'Barril 3D (N/A)')
    ax6.set_aspect('equal')
    ax6.axis('off')

    # 7. Wireframe
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.set_facecolor('white')
    if len(tris) > 0:
        ax7.triplot(puntos_cm[:, 0], puntos_cm[:, 1], tris, color='sienna', linewidth=0.3)
        ax7.set_xlim(puntos_cm[:, 0].min()-3, puntos_cm[:, 0].max()+3)
        ax7.set_ylim(puntos_cm[:, 1].min()-3, puntos_cm[:, 1].max()+3)
    ax7.set_title('Wireframe (cm)')
    ax7.set_aspect('equal')
    ax7.axis('off')

    # 8. Info
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    info = f"""{nombre_vaca.upper()} - BARRIL

Peso real:         {peso_real} kg
Altura real:       {metricas['altura_real_cm']:.0f} cm
Escala:            {metricas['escala']:.4f} cm/px

BARRIL:
  Largo:           {metricas['largo_cm']:.1f} cm
  Volumen:         {f"{metricas['vol_barril_litros']:.1f} litros" if metricas['vol_barril_litros'] is not None else "N/A"}
  Peso barril:     {f"{metricas['vol_barril_litros']*1.03:.1f} kg" if metricas['vol_barril_litros'] is not None else "N/A"}

Triangulos:        {metricas['num_triangulos']}
Fotos procesadas:  {metricas['fotos_validas']}/{metricas['fotos_procesadas']}"""

    ax8.text(0.05, 0.95, info, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════
# Parsear nombre de carpeta
# ═══════════════════════════════════════

def parsear_nombre(nombre):
    """Parsea 'vaca_370_36' → (categoria='vaca', peso=370, meses=36)"""
    match = re.match(r'(\w+?)_(\d+)_(\d+)', nombre)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return 'desconocido', 0, 0


def meses_a_rango(meses):
    """Convierte meses a rango de edad para breed_coefficients."""
    if meses <= 6:
        return "0-6"
    elif meses <= 12:
        return "6-12"
    elif meses <= 18:
        return "12-18"
    elif meses <= 24:
        return "18-24"
    elif meses <= 36:
        return "24-36"
    else:
        return "36+"


# ═══════════════════════════════════════
# Fusionar puntos de múltiples fotos
# ═══════════════════════════════════════

CANVAS_W = 500  # resolución del canvas normalizado
CANVAS_H = 350


def normalizar_mask_a_canvas(mask, bbox):
    """Normaliza una máscara al canvas canónico usando el bbox YOLO.
    Todas las fotos quedan alineadas al mismo espacio.
    """
    x1, y1, x2, y2 = bbox
    # Recortar la máscara al bbox
    crop = mask[y1:y2, x1:x2]
    if crop.shape[0] < 5 or crop.shape[1] < 5:
        return None
    # Resize al canvas canónico
    resized = cv2.resize(crop, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_NEAREST)
    return resized


def fusionar_fotos(resultados, cow_height_cm):
    """Fusiona las máscaras de BARRIL de todas las fotos en un solo modelo.

    1. Normaliza cada máscara de barril al canvas canónico (alineadas por bbox)
    2. Acumula las máscaras → mapa de densidad
    3. Umbraliza: zonas con >=50% de aparición se mantienen
    4. Samplea + triangula sobre la máscara fusionada
    5. Colores de la foto con mejor barril
    """
    n = len(resultados)
    if n == 0:
        return None

    # ── Paso 1: Acumular máscaras de BARRIL normalizadas ──
    acumulador = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    masks_norm = []
    for r in resultados:
        # Usar mask_barril solo si existe y no es None
        if r.get('mask_barril') is not None:
            m = normalizar_mask_a_canvas(r['mask_barril'], r['bbox'])
        else:
            continue  # sin barril válido, no aporta a la fusión
        if m is not None:
            acumulador += (m > 0).astype(np.float32)
            masks_norm.append(m)

    if len(masks_norm) == 0:
        return None

    # ── Paso 2: Máscara fusionada (aparece en >=50% de las fotos) ──
    # Mínimo 1 foto si solo hay 1-2
    umbral = max(1, len(masks_norm) * 0.5)
    mask_fusion = (acumulador >= umbral).astype(np.uint8) * 255

    # Limpiar morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fusion = cv2.morphologyEx(mask_fusion, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fusion = cv2.morphologyEx(mask_fusion, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_fusion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contorno_fusion = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contorno_fusion) < 200:
        return None

    # Limpiar máscara al contorno más grande
    mask_clean = np.zeros_like(mask_fusion)
    cv2.drawContours(mask_clean, [contorno_fusion], -1, 255, -1)

    print(f"  [Fusión] {len(masks_norm)} máscaras fusionadas, umbral={umbral:.1f}")
    print(f"  [Fusión] Área fusionada: {cv2.contourArea(contorno_fusion)} px (canvas {CANVAS_W}x{CANVAS_H})")

    # ── Paso 3: Samplear puntos del canvas fusionado ──
    pts_b, pts_i = samplear(contorno_fusion, mask_clean, n_borde=120, n_interior=80)
    puntos_canvas, tris = triangular(pts_b, pts_i, mask_clean)
    if puntos_canvas is None or len(tris) == 0:
        return None

    print(f"  [Fusión] Puntos: {len(puntos_canvas)} | Triángulos: {len(tris)}")

    # ── Paso 4: Escalar canvas a cm ──
    # El canvas representa el bbox completo de la vaca
    # Alto del canvas = cow_height_cm
    escala_cm_per_canvas_px = cow_height_cm / CANVAS_H
    puntos_cm = puntos_canvas.astype(float) * escala_cm_per_canvas_px
    puntos_cm[:, 1] = puntos_cm[:, 1].max() - puntos_cm[:, 1]  # flip Y

    # ── Paso 5: Colores de la foto con mayor área ──
    # Elegir la foto de referencia: la que tenga mayor área de barril (o silueta si no hay barril)
    def _area_ref(r):
        if r.get('contorno_barril') is not None:
            return cv2.contourArea(r['contorno_barril'])
        return cv2.contourArea(r['contorno_full'])
    mejor_ref = max(resultados, key=_area_ref)
    img_ref = mejor_ref['img']
    bbox_ref = mejor_ref['bbox']
    rx1, ry1, rx2, ry2 = bbox_ref
    ref_w, ref_h = rx2 - rx1, ry2 - ry1

    colores = []
    for pt in puntos_canvas:
        # Mapear canvas → coordenadas de la imagen de referencia
        img_x = int(rx1 + pt[0] * ref_w / CANVAS_W)
        img_y = int(ry1 + pt[1] * ref_h / CANVAS_H)
        img_x = max(0, min(img_x, img_ref.shape[1] - 1))
        img_y = max(0, min(img_y, img_ref.shape[0] - 1))
        b, g, r = img_ref[img_y, img_x]
        colores.append([int(r), int(g), int(b)])
    colores = np.array(colores)

    # ── Paso 6: Volumen total ──
    depths = profundidad_eliptica(puntos_cm)
    pts_r = np.column_stack([puntos_cm[:, 0], puntos_cm[:, 1], depths])
    pts_l = np.column_stack([puntos_cm[:, 0], puntos_cm[:, 1], -depths])
    pts_3d = np.vstack([pts_r, pts_l])

    try:
        hull = ConvexHull(pts_3d)
        vol_total_litros = hull.volume / 1000.0
        sup_cm2 = hull.area
    except Exception:
        vol_total_litros = sup_cm2 = 0

    # ── Paso 7: Vol barril = vol total (la fusión YA es solo barril) ──
    vol_barril_litros = vol_total_litros
    mask_torso_canvas = mask_clean  # la fusión es el barril

    x_min, x_max = puntos_cm[:, 0].min(), puntos_cm[:, 0].max()
    y_min, y_max = puntos_cm[:, 1].min(), puntos_cm[:, 1].max()

    return {
        'puntos_cm': puntos_cm,
        'puntos_canvas': puntos_canvas,
        'tris': tris,
        'colores': colores,
        'mask_fusion': mask_clean,
        'mask_torso_canvas': mask_torso_canvas,
        'acumulador': acumulador,
        'escala': escala_cm_per_canvas_px,
        'largo_cm': round(x_max - x_min, 1),
        'alto_cm': round(y_max - y_min, 1),
        'vol_total_litros': round(vol_total_litros, 1),
        'vol_barril_litros': round(vol_barril_litros, 1),
        'superficie_cm2': round(sup_cm2, 1),
        'num_triangulos': len(tris),
        'img_ref': img_ref,
        'bbox_ref': bbox_ref,
        'mejor_ref': mejor_ref,
    }


# ═══════════════════════════════════════
# Visualización (actualizada para fusión)
# ═══════════════════════════════════════

def generar_visualizacion_fusion(nombre_vaca, peso_real, fusion, resultados, metricas, output_path):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'Modelo 3D Fusionado - {nombre_vaca} ({peso_real} kg) [{len(resultados)} fotos]',
                 fontsize=14, fontweight='bold')

    puntos_cm = fusion['puntos_cm']
    puntos_canvas = fusion['puntos_canvas']
    tris = fusion['tris']
    colores = fusion['colores']

    # 1. Mapa de densidad (acumulador)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(fusion['acumulador'], cmap='hot', interpolation='nearest')
    ax1.set_title(f'Densidad ({len(resultados)} fotos superpuestas)')
    ax1.axis('off')

    # 2. Máscara fusionada + contorno torso
    ax2 = fig.add_subplot(2, 3, 2)
    vis_mask = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    vis_mask[fusion['mask_fusion'] > 0] = [0, 200, 0]
    if fusion['mask_torso_canvas'] is not None:
        vis_mask[fusion['mask_torso_canvas'] > 0] = [0, 200, 200]
    ax2.imshow(vis_mask)
    ax2.set_title('Barril fusionado (verde)')
    ax2.axis('off')

    # 3. Malla sobre canvas
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(fusion['mask_fusion'], cmap='gray', alpha=0.3)
    if len(tris) > 0:
        ax3.triplot(puntos_canvas[:, 0], puntos_canvas[:, 1], tris, 'b-', lw=0.4)
    ax3.plot(puntos_canvas[:, 0], puntos_canvas[:, 1], 'r.', ms=1.5)
    ax3.set_title(f'Malla fusionada ({len(tris)} triangulos)')
    ax3.axis('off')

    # 4. Modelo con textura cm
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor('black')
    if len(tris) > 0:
        polys = [puntos_cm[t] for t in tris]
        fcolors = [(colores[t] / 255.0).mean(axis=0) for t in tris]
        ax4.add_collection(PolyCollection(polys, facecolors=fcolors, edgecolors='none', alpha=0.9))
        x_min, x_max = puntos_cm[:, 0].min(), puntos_cm[:, 0].max()
        y_min, y_max = puntos_cm[:, 1].min(), puntos_cm[:, 1].max()
        ax4.set_xlim(x_min - 3, x_max + 3)
        ax4.set_ylim(y_min - 3, y_max + 3)
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
    if len(tris) > 0:
        ax5.triplot(puntos_cm[:, 0], puntos_cm[:, 1], tris, color='sienna', linewidth=0.3)
        ax5.set_xlim(puntos_cm[:, 0].min()-3, puntos_cm[:, 0].max()+3)
        ax5.set_ylim(puntos_cm[:, 1].min()-3, puntos_cm[:, 1].max()+3)
    ax5.set_title('Wireframe (cm)')
    ax5.set_aspect('equal')
    ax5.axis('off')

    # 6. Info
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    info = f"""{nombre_vaca.upper()} - MODELO FUSIONADO

Peso real:         {peso_real} kg
Escala:            {metricas['escala']:.4f} cm/px-canvas
Metodo:            Fusion de {metricas['fotos_validas']} fotos

MEDIDAS (silueta fusionada):
  Largo:           {metricas['largo_cm']:.1f} cm
  Alto:            {metricas['alto_cm']:.1f} cm
  Vol total:       {metricas['vol_total_litros']:.1f} litros
  Superficie:      {metricas['superficie_cm2']:.0f} cm2

BARRIL (solo torso):
  Vol barril:      {f"{metricas['vol_barril_litros']:.1f} litros" if metricas['vol_barril_litros'] is not None else "N/A"}

FOTOS:
  Procesadas:      {metricas['fotos_procesadas']}
  Descartadas:     {metricas['fotos_descartadas']}
  Usadas fusion:   {metricas['fotos_validas']}

Triangulos:        {metricas['num_triangulos']}"""

    ax6.text(0.05, 0.95, info, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════
# Procesar un individuo (fusión multi-foto)
# ═══════════════════════════════════════

def procesar_individuo(individuo_dir, fotos_dir, output_dir, cow_model, coco_model, cow_height_override=None, seg_model=None, barril_model=None, silueta_model=None):
    nombre = individuo_dir.name
    categoria, peso_real, meses = parsear_nombre(nombre)
    edad_rango = meses_a_rango(meses)
    if cow_height_override is not None:
        cow_height_cm = cow_height_override
    else:
        cow_height_cm = get_estimated_height(categoria, edad_rango)

    fotos = sorted([f for f in fotos_dir.iterdir()
                    if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if not fotos:
        return None

    print(f"\n{'='*60}")
    print(f"  {nombre.upper()}")
    print(f"  Categoria: {categoria} | Peso real: {peso_real} kg | Edad: {meses} meses ({edad_rango})")
    print(f"  Altura real: {cow_height_cm} cm | Fotos: {len(fotos)}")
    print(f"{'='*60}")

    # ── Fase 1: Procesar cada foto (filtrar frames donde el animal no está completo) ──
    resultados = []
    descartados_incompletos = 0
    for i, foto in enumerate(fotos):
        print(f"  [{i+1}/{len(fotos)}] {foto.name} ... ", end="", flush=True)
        img = cv2.imread(str(foto))
        if img is None:
            print("ERROR lectura")
            continue

        # Verificar que el animal está completo (bbox no toca bordes de imagen)
        bbox = detectar_vaca(img, cow_model, coco_model)
        if bbox is None:
            print("no vaca detectada")
            continue
        img_h, img_w = img.shape[:2]
        bx1, by1, bx2, by2 = bbox
        margin_px = 5  # tolerancia de píxeles al borde
        if bx1 <= margin_px or by1 <= margin_px or bx2 >= img_w - margin_px or by2 >= img_h - margin_px:
            print(f"INCOMPLETO (bbox toca borde)")
            descartados_incompletos += 1
            continue

        r = procesar_foto(img, cow_model, coco_model, cow_height_cm, foto.name, seg_model=seg_model, barril_model=barril_model, silueta_model=silueta_model)
        if r is None:
            continue
        vb = r['vol_barril_litros']
        vb_str = f"{vb:.1f}L ({vb*1.03:.1f}kg)" if vb is not None else "N/A"
        print(f"Vol: {r['vol_total_litros']:.1f}L ({r['vol_total_litros']*1.03:.1f}kg) | Barril: {vb_str}")
        resultados.append(r)

    if descartados_incompletos > 0:
        print(f"\n  Frames incompletos descartados: {descartados_incompletos}")

    if len(resultados) == 0:
        print(f"\n  SKIP: Ninguna foto válida")
        return None

    # Filtrar outliers
    validos, descartados = filtrar_outliers(resultados)
    if not validos:
        validos = resultados

    if descartados:
        print(f"\n  OUTLIERS descartados ({len(descartados)}):")
        for d in descartados:
            print(f"    - {d['nombre']}: {d['vol_total_litros']:.1f}L")

    # ── Fase 2: Usar PROMEDIO de todos los frames válidos ──
    vols_barril_validos = [r['vol_barril_litros'] for r in validos if r['vol_barril_litros'] is not None]
    vol_barril_promedio = statistics.mean(vols_barril_validos) if vols_barril_validos else None
    peso_barril_promedio = vol_barril_promedio * 1.03 if vol_barril_promedio is not None else None
    vol_total_promedio = statistics.mean([r['vol_total_litros'] for r in validos])

    # Mejor foto para visualización: preferir las que tienen barril detectado;
    # entre esas, la más cercana al centro de la imagen.
    def _dist_al_centro(r):
        bx1, by1, bx2, by2 = r['bbox']
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        ih, iw = r['img'].shape[:2]
        return ((cx - iw/2.0) ** 2 + (cy - ih/2.0) ** 2) ** 0.5
    con_barril = [r for r in validos if r.get('contorno_barril') is not None]
    candidatos = con_barril if con_barril else validos
    mejor = min(candidatos, key=_dist_al_centro)

    barril_str = f"barril={vol_barril_promedio:.1f}L → {peso_barril_promedio:.1f}kg" if vol_barril_promedio is not None else "barril=N/A"
    print(f"\n  PROMEDIO ({len(validos)} frames): {barril_str} (real={peso_real}kg, target 50%={peso_real*0.5:.0f}kg)")
    print(f"  Foto referencia (más cercana al promedio): {mejor['nombre']}")

    # ── Fase 3: Guardar modelo de la mejor foto ──
    ind_output = output_dir / nombre
    ind_output.mkdir(exist_ok=True)

    # Usar malla del barril para PLY si existe, sino silueta completa
    if mejor.get('puntos_barril_cm') is not None and mejor.get('tris_barril') is not None and len(mejor['tris_barril']) > 0:
        puntos_cm = mejor['puntos_barril_cm']
        tris = mejor['tris_barril']
        colores_m = mejor['colores_barril']
    else:
        puntos_cm = mejor['puntos_cm']
        tris = mejor['tris']
        colores_m = mejor['colores']
    escala_info = f"Escala: {mejor['escala']:.4f} cm/px | Peso real: {peso_real} kg | Alto: {cow_height_cm} cm"

    # PLY lateral (barril)
    ply_lat = ind_output / f"{nombre}_lateral.ply"
    guardar_ply(str(ply_lat), puntos_cm, tris, colores_m, simetrico=False, escala_info=escala_info)

    # PLY 3D simétrico (barril)
    ply_3d = ind_output / f"{nombre}_3d.ply"
    pts_3d, tris_3d = guardar_ply(str(ply_3d), puntos_cm, tris, colores_m, simetrico=True, escala_info=escala_info)

    # Volumen final del PLY
    try:
        hull = ConvexHull(pts_3d)
        vol_final = hull.volume / 1000.0
        sup_final = hull.area
    except Exception:
        vol_final = mejor['vol_total_litros']
        sup_final = mejor['superficie_cm2']

    vols_indiv = [r['vol_total_litros'] for r in validos]
    vol_mean = statistics.mean(vols_indiv)
    vol_std = statistics.stdev(vols_indiv) if len(vols_indiv) > 1 else 0

    contorno_ref = mejor['contorno_barril'] if mejor.get('contorno_barril') is not None else mejor['contorno_full']
    area_px = cv2.contourArea(contorno_ref)
    area_cm2 = area_px * mejor['escala'] ** 2

    metricas = {
        'escala': mejor['escala'],
        'largo_cm': mejor['largo_cm'],
        'altura_real_cm': cow_height_cm,
        'area_cm2': round(area_cm2, 1),
        'vol_total_litros': round(vol_total_promedio, 1),
        'volumen_cm3': round(vol_total_promedio * 1000, 0),
        'vol_barril_litros': round(vol_barril_promedio, 1) if vol_barril_promedio is not None else None,
        'superficie_cm2': mejor['superficie_cm2'],
        'num_triangulos': mejor['num_triangulos'],
        'fotos_procesadas': len(resultados),
        'fotos_descartadas': len(descartados),
        'fotos_incompletas': descartados_incompletos,
        'fotos_validas': len(validos),
        'foto_usada': mejor['nombre'],
    }

    # Visualización estilo MODELO_VACA2.png
    vis_path = ind_output / f"{nombre}_modelo.png"
    generar_visualizacion(nombre, peso_real, mejor, metricas, vis_path)

    # JSON
    resumen = {
        'individuo': nombre,
        'categoria': categoria,
        'peso_real_kg': peso_real,
        'meses': meses,
        'edad_rango': edad_rango,
        'altura_real_cm': cow_height_cm,
        'metodo': 'promedio_frames',
        'escala_cm_px': round(mejor['escala'], 6),
        'vol_total_litros': round(vol_total_promedio, 1),
        'peso_vol_kg': round(vol_total_promedio * 1.03, 1),
        'vol_total_std': round(vol_std, 1),
        'vol_barril_litros': round(vol_barril_promedio, 1) if vol_barril_promedio is not None else None,
        'peso_barril_kg': round(vol_barril_promedio * 1.03, 1) if vol_barril_promedio is not None else None,
        'vol_barril_std': round(statistics.stdev(vols_barril_validos) if len(vols_barril_validos) > 1 else 0, 1) if vols_barril_validos else None,
        'fotos_procesadas': len(resultados),
        'fotos_validas': len(validos),
        'fotos_descartadas': len(descartados),
        'fotos_con_barril': len(vols_barril_validos),
        'resultados_por_foto': [
            {
                'foto': r['nombre'],
                'vol_total_litros': r['vol_total_litros'],
                'peso_vol_kg': round(r['vol_total_litros'] * 1.03, 1),
                'vol_barril_litros': r['vol_barril_litros'],
                'peso_barril_kg': round(r['vol_barril_litros'] * 1.03, 1) if r['vol_barril_litros'] is not None else None,
                'descartada': r in descartados,
            }
            for r in resultados
        ],
    }
    with open(ind_output / f"{nombre}_resumen.json", 'w') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    peso_vol = metricas['vol_total_litros'] * 1.03
    print(f"\n  RESULTADO {nombre.upper()} (PROMEDIO {len(validos)} frames completos):")
    print(f"    Vol total:  {metricas['vol_total_litros']:.1f} L → peso: {peso_vol:.1f} kg (real: {peso_real} kg)")
    if metricas['vol_barril_litros'] is not None:
        peso_barril = metricas['vol_barril_litros'] * 1.03
        error_50 = (peso_barril - peso_real * 0.5) / peso_real * 100
        print(f"    Vol barril: {metricas['vol_barril_litros']:.1f} L → peso: {peso_barril:.1f} kg (target 50%: {peso_real*0.5:.0f} kg, error: {error_50:+.1f}%)")
    else:
        print(f"    Vol barril: N/A (no se detectó barril en ninguna foto)")
    print(f"    Archivos:   {ind_output}/")

    return resumen


# ═══════════════════════════════════════
# Procesar individuo con SfM REAL
# ═══════════════════════════════════════

def procesar_individuo_sfm_real(individuo_dir, fotos_dir, output_dir, cow_model, coco_model, cow_height_override=None):
    """Procesa un individuo usando fotogrametria real (SfM + espejo 180°)."""
    nombre = individuo_dir.name
    categoria, peso_real, meses = parsear_nombre(nombre)
    edad_rango = meses_a_rango(meses)
    if cow_height_override is not None:
        cow_height_cm = cow_height_override
    else:
        cow_height_cm = get_estimated_height(categoria, edad_rango)

    fotos = sorted([f for f in fotos_dir.iterdir()
                    if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if not fotos:
        return None

    print(f"\n{'='*60}")
    print(f"  {nombre.upper()} [SfM REAL]")
    print(f"  Categoria: {categoria} | Peso real: {peso_real} kg | Edad: {meses} meses ({edad_rango})")
    print(f"  Altura estimada: {cow_height_cm} cm | Fotos: {len(fotos)}")
    print(f"{'='*60}")

    # ── Fase 1: Cargar fotos, detectar vaca, segmentar ──
    frames = []
    masks_torso = []
    masks_full = []
    bboxes = []

    for i, foto in enumerate(fotos):
        print(f"  [{i+1}/{len(fotos)}] {foto.name} ... ", end="", flush=True)
        img = cv2.imread(str(foto))
        if img is None:
            print("ERROR lectura")
            continue

        bbox = detectar_vaca(img, cow_model, coco_model)
        if bbox is None:
            print("no vaca detectada")
            continue

        mask_f, contorno_f = segmentar(img, bbox, nombre_foto=f"{nombre}_{foto.stem}")
        if mask_f is None:
            print("segmentacion fallo")
            continue

        mask_t, _ = recortar_torso(mask_f, bbox)

        frames.append(img)
        masks_full.append(mask_f)
        masks_torso.append(mask_t)
        bboxes.append(bbox)
        print(f"OK (bbox={bbox.tolist()})")

    if len(frames) < 2:
        print(f"\n  SKIP: Solo {len(frames)} frames validos (minimo 2)")
        # Fallback a metodo clasico
        if len(frames) == 1:
            print(f"  Fallback: usando procesar_individuo() clasico")
            return procesar_individuo(individuo_dir, fotos_dir, output_dir, cow_model, coco_model)
        return None

    # ── Fase 2: SfM real ──
    print(f"\n  Ejecutando SfM real con {len(frames)} frames...")
    sfm_result = sfm_real_desde_frames(
        frames, masks_torso, cow_height_cm,
        bboxes=bboxes, masks_full=masks_full
    )

    if sfm_result is None:
        print(f"  ERROR: SfM real fallo. Fallback a procesar_individuo() clasico.")
        return procesar_individuo(individuo_dir, fotos_dir, output_dir, cow_model, coco_model)

    # ── Validación geométrica de sanidad ──
    # Una vaca real no mide más de 280cm de largo ni pesa más de 1200kg.
    # Si el SfM produce dimensiones absurdas, el resultado es inestable
    # (típico con < 30 puntos 3D) → fallback al método híbrido.
    sfm_method = sfm_result.get('method', 'sfm_real')
    sfm_largo = sfm_result.get('largo_cm', 0)
    sfm_vol = sfm_result.get('volumen_litros', 0)
    sfm_ancho = sfm_result.get('ancho_cm', 0)

    if sfm_method != 'hibrido':
        sanity_fail = False
        reasons = []
        if sfm_largo > 280:
            sanity_fail = True
            reasons.append(f"largo={sfm_largo:.0f}cm > 280cm")
        if sfm_vol > 1200:
            sanity_fail = True
            reasons.append(f"vol={sfm_vol:.0f}L > 1200L")
        if sfm_ancho > 120:
            sanity_fail = True
            reasons.append(f"ancho={sfm_ancho:.0f}cm > 120cm")

        if sanity_fail:
            print(f"  [SANIDAD] SfM produjo dimensiones imposibles: {', '.join(reasons)}")
            print(f"  [SANIDAD] Descartando SfM → fallback a procesar_individuo() clasico.")
            return procesar_individuo(individuo_dir, fotos_dir, output_dir, cow_model, coco_model)

    # ── Fase 3: Guardar PLY, JSON, PNG ──
    ind_output = output_dir / nombre
    ind_output.mkdir(exist_ok=True)

    points_3d = sfm_result['points_3d']
    colors = sfm_result['colors']
    triangles = sfm_result['triangles']
    method = sfm_result.get('method', 'sfm_real')

    # PLY 3D
    ply_3d = ind_output / f"{nombre}_3d.ply"
    if triangles is not None and len(triangles) > 0:
        guardar_ply_con_malla(str(ply_3d), points_3d, triangles, colors)
    else:
        from core.reconstruccion_3d import guardar_ply as guardar_ply_nube
        guardar_ply_nube(str(ply_3d), points_3d, colors)

    # Visualizacion PNG
    vis_path = ind_output / f"{nombre}_modelo.png"
    try:
        from core.reconstruccion_3d import generar_imagen_resumen
        generar_imagen_resumen(sfm_result, str(vis_path), vaca_name=nombre)
    except Exception as e:
        print(f"  WARNING: No se pudo generar imagen resumen: {e}")

    # JSON resumen
    resumen = {
        'individuo': nombre,
        'categoria': categoria,
        'peso_real_kg': peso_real,
        'meses': meses,
        'edad_rango': edad_rango,
        'altura_estimada_cm': cow_height_cm,
        'metodo': method,
        'largo_cm': sfm_result.get('largo_cm', 0),
        'alto_cm': sfm_result.get('alto_cm', 0),
        'ancho_cm': sfm_result.get('ancho_cm', 0),
        'vol_total_litros': sfm_result.get('volumen_litros', 0),
        'vol_barril_litros': sfm_result.get('volumen_barril_litros', 0),
        'peso_estimado_kg': sfm_result.get('peso_kg', 0),
        'peso_barril_kg': sfm_result.get('peso_barril_kg', 0),
        'superficie_cm2': sfm_result.get('superficie_cm2', 0),
        'num_points': sfm_result.get('num_points', 0),
        'num_triangles': sfm_result.get('num_triangles', 0),
        'num_pairs': sfm_result.get('num_pairs', 0),
        'fotos_procesadas': len(fotos),
        'fotos_validas': len(frames),
        'fotos_descartadas': len(fotos) - len(frames),
    }
    with open(ind_output / f"{nombre}_resumen.json", 'w') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\n  RESULTADO {nombre.upper()} ({method.upper()}):")
    print(f"    Vol total:  {sfm_result.get('volumen_litros', 0):.1f} L")
    print(f"    Peso est:   {sfm_result.get('peso_kg', 0)} kg (real: {peso_real} kg)")
    print(f"    Largo:      {sfm_result.get('largo_cm', 0):.1f} cm | Alto: {sfm_result.get('alto_cm', 0):.1f} cm")
    print(f"    Puntos 3D:  {sfm_result.get('num_points', 0)} | Pares: {sfm_result.get('num_pairs', 0)}")
    print(f"    Archivos:   {ind_output}/")

    return resumen


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,
                        help='Directorio del dataset (default: checkpoints/Dataset Modelo 3d "grandes")')
    parser.add_argument('--output', type=str, default=None,
                        help='Directorio de salida (default: output_modelos3d_grandes)')
    parser.add_argument('--alturas-key', type=str, default=None,
                        help='Clave en alturas_individuos.json (default: alturas_<dataset_name>_cm)')
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    dataset_dir = Path(args.dataset) if args.dataset else project / 'checkpoints' / 'Dataset Modelo 3d "grandes" '
    output_dir = Path(args.output) if args.output else project / "output_modelos3d_grandes"
    output_dir.mkdir(exist_ok=True)

    # Directorio para imágenes diagnóstico de segmentación
    _diag_output_dir[0] = str(output_dir / "_diagnostico_segmentacion")
    _diag_counter[0] = 0

    # Cargar alturas reales de cada individuo
    alturas_path = project / "data" / "alturas_individuos.json"
    alturas = {}
    if alturas_path.exists():
        with open(alturas_path) as f:
            data_alturas = json.load(f)
        # Seleccionar la clave de alturas según el dataset (o la pasada por CLI)
        dataset_name = dataset_dir.name if dataset_dir else ''
        alturas_key = args.alturas_key or f'alturas_{dataset_name}_cm'
        if alturas_key in data_alturas:
            alturas = data_alturas[alturas_key]
        else:
            alturas = data_alturas.get('alturas_cm', {})
        print(f"Alturas reales cargadas: {len(alturas)} individuos (clave: {alturas_key if alturas_key in data_alturas else 'alturas_cm'})")
        for nombre, altura in alturas.items():
            print(f"  - {nombre}: {altura} cm")
    else:
        print("ADVERTENCIA: No se encontró alturas_individuos.json, usando alturas estimadas")

    print("\nCargando modelos YOLO...")
    cow_model = YOLO(str(project / "models" / "cow.pt"))
    coco_model = YOLO(str(project / "yolov8n.pt"))
    seg_model = YOLO(str(project / "yolov8s-seg.pt"))

    silueta_model = None
    silueta_path = project / "models" / "silueta_seg.pt"
    if silueta_path.exists():
        silueta_model = YOLO(str(silueta_path))
        print("  silueta_seg.pt cargado (silueta custom)")

    barril_model = None
    barril_path = project / "models" / "barril_seg.pt"
    if barril_path.exists():
        barril_model = YOLO(str(barril_path))
        print("  barril_seg.pt cargado (barril custom)")

    modelos = ["cow.pt", "yolov8s-seg.pt (fallback)"]
    if silueta_model: modelos.insert(1, "silueta_seg.pt")
    if barril_model: modelos.append("barril_seg.pt")
    print(f"  Modelos: {' + '.join(modelos)}")

    # Buscar individuos: primero intenta subcarpeta 3d_modelo_*, sino fotos directas
    individuos = []
    for ind_dir in sorted(dataset_dir.iterdir()):
        if not ind_dir.is_dir():
            continue
        # Opción 1: Buscar subcarpeta 3d_modelo_* (case insensitive)
        fotos_dir = None
        for sub in ind_dir.iterdir():
            if sub.is_dir() and sub.name.lower().startswith('3d_modelo'):
                fotos = [f for f in sub.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
                if fotos:
                    fotos_dir = sub
                    break
        # Opción 2: fotos directamente en la carpeta del individuo
        if fotos_dir is None:
            fotos = [f for f in ind_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
            if fotos:
                fotos_dir = ind_dir
        if fotos_dir:
            individuos.append((ind_dir, fotos_dir))

    print(f"\nIndividuos con carpeta 3d_modelo y fotos: {len(individuos)}")
    for ind_dir, fotos_dir in individuos:
        cat, peso, meses = parsear_nombre(ind_dir.name)
        n_fotos = len([f for f in fotos_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
        altura_str = f"{alturas[ind_dir.name]} cm (real)" if ind_dir.name in alturas else "estimada"
        print(f"  - {ind_dir.name}: {n_fotos} fotos (peso: {peso} kg, {meses} meses, alto: {altura_str})")

    # Procesar cada individuo con mejor foto
    resumen_global = []
    for ind_dir, fotos_dir in individuos:
        cow_height = alturas.get(ind_dir.name, None)
        resumen = procesar_individuo(ind_dir, fotos_dir, output_dir, cow_model, coco_model, cow_height_override=cow_height, seg_model=seg_model, barril_model=barril_model, silueta_model=silueta_model)
        if resumen:
            resumen_global.append(resumen)

    # Resumen global
    print(f"\n\n{'#'*60}")
    print(f"  RESUMEN GLOBAL - {len(resumen_global)} individuos procesados")
    print(f"{'#'*60}")
    print(f"\n  {'Individuo':<20} {'Peso Real':>10} {'Vol(L)':>8} {'Peso Vol':>10} {'Barril(L)':>10} {'Peso Barr':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for r in resumen_global:
        peso_vol = r['vol_total_litros'] * 1.03
        vbl = r['vol_barril_litros']
        if vbl is not None:
            peso_barril = vbl * 1.03
            barril_str = f"{vbl:>10.1f} {peso_barril:>8.1f} kg"
        else:
            barril_str = f"{'N/A':>10} {'N/A':>10}"
        print(f"  {r['individuo']:<20} {r['peso_real_kg']:>8} kg {r['vol_total_litros']:>8.1f} {peso_vol:>8.1f} kg {barril_str}")

    # Guardar resumen global
    with open(output_dir / "resumen_global.json", 'w') as f:
        json.dump(resumen_global, f, indent=2, ensure_ascii=False)
    print(f"\n  Archivos en: {output_dir}/")


if __name__ == '__main__':
    main()
