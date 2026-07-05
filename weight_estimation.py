"""
Estimación de peso de ganado a partir de imágenes usando visión por computadora.

Clase principal: WeightEstimator
  - Detecta el cuerpo del animal y sus keypoints con YOLO (cow.pt)
  - Detecta ojos con YOLO de segmentación (eye.pt) para escala por distancia inter-ocular
  - Detecta postes rojos (sticker.pt / color) para escala por altura conocida (50 cm)
  - Calcula dist1 (Body Length: pinbone → shoulderbone) y dist2 (Girth Vertical)
  - Aplica fórmula de Schaeffer: weight_kg = (BL × GV² × lb) / 300
  - Corrige por raza, categoría y edad (breed_coefficients.py)

Dependencias: ultralytics, opencv-python, numpy, depth_estimation, breed_coefficients
"""
import cv2
import numpy as np
import math
from ultralytics import YOLO
import os
import time

import base64

try:
    from depth_estimation import DepthEstimator
    DEPTH_ESTIMATOR_AVAILABLE = True
except ImportError:
    DEPTH_ESTIMATOR_AVAILABLE = False
    DepthEstimator = None


# ── Helpers para dibujar cinta roja + detección de piso (pasto) ──
# Saturación/Valor bajos para aceptar cintas pálidas/con sombra
_RED_HSV_LOWER1 = np.array([0, 40, 30])
_RED_HSV_UPPER1 = np.array([18, 255, 255])
_RED_HSV_LOWER2 = np.array([160, 40, 30])
_RED_HSV_UPPER2 = np.array([180, 255, 255])
_FLOOR_GREEN_LOWER = np.array([20, 25, 40])
_FLOOR_GREEN_UPPER = np.array([95, 255, 255])
_EXCLUDE_GRASS_FROM_RED_LOWER = np.array([21, 38, 123])
_EXCLUDE_GRASS_FROM_RED_UPPER = np.array([30, 115, 255])


def _rotated_tape_from_roi(roi_bgr, x_offset=0, y_offset=0):
    """Detecta la cinta roja como rectángulo rotado (no axis-aligned).
    Devuelve dict con los 2 extremos del eje largo del rect (top y bottom),
    el largo diagonal real (tape_px), el ángulo respecto a la vertical y
    los 4 corners en coords absolutas. None si no hay contorno válido.

    Soporta postes inclinados: el bbox axis-aligned tradicional sub-mide la
    cinta cuando está inclinada; el rotated rect preserva el largo real.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, _RED_HSV_LOWER1, _RED_HSV_UPPER1)
    red2 = cv2.inRange(hsv, _RED_HSV_LOWER2, _RED_HSV_UPPER2)
    red = cv2.bitwise_or(red1, red2)
    grass = cv2.inRange(hsv, _EXCLUDE_GRASS_FROM_RED_LOWER, _EXCLUDE_GRASS_FROM_RED_UPPER)
    red = cv2.bitwise_and(red, cv2.bitwise_not(grass))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Combinar todos los contornos significativos para cubrir cintas fragmentadas
    # (la línea quedaba corta cuando agarrábamos solo el mayor blob).
    valid_cnts = [c for c in contours if cv2.contourArea(c) >= 4]
    if not valid_cnts:
        return None
    all_pts = np.concatenate([c.reshape(-1, 2) for c in valid_cnts], axis=0).astype(np.float32)
    if all_pts.shape[0] < 4:
        return None
    rot = cv2.minAreaRect(all_pts)
    box = cv2.boxPoints(rot)
    box = np.array(box, dtype=np.float32)
    edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    edge_lens = [float(np.linalg.norm(b - a)) for a, b in edges]
    short_idx = sorted(range(4), key=lambda i: edge_lens[i])[:2]
    end_midpts = [(edges[i][0] + edges[i][1]) / 2.0 for i in short_idx]
    if end_midpts[0][1] < end_midpts[1][1]:
        top_end, bot_end = end_midpts[0], end_midpts[1]
    else:
        top_end, bot_end = end_midpts[1], end_midpts[0]
    tape_px = float(np.linalg.norm(bot_end - top_end))
    if tape_px < 5:
        return None
    dx = float(top_end[0] - bot_end[0])
    dy_up = float(bot_end[1] - top_end[1])
    angle_deg = float(np.degrees(np.arctan2(dx, max(1e-3, dy_up))))
    abs_corners = [[float(c[0] + x_offset), float(c[1] + y_offset)] for c in box.tolist()]
    return {
        'top_x': float(top_end[0] + x_offset),
        'top_y': float(top_end[1] + y_offset),
        'bot_x': float(bot_end[0] + x_offset),
        'bot_y': float(bot_end[1] + y_offset),
        'tape_px': tape_px,
        'angle_deg': angle_deg,
        'rot_corners': abs_corners,
    }


def _find_red_tape_rows(roi_bgr):
    """Dentro del ROI del bbox del poste, busca el tramo vertical rojo continuo
    empezando desde abajo. Retorna (y_top, y_bottom) relativos al ROI, o None.
    """
    if roi_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, _RED_HSV_LOWER1, _RED_HSV_UPPER1)
    red2 = cv2.inRange(hsv, _RED_HSV_LOWER2, _RED_HSV_UPPER2)
    red = cv2.bitwise_or(red1, red2)
    grass = cv2.inRange(hsv, _EXCLUDE_GRASS_FROM_RED_LOWER, _EXCLUDE_GRASS_FROM_RED_UPPER)
    red = cv2.bitwise_and(red, cv2.bitwise_not(grass))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # Para bbox estrechos (10-20 px), usar casi todo el ancho. Para anchos
    # usar solo el centro (evita confundir con objetos laterales).
    w_roi = red.shape[1]
    if w_roi <= 20:
        col_start, col_end = 0, w_roi
    else:
        col_start = int(w_roi * 0.25)
        col_end = int(w_roi * 0.75)
    center = red[:, col_start:col_end]
    row_ratio = np.mean(center > 0, axis=1)
    rows_with_red = np.where(row_ratio >= 0.15)[0]
    if rows_with_red.size == 0:
        return None

    y_bottom = int(rows_with_red.max())
    y_top = y_bottom
    min_gap = 4
    gap = 0
    for row in range(y_bottom, -1, -1):
        if row_ratio[row] >= 0.2:
            y_top = row
            gap = 0
        else:
            gap += 1
            if gap >= min_gap:
                break
    return y_top, y_bottom


def _detect_floor_below(image_bgr, x_center, y_start, _unused_fallback=None):
    """Detecta el piso usando AUTO-CALIBRACIÓN desde la propia imagen.

    1. Muestreo color del PASTO en el fondo de la imagen (últimas ~30 filas)
    2. Muestreo color del POSTE justo debajo de la cinta
    3. Camino recto hacia abajo en x=cx
    4. En cada fila comparo pixel vs ambos samples (LAB):
       - Más cerca del poste → sigue siendo poste
       - Más cerca del pasto → pasto
    5. Primera fila donde "pasto gana" por N consecutivas → piso

    Retorna (floor_y, confianza). Si no encuentra, (h_img-1, 0.0).
    """
    h_img, w_img = image_bgr.shape[:2]
    y_start = max(0, min(y_start, h_img - 2))
    if y_start >= h_img - 2:
        return h_img - 1, 0.0

    cx = int(x_center)
    cx = max(0, min(cx, w_img - 1))

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)

    # ── Muestra de POSTE: 15 filas debajo de la cinta, ±3 px de cx ──
    ps_y0 = min(h_img - 1, y_start + 2)
    ps_y1 = min(h_img, ps_y0 + 15)
    ps_x0 = max(0, cx - 3)
    ps_x1 = min(w_img, cx + 4)
    post_sample = lab[ps_y0:ps_y1, ps_x0:ps_x1]
    if post_sample.size == 0:
        return h_img - 1, 0.0
    post_lab = np.median(post_sample.reshape(-1, 3), axis=0).astype(np.int16)

    # ── Muestra de PASTO: últimas 30 filas de la imagen, todo el ancho alrededor del poste ──
    gs_half = 80
    gs_x0 = max(0, cx - gs_half)
    gs_x1 = min(w_img, cx + gs_half + 1)
    gs_y0 = max(0, h_img - 30)
    gs_y1 = h_img
    grass_sample = lab[gs_y0:gs_y1, gs_x0:gs_x1]
    if grass_sample.size == 0:
        return h_img - 1, 0.0
    grass_lab = np.median(grass_sample.reshape(-1, 3), axis=0).astype(np.int16)

    # Si los samples son muy parecidos (poste y pasto casi iguales), fallback
    sep_dist = float(np.linalg.norm(post_lab - grass_lab))
    if sep_dist < 10:
        # No distinguibles: fallback al fondo
        return h_img - 1, 0.0

    # ── Caminar hacia abajo en x=cx, columna ±2 px ──
    scan_y0 = ps_y1
    col_half = 2
    cx0 = max(0, cx - col_half)
    cx1 = min(w_img, cx + col_half + 1)

    # Scan pre-computado para poder analizar TODA la secuencia y elegir la
    # primera transición real (no la primera zona 100% estable)
    row_grass_ratio = np.zeros(h_img - scan_y0, dtype=np.float32)
    for i, row in enumerate(range(scan_y0, h_img)):
        col_lab = lab[row, cx0:cx1]
        d_post = np.linalg.norm(col_lab - post_lab, axis=1)
        d_grass = np.linalg.norm(col_lab - grass_lab, axis=1)
        row_grass_ratio[i] = float(np.mean(d_grass < d_post))

    # Buscamos la primera fila donde:
    #   - la fila actual tiene >= 40% pasto (transición iniciada)
    #   - en las próximas 10 filas, >= 50% son >= 40% pasto (es real, no ruido)
    persist_window = 10
    persist_frac = 0.5
    min_ratio = 0.4

    for i in range(len(row_grass_ratio)):
        if row_grass_ratio[i] < min_ratio:
            continue
        window = row_grass_ratio[i:i + persist_window]
        if len(window) < 3:
            continue
        passing = np.mean(window >= min_ratio)
        if passing >= persist_frac:
            return int(scan_y0 + i), float(row_grass_ratio[i])

    return h_img - 1, 0.0


def _draw_tape_and_floor_on(image_rgb, bbox):
    """Sobre image_rgb (RGB), dibuja la cinta roja detectada dentro del bbox
    y la línea de piso (pasto) debajo. Retorna {cx, top_tape, bottom_tape, floor, tape_px}
    o None si no se pudo detectar la cinta.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_img, w_img = image_rgb.shape[:2]
    x1 = max(0, min(x1, w_img - 1))
    x2 = max(0, min(x2, w_img))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(0, min(y2, h_img))
    if x2 <= x1 or y2 <= y1:
        return None

    # image_rgb es RGB. Para HSV necesitamos BGR o RGB → convertimos el ROI.
    roi_rgb = image_rgb[y1:y2, x1:x2]
    roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

    rot = _rotated_tape_from_roi(roi_bgr, x_offset=x1, y_offset=y1)
    if rot is None:
        # Fallback al método axis-aligned si el rotated falla
        tape = _find_red_tape_rows(roi_bgr)
        cx = (x1 + x2) // 2
        if tape is None:
            return None
        y_top_rel, y_bot_rel = tape
        line_y1 = y1 + y_top_rel
        line_y2 = y1 + y_bot_rel
        rot = {
            'top_x': float(cx), 'top_y': float(line_y1),
            'bot_x': float(cx), 'bot_y': float(line_y2),
            'tape_px': float(abs(line_y2 - line_y1)),
            'angle_deg': 0.0,
            'rot_corners': [[float(cx), float(line_y1)], [float(cx), float(line_y2)],
                            [float(cx), float(line_y2)], [float(cx), float(line_y1)]],
        }

    cx = int(round(rot['bot_x']))
    line_y1 = int(round(rot['top_y']))
    line_y2 = int(round(rot['bot_y']))
    top_x = int(round(rot['top_x']))

    # Cinta roja inclinada: línea entre los 2 extremos del rect rotado
    red_color = (255, 0, 0)
    cv2.line(image_rgb, (top_x, line_y1), (cx, line_y2), red_color, 1)
    cv2.circle(image_rgb, (top_x, line_y1), 2, red_color, -1)
    cv2.circle(image_rgb, (cx, line_y2), 2, red_color, -1)

    # Rotated rect en celeste para que se vea el tilt
    cyan = (100, 220, 255)
    pts = np.array([[int(round(c[0])), int(round(c[1]))] for c in rot['rot_corners']], dtype=np.int32)
    cv2.polylines(image_rgb, [pts], isClosed=True, color=cyan, thickness=1)
    if abs(rot['angle_deg']) > 1.0:
        cv2.putText(image_rgb, f"{rot['angle_deg']:+.1f}°",
                    (top_x + 6, line_y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cyan, 1)

    # Nuevo setup: cinta llega al piso → floor = bottom_tape (donde toca el suelo)
    floor_y = line_y2

    floor_color = (0, 255, 255)  # RGB cyan
    half_w = max(30, (x2 - x1))
    cv2.line(image_rgb, (cx - half_w, floor_y), (cx + half_w, floor_y), floor_color, 1)
    cv2.putText(image_rgb, "piso",
                (cx + half_w + 5, floor_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, floor_color, 2)

    return {
        'cx': cx,
        'top_tape': line_y1,
        'top_tape_x': top_x,
        'bottom_tape': line_y2,
        'floor': floor_y,
        'tape_px': float(rot['tape_px']),  # diagonal real
        'angle_deg': float(rot['angle_deg']),
        'rot_corners': rot['rot_corners'],
    }

try:
    from transformers import pipeline as hf_pipeline
    HF_DEPTH_AVAILABLE = True
except ImportError:
    HF_DEPTH_AVAILABLE = False

from breed_coefficients import get_weight_multiplier

# HSV del rojo de referencia (rojo tiene dos rangos: 0-10 y 170-180)
# Rangos más permisivos para detectar rojo puro (bandas rojas de 50cm)
RED_HSV_LOWER1 = np.array([0, 100, 100])    # Rojo bajo (0-10) - más permisivo
RED_HSV_UPPER1 = np.array([10, 255, 255])
RED_HSV_LOWER2 = np.array([170, 100, 100])  # Rojo alto (170-180) - más permisivo
RED_HSV_UPPER2 = np.array([180, 255, 255])
# HSV del pasto para excluirlo
GRASS_HSV_LOWER = np.array([21, 38, 123])
GRASS_HSV_UPPER = np.array([30, 115, 255])

class WeightEstimator:
    """Clase para estimar el peso del ganado usando detección de puntos clave"""
    
    def __init__(self, eye_model_path="models_yolo/eye.pt", cow_model_path="models_yolo/cow.pt",
                 conf_threshold=0.25, iou_threshold=0.45, eye_conf_multiplier=0.5, keypoint_conf_multiplier=0.5,
                 use_postes_reference=False, poste1_height_cm=100, poste2_height_cm=100,
                 distancia_postes_cm=200, focal_length_px=None, use_monocular_depth=False):
        """
        Inicializa los modelos YOLO para detección de ojos y puntos clave del ganado
        
        Args:
            eye_model_path: Ruta al modelo YOLO para detección de ojos
            cow_model_path: Ruta al modelo YOLO para detección de puntos clave del ganado
            conf_threshold: Umbral de confianza base para detecciones (0.0-1.0)
            iou_threshold: Umbral de IoU para Non-Maximum Suppression (0.0-1.0)
            eye_conf_multiplier: Multiplicador para conf de eye_model (más bajo = más permisivo para ojos)
            keypoint_conf_multiplier: Multiplicador para conf de cow_model keypoints (más bajo = más permisivo)
            use_postes_reference: Si True, intenta usar dos postes como referencia de profundidad
            poste1_height_cm: Altura real del poste 1 en cm
            poste2_height_cm: Altura real del poste 2 en cm
            distancia_postes_cm: Distancia real entre los dos postes en cm
            focal_length_px: Longitud focal de la cámara en píxeles (None para estimar)
        """
        self.eye_model = YOLO(eye_model_path)
        self.cow_model = YOLO(cow_model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # Para eye_model y keypoints usamos umbrales más bajos porque trabajan con ROIs recortados
        self.eye_conf = max(0.001, conf_threshold * eye_conf_multiplier)  # Más permisivo para ojos
        self.keypoint_conf = max(0.001, conf_threshold * keypoint_conf_multiplier)  # Más permisivo para keypoints
        self.poste_height_cm = (poste1_height_cm + poste2_height_cm) / 2

        # Modelo COCO preentrenado como fallback para detección de vacas
        # COCO class 19 = "cow" - detecta vacas de forma confiable en escenarios variados
        self.coco_model = None
        try:
            self.coco_model = YOLO("yolov8n.pt")  # auto-descarga ~6MB si no existe
            print("[WeightEstimator] Modelo COCO fallback (yolov8n.pt) cargado OK")
        except Exception as e:
            print(f"[WeightEstimator] No se pudo cargar modelo COCO fallback: {e}")
        
        # Inicializar estimador de profundidad con postes si está disponible
        self.use_postes_reference = use_postes_reference and DEPTH_ESTIMATOR_AVAILABLE
        if self.use_postes_reference:
            try:
                self.depth_estimator = DepthEstimator(
                    poste1_height_cm=poste1_height_cm,
                    poste2_height_cm=poste2_height_cm,
                    distancia_postes_cm=distancia_postes_cm,
                    focal_length_px=focal_length_px,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
            except Exception as e:
                print(f"Advertencia: No se pudo inicializar DepthEstimator: {e}")
                self.use_postes_reference = False
                self.depth_estimator = None
        else:
            self.depth_estimator = None

        # Inicializar pipeline de profundidad monocular (Depth Anything V2)
        self.depth_pipe = None
        if use_monocular_depth and HF_DEPTH_AVAILABLE:
            try:
                self.depth_pipe = hf_pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf"
                )
                print("Depth Anything V2 cargado correctamente para estimación de circunferencia torácica")
            except Exception as e:
                print(f"Advertencia: No se pudo cargar Depth Anything V2: {e}")
                self.depth_pipe = None
        elif use_monocular_depth and not HF_DEPTH_AVAILABLE:
            print("Advertencia: use_monocular_depth=True pero 'transformers' no está instalado. Usando fallback.")

    @staticmethod
    def euclidean(pt1, pt2):
        """Calcula la distancia euclidiana entre dos puntos"""
        return math.sqrt((pt1[0] - pt2[0])**2 + abs(pt1[1] - pt2[1])**2)

    def _measure_post_height(self, image, bbox, debug=False):
        """
        Mide la altura del poste usando directamente la altura del bounding box de YOLO.
        El modelo ya encuadra el poste correctamente, así que la altura del bbox
        es la medida más robusta (no depende de color, inclinación ni sombras).

        Returns:
            altura_px: Altura del bbox en píxeles, o None si inválido
        """
        x1, y1, x2, y2 = map(int, bbox)
        height_px = abs(y2 - y1)
        if debug:
            print(f"[MEASURE_POST] bbox={bbox} height_px={height_px}")
        return height_px if height_px > 0 else None
    
    def _select_post_for_scale(self, postes, image, animal_bbox=None, band_tolerance=0.5):
        """
        Selecciona el poste para escala: el más grande y más cercano al animal.
        Considera TODOS los postes detectados en la imagen completa, no solo los que están
        dentro de la banda vertical del animal. La banda vertical es solo una preferencia,
        no una restricción absoluta.
        También calcula la altura medida del poste (línea roja dentro del bbox).
        """
        if not postes:
            return None, [], []

        rejected = []
        candidates = []
        candidates_in_band = []  # Postes dentro de la banda (preferidos)

        if animal_bbox:
            ax1, ay1, ax2, ay2 = animal_bbox
            animal_height = max(1, ay2 - ay1)
            band_top = ay1 - int(animal_height * band_tolerance)
            img_h = image.shape[0]
            band_bottom = img_h - 1  # No limitar por abajo: los postes están a nivel del suelo
            animal_center = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
        else:
            band_top = None
            band_bottom = None
            animal_center = None

        # Procesar TODOS los postes detectados en la imagen completa
        for p in postes:
            x1, y1, x2, y2 = p['bbox']
            height = y2 - y1
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            distance = self.euclidean(center, animal_center) if animal_center else 0
            
            # Calcular altura medida del poste (línea roja)
            measured_height = self._measure_post_height(image, p['bbox'], debug=False)
            
            # Verificar si está dentro de la banda vertical (preferencia, no requisito)
            in_band = True
            if band_top is not None and band_bottom is not None:
                in_band = y1 >= band_top  # Solo verificar límite superior (excluir cielo/fondo)
            
            poste_data = {
                **p, 
                'height': height, 
                'distance': distance, 
                'measured_height_px': measured_height,
                'in_band': in_band
            }
            
            if in_band:
                # Preferir postes dentro de la banda
                candidates_in_band.append(poste_data)
            else:
                # Postes fuera de la banda también son candidatos válidos
                candidates.append(poste_data)

        # Si hay postes dentro de la banda, usarlos primero (ordenados por tamaño y distancia)
        if candidates_in_band:
            candidates_in_band.sort(key=lambda c: (c['height'], -c['distance']), reverse=True)
            selected = candidates_in_band[0]
            # Los demás postes dentro de la banda son también candidatos
            all_candidates = candidates_in_band + candidates
            return selected, all_candidates, rejected
        
        # Si no hay postes en la banda, usar todos los postes detectados
        if candidates:
            candidates.sort(key=lambda c: (c['height'], -c['distance']), reverse=True)
            selected = candidates[0]
            return selected, candidates, rejected
        
        # No hay postes detectados
        return None, [], rejected
    
    def _estimate_girth_circumference(self, image_bgr, kp3, kp4, cm_per_px, _log):
        """
        Estima la circunferencia torácica usando profundidad monocular.

        Algoritmo:
        1. Ejecutar depth_pipe sobre la imagen -> mapa de profundidad relativa
        2. El girth vertical (semi-eje b) = dist(KP3, KP4) / 2, en cm
        3. Punto medio del girth = midpoint(KP3, KP4) en pixeles
        4. Desde el punto medio, recorrer horizontalmente en ambas direcciones
           buscando pixeles con profundidad similar (dentro del 15% del valor central)
        5. El ancho horizontal encontrado = semi-eje a, convertido a cm
        6. Circunferencia elipse = Ramanujan: pi(a+b)(1 + 3h/(10+sqrt(4-3h)))
           donde h = ((a-b)/(a+b))^2

        Args:
            image_bgr: Imagen en formato BGR (numpy array)
            kp3: Tupla (x, y) del keypoint girth bottom
            kp4: Tupla (x, y) del keypoint girth top
            cm_per_px: Factor de escala cm/pixel
            _log: Funcion de logging

        Returns:
            (circumference_cm, left_edge, right_edge) o None si no se pudo calcular
        """
        if self.depth_pipe is None:
            return None

        try:
            from PIL import Image as PILImage

            # Convertir BGR a RGB y luego a PIL Image
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)

            # Ejecutar pipeline de profundidad
            depth_result = self.depth_pipe(pil_image)
            depth_map = np.array(depth_result['depth'], dtype=np.float32)

            _log(f"depth_map: shape={depth_map.shape} min={depth_map.min():.2f} max={depth_map.max():.2f}")

            # Punto medio del girth en pixeles
            mid_x = int((kp3[0] + kp4[0]) / 2)
            mid_y = int((kp3[1] + kp4[1]) / 2)
            h, w = depth_map.shape[:2]

            # Asegurar que el punto medio esta dentro de la imagen
            mid_x = max(2, min(mid_x, w - 3))
            mid_y = max(2, min(mid_y, h - 3))

            # Valor de profundidad central (promedio 5x5 px)
            patch = depth_map[mid_y - 2:mid_y + 3, mid_x - 2:mid_x + 3]
            center_depth = float(np.mean(patch))

            if center_depth <= 0:
                _log(f"girth_depth: center_depth={center_depth:.2f} <= 0, abortando")
                return None

            _log(f"girth_depth: mid=({mid_x},{mid_y}) center_depth={center_depth:.2f}")

            # Umbral de similitud: 15% del valor central
            depth_tolerance = center_depth * 0.15
            depth_min = center_depth - depth_tolerance
            depth_max = center_depth + depth_tolerance

            # Escanear hacia la derecha
            right_edge = mid_x
            consecutive_out = 0
            for x in range(mid_x + 1, w):
                val = float(depth_map[mid_y, x])
                if depth_min <= val <= depth_max:
                    right_edge = x
                    consecutive_out = 0
                else:
                    consecutive_out += 1
                    if consecutive_out >= 3:
                        break

            # Escanear hacia la izquierda
            left_edge = mid_x
            consecutive_out = 0
            for x in range(mid_x - 1, -1, -1):
                val = float(depth_map[mid_y, x])
                if depth_min <= val <= depth_max:
                    left_edge = x
                    consecutive_out = 0
                else:
                    consecutive_out += 1
                    if consecutive_out >= 3:
                        break

            ancho_horizontal_px = right_edge - left_edge
            if ancho_horizontal_px <= 0:
                _log(f"girth_depth: ancho_horizontal_px={ancho_horizontal_px} <= 0, abortando")
                return None

            # Distancia vertical del girth en pixeles
            dist_vertical_px = self.euclidean(kp3, kp4)

            # Semi-ejes de la elipse en cm
            semi_eje_a = (ancho_horizontal_px * cm_per_px) / 2  # horizontal
            semi_eje_b = (dist_vertical_px * cm_per_px) / 2     # vertical

            _log(f"girth_ellipse: ancho_px={ancho_horizontal_px} vert_px={dist_vertical_px:.1f} semi_a={semi_eje_a:.2f}cm semi_b={semi_eje_b:.2f}cm")

            # Ramanujan segunda aproximacion para circunferencia de elipse
            a, b = semi_eje_a, semi_eje_b
            if a + b <= 0:
                _log(f"girth_ellipse: a+b={a+b:.2f} <= 0, abortando")
                return None

            h_param = ((a - b) / (a + b)) ** 2
            circumference = math.pi * (a + b) * (1 + 3 * h_param / (10 + math.sqrt(4 - 3 * h_param)))

            _log(f"girth_circumference: {circumference:.2f}cm (Ramanujan) left={left_edge} right={right_edge}")

            return (circumference, left_edge, right_edge)

        except Exception as e:
            _log(f"girth_depth: ERROR {e}")
            return None

    def _get_depth_scale_correction(self, image_bgr, animal_bbox, poste_candidates, _log):
        """
        Usa Depth Anything V2 para estimar la profundidad relativa entre la vaca
        y los postes de referencia, retornando un factor de corrección para la escala cm/px.

        La escala base (cm/px) asume que vaca y poste están a la misma distancia de la cámara.
        Si la vaca está más cerca o más lejos, la escala necesita ajuste proporcional.

        Depth Anything V2 produce valores de profundidad relativa (disparity-like):
        - Valores más altos = más cerca de la cámara
        - correction = depth_post / depth_cow
          - Si cow más cerca (depth_cow > depth_post): ratio < 1 → escala se reduce
          - Si cow más lejos (depth_cow < depth_post): ratio > 1 → escala se aumenta

        Returns:
            float: factor de corrección (1.0 si no se puede calcular)
        """
        if self.depth_pipe is None:
            return 1.0

        if not animal_bbox or not poste_candidates:
            return 1.0

        try:
            from PIL import Image as PILImage

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image_rgb)

            depth_result = self.depth_pipe(pil_image)
            depth_map = np.array(depth_result['depth'], dtype=np.float32)

            h, w = depth_map.shape[:2]
            _log(f"depth_correction: depth_map shape={depth_map.shape} min={depth_map.min():.2f} max={depth_map.max():.2f}")

            # Muestrear profundidad en el centro de la vaca (parche 5x5)
            ax1, ay1, ax2, ay2 = map(int, animal_bbox)
            cow_cx = max(2, min((ax1 + ax2) // 2, w - 3))
            cow_cy = max(2, min((ay1 + ay2) // 2, h - 3))
            cow_depth = float(np.mean(depth_map[cow_cy - 2:cow_cy + 3, cow_cx - 2:cow_cx + 3]))

            _log(f"depth_correction: cow center=({cow_cx},{cow_cy}) depth_value={cow_depth:.4f}")

            # Muestrear profundidad en cada poste candidato
            post_depths = []
            for idx, p in enumerate(poste_candidates):
                px1, py1, px2, py2 = map(int, p['bbox'])
                pcx = max(2, min((px1 + px2) // 2, w - 3))
                pcy = max(2, min((py1 + py2) // 2, h - 3))
                pd = float(np.mean(depth_map[pcy - 2:pcy + 3, pcx - 2:pcx + 3]))
                if pd > 0:
                    post_depths.append(pd)
                    _log(f"depth_correction: post[{idx}] center=({pcx},{pcy}) depth_value={pd:.4f}")

            if not post_depths or cow_depth <= 0:
                _log(f"depth_correction: skipped (cow_depth={cow_depth:.4f} valid_posts={len(post_depths)})")
                return 1.0

            avg_post_depth = sum(post_depths) / len(post_depths)

            # Factor de corrección: depth_post / depth_cow
            # (Depth Anything V2 produce inverse depth: mayor valor = más cerca)
            correction = avg_post_depth / cow_depth

            # Limitar a rango razonable (0.5x a 2.0x) para evitar valores extremos
            correction_clamped = max(0.5, min(2.0, correction))

            _log(f"depth_correction: avg_post_depth={avg_post_depth:.4f} cow_depth={cow_depth:.4f} "
                 f"raw_ratio={correction:.4f} clamped={correction_clamped:.4f}")

            if abs(correction - correction_clamped) > 0.01:
                _log(f"depth_correction: WARNING ratio clamped from {correction:.4f} to {correction_clamped:.4f}")

            return correction_clamped

        except Exception as e:
            _log(f"depth_correction: ERROR {e}")
            return 1.0

    # ── Helpers for scan/analyze two-phase flow ──

    def _load_and_resize(self, img_path, _log):
        """Load image and apply letterbox resize. Returns tuple of image data."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {img_path}")
        _log(f"start img_path={img_path} shape={getattr(img, 'shape', None)}")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        img = np.ascontiguousarray(img)

        new_width = 1040
        new_height = 640
        h_orig, w_orig = img.shape[:2]
        scale_factor = min(new_width / w_orig, new_height / h_orig)
        scaled_w = int(w_orig * scale_factor)
        scaled_h = int(h_orig * scale_factor)
        scaled_img = cv2.resize(img, (scaled_w, scaled_h))
        resized_image = np.full((new_height, new_width, 3), 114, dtype=np.uint8)
        pad_x = (new_width - scaled_w) // 2
        pad_y = (new_height - scaled_h) // 2
        resized_image[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = scaled_img
        _log(f"resize: orig={w_orig}x{h_orig} scale={scale_factor:.4f} scaled={scaled_w}x{scaled_h} pad=({pad_x},{pad_y})")

        img_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        return img, resized_image, img_rgb, scale_factor, pad_x, pad_y, w_orig, h_orig

    def _detect_all_cows(self, img, resized_image, scale_factor, pad_x, pad_y, w_orig, h_orig, yolo_imgsz, _log):
        """Run 4-strategy cow detection. Returns (boxes, keypoints, scores, classes, detected) — all arrays contain ALL detections."""
        new_width, new_height = 1040, 640
        _cow_kwargs = dict(save=False, conf=self.keypoint_conf, iou=self.iou_threshold)
        if yolo_imgsz:
            _cow_kwargs['imgsz'] = yolo_imgsz

        _cow_boxes_mapped = None
        _cow_keypoints_mapped = None
        _cow_scores = None
        _cow_classes = None
        _cow_detected = False

        # Strategy 1: original image
        _log(f"cow_detection: intentando en imagen original ({w_orig}x{h_orig}) conf={self.keypoint_conf:.4f}")
        results2_orig = self.cow_model(img, **_cow_kwargs)
        for result in results2_orig:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_orig = result.boxes.xyxy.cpu().numpy()
                if len(boxes_orig) > 0:
                    _cow_detected = True
                    _cow_boxes_mapped = boxes_orig.copy()
                    _cow_boxes_mapped[:, 0] = boxes_orig[:, 0] * scale_factor + pad_x
                    _cow_boxes_mapped[:, 1] = boxes_orig[:, 1] * scale_factor + pad_y
                    _cow_boxes_mapped[:, 2] = boxes_orig[:, 2] * scale_factor + pad_x
                    _cow_boxes_mapped[:, 3] = boxes_orig[:, 3] * scale_factor + pad_y
                    _cow_scores = result.boxes.conf.cpu().numpy()
                    _cow_classes = result.boxes.cls.cpu().numpy()
                    if result.keypoints is not None:
                        kps_orig = result.keypoints.data.cpu().numpy()
                        _cow_keypoints_mapped = kps_orig.copy()
                        _cow_keypoints_mapped[..., 0] = kps_orig[..., 0] * scale_factor + pad_x
                        _cow_keypoints_mapped[..., 1] = kps_orig[..., 1] * scale_factor + pad_y
                    _log(f"cow_detection: {len(boxes_orig)} detecciones en imagen original (mapeadas a letterbox)")
                    break

        if not _cow_detected:
            # Strategy 2: letterbox image
            _log(f"cow_detection: nada en original, reintentando en letterbox ({new_width}x{new_height})")
            results2_lb = self.cow_model(resized_image, **_cow_kwargs)
            for result in results2_lb:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes_lb = result.boxes.xyxy.cpu().numpy()
                    if len(boxes_lb) > 0:
                        _cow_detected = True
                        _cow_boxes_mapped = boxes_lb
                        _cow_scores = result.boxes.conf.cpu().numpy()
                        _cow_classes = result.boxes.cls.cpu().numpy()
                        if result.keypoints is not None:
                            _cow_keypoints_mapped = result.keypoints.data.cpu().numpy()
                        _log(f"cow_detection: {len(boxes_lb)} detecciones en letterbox (fallback 1)")
                        break

        if not _cow_detected:
            # Strategy 3: augment=True
            _log(f"cow_detection: reintentando con augment=True en imagen original")
            try:
                _aug_kwargs = dict(_cow_kwargs)
                _aug_kwargs['augment'] = True
                results2_aug = self.cow_model(img, **_aug_kwargs)
                for result in results2_aug:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes_aug = result.boxes.xyxy.cpu().numpy()
                        if len(boxes_aug) > 0:
                            _cow_detected = True
                            _cow_boxes_mapped = boxes_aug.copy()
                            _cow_boxes_mapped[:, 0] = boxes_aug[:, 0] * scale_factor + pad_x
                            _cow_boxes_mapped[:, 1] = boxes_aug[:, 1] * scale_factor + pad_y
                            _cow_boxes_mapped[:, 2] = boxes_aug[:, 2] * scale_factor + pad_x
                            _cow_boxes_mapped[:, 3] = boxes_aug[:, 3] * scale_factor + pad_y
                            _cow_scores = result.boxes.conf.cpu().numpy()
                            _cow_classes = result.boxes.cls.cpu().numpy()
                            if result.keypoints is not None:
                                kps_aug = result.keypoints.data.cpu().numpy()
                                _cow_keypoints_mapped = kps_aug.copy()
                                _cow_keypoints_mapped[..., 0] = kps_aug[..., 0] * scale_factor + pad_x
                                _cow_keypoints_mapped[..., 1] = kps_aug[..., 1] * scale_factor + pad_y
                            _log(f"cow_detection: {len(boxes_aug)} detecciones con augment=True (fallback 2)")
                            break
            except Exception as e:
                _log(f"cow_detection: augment falló: {e}")

        if not _cow_detected and self.coco_model is not None:
            # Strategy 4: COCO fallback — return ALL cow detections
            _log(f"cow_detection: reintentando con modelo COCO (yolov8n) en imagen original")
            try:
                _coco_results = self.coco_model(img, save=False, conf=0.1, iou=self.iou_threshold, classes=[19])
                _all_coco_boxes = []
                _all_coco_scores = []
                for result in _coco_results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes_coco = result.boxes.xyxy.cpu().numpy()
                        scores_coco = result.boxes.conf.cpu().numpy()
                        for box, score in zip(boxes_coco, scores_coco):
                            _all_coco_boxes.append(box)
                            _all_coco_scores.append(float(score))

                if _all_coco_boxes:
                    _cow_detected = True
                    # Sort by score descending (best first)
                    _sorted_idx = sorted(range(len(_all_coco_scores)),
                                         key=lambda i: _all_coco_scores[i], reverse=True)
                    _all_coco_boxes = [_all_coco_boxes[i] for i in _sorted_idx]
                    _all_coco_scores = [_all_coco_scores[i] for i in _sorted_idx]
                    _log(f"cow_detection: COCO detectó {len(_all_coco_boxes)} vacas "
                         f"scores={[f'{s:.3f}' for s in _all_coco_scores]} (fallback 3)")

                    # Map ALL boxes to letterbox space
                    mapped_boxes = []
                    for box in _all_coco_boxes:
                        mapped = box.copy()
                        mapped[0] = box[0] * scale_factor + pad_x
                        mapped[1] = box[1] * scale_factor + pad_y
                        mapped[2] = box[2] * scale_factor + pad_x
                        mapped[3] = box[3] * scale_factor + pad_y
                        mapped_boxes.append(mapped)
                    _cow_boxes_mapped = np.array(mapped_boxes)
                    _cow_scores = np.array(_all_coco_scores)
                    _cow_classes = np.array([19.0] * len(_all_coco_boxes))
                    _cow_keypoints_mapped = None

                    # Two-stage: try cow.pt on best detection's crop to recover keypoints
                    _best_box = _all_coco_boxes[0]
                    cx1, cy1, cx2, cy2 = map(int, _best_box)
                    cw, ch = cx2 - cx1, cy2 - cy1
                    margin_x, margin_y = int(cw * 0.25), int(ch * 0.25)
                    crop_x1 = max(0, cx1 - margin_x)
                    crop_y1 = max(0, cy1 - margin_y)
                    crop_x2 = min(img.shape[1], cx2 + margin_x)
                    crop_y2 = min(img.shape[0], cy2 + margin_y)
                    cow_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

                    if cow_crop.size > 0:
                        _log(f"cow_detection: two-stage cow.pt on best COCO crop [{crop_x1},{crop_y1},{crop_x2},{crop_y2}]")
                        try:
                            _crop_results = self.cow_model(cow_crop, save=False, conf=self.keypoint_conf, iou=self.iou_threshold)
                            for _cr in _crop_results:
                                if _cr.boxes is not None and len(_cr.boxes) > 0 and _cr.keypoints is not None:
                                    _crop_kps = _cr.keypoints.data.cpu().numpy()
                                    _crop_boxes = _cr.boxes.xyxy.cpu().numpy()
                                    # Map crop coords → original → letterbox
                                    _crop_kps[..., 0] = (_crop_kps[..., 0] + crop_x1) * scale_factor + pad_x
                                    _crop_kps[..., 1] = (_crop_kps[..., 1] + crop_y1) * scale_factor + pad_y
                                    _crop_boxes[:, 0] = (_crop_boxes[:, 0] + crop_x1) * scale_factor + pad_x
                                    _crop_boxes[:, 1] = (_crop_boxes[:, 1] + crop_y1) * scale_factor + pad_y
                                    _crop_boxes[:, 2] = (_crop_boxes[:, 2] + crop_x1) * scale_factor + pad_x
                                    _crop_boxes[:, 3] = (_crop_boxes[:, 3] + crop_y1) * scale_factor + pad_y
                                    # Build keypoints array for ALL cows: only index 0 has real data
                                    n_kps = _crop_kps.shape[1]
                                    _cow_keypoints_mapped = np.zeros((len(_all_coco_boxes), n_kps, 3))
                                    _cow_keypoints_mapped[0] = _crop_kps[0]
                                    # Also refine box 0 with cow.pt's tighter bbox
                                    _cow_boxes_mapped[0] = _crop_boxes[0]
                                    _log(f"cow_detection: two-stage SUCCESS - keypoints for cow 0")
                                    break
                            if _cow_keypoints_mapped is None:
                                _log(f"cow_detection: two-stage - cow.pt found no keypoints on crop")
                        except Exception as e:
                            _log(f"cow_detection: two-stage cow.pt crop failed: {e}")
            except Exception as e:
                _log(f"cow_detection: COCO fallback falló: {e}")

        if not _cow_detected:
            _log(f"cow_detection: no se detectó animal con ninguna estrategia")

        return _cow_boxes_mapped, _cow_keypoints_mapped, _cow_scores, _cow_classes, _cow_detected

    def scan_detections(self, img_path, debug=False, debug_context="SCAN", yolo_imgsz=None):
        """
        Phase 1 of two-phase flow: run detection only (no weight estimation).
        Returns all cows (with thumbnails) and all posts detected.
        """
        def _log(msg):
            if debug:
                print(f"[SCAN] {debug_context} {msg}")

        img, resized_image, img_rgb, scale_factor, pad_x, pad_y, w_orig, h_orig = \
            self._load_and_resize(img_path, _log)

        boxes, keypoints, scores, classes, detected = \
            self._detect_all_cows(img, resized_image, scale_factor, pad_x, pad_y, w_orig, h_orig, yolo_imgsz, _log)

        # Build cow list with thumbnails
        cows = []
        if detected and boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                has_kp = (keypoints is not None and i < len(keypoints) and len(keypoints[i]) >= 5)
                score_val = float(scores[i]) if scores is not None else 0.0

                # Crop thumbnail from the resized RGB image
                th_y1 = max(0, y1)
                th_y2 = min(resized_image.shape[0], y2)
                th_x1 = max(0, x1)
                th_x2 = min(resized_image.shape[1], x2)
                crop = resized_image[th_y1:th_y2, th_x1:th_x2]
                thumb_b64 = ''
                if crop.size > 0:
                    # Resize to max 120px wide
                    ch, cw = crop.shape[:2]
                    if cw > 120:
                        ratio = 120.0 / cw
                        crop = cv2.resize(crop, (120, int(ch * ratio)))
                    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    thumb_b64 = base64.b64encode(buf).decode('utf-8')

                cows.append({
                    'index': i,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': round(score_val, 3),
                    'has_keypoints': has_kp,
                    'thumbnail_b64': thumb_b64,
                })

        # Detect posts
        posts = []
        if self.use_postes_reference and self.depth_estimator:
            postes_all = self.depth_estimator.detect_postes_all(resized_image)
            if len(postes_all) < 2:
                postes_orig = self.depth_estimator.detect_postes_all(img)
                if len(postes_orig) > len(postes_all):
                    postes_mapped = []
                    for p in postes_orig:
                        ox1, oy1, ox2, oy2 = p['bbox']
                        mx1 = ox1 * scale_factor + pad_x
                        my1 = oy1 * scale_factor + pad_y
                        mx2 = ox2 * scale_factor + pad_x
                        my2 = oy2 * scale_factor + pad_y
                        mapped_p = dict(p)
                        mapped_p['bbox'] = [mx1, my1, mx2, my2]
                        postes_mapped.append(mapped_p)
                    postes_all = postes_mapped

            # Build post list from postes_all in detection order (by score desc).
            # This is the order the user sees in the UI and selects from.
            # We measure height for each post here; _select_post_for_scale is NOT used
            # because it reorders by height/distance, which would make the indices
            # inconsistent with what estimate_weight receives as post_indices.
            scan_posts_for_ui = []
            for p in postes_all:
                measured_height = self._measure_post_height(resized_image, p['bbox'], debug=False)
                scan_posts_for_ui.append({**p, 'measured_height_px': measured_height})

            for idx, p in enumerate(scan_posts_for_ui):
                measured_h = p.get('measured_height_px')

                # Crop post thumbnail from resized image
                pb = p['bbox']
                px1c, py1c, px2c, py2c = int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])
                # Add small margin around post
                pm = max(10, int((px2c - px1c) * 0.3))
                px1c = max(0, px1c - pm)
                py1c = max(0, py1c - pm)
                px2c = min(resized_image.shape[1], px2c + pm)
                py2c = min(resized_image.shape[0], py2c + pm)
                post_crop = resized_image[py1c:py2c, px1c:px2c]
                post_thumb_b64 = ''
                if post_crop.size > 0:
                    ph, pw = post_crop.shape[:2]
                    if pw > 100:
                        ratio = 100.0 / pw
                        post_crop = cv2.resize(post_crop, (100, int(ph * ratio)))
                    _, pbuf = cv2.imencode('.jpg', post_crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    post_thumb_b64 = base64.b64encode(pbuf).decode('utf-8')

                posts.append({
                    'index': idx,
                    'bbox': [float(v) for v in p['bbox']],
                    'measured_height_px': float(measured_h) if measured_h else None,
                    'score': round(float(p.get('score', 0)), 3),
                    'red_ratio': round(float(p.get('yellow_ratio', 0)), 3),
                    'in_band': bool(p.get('in_band', False)),
                    'thumbnail_b64': post_thumb_b64,
                })

        # Build annotated preview image showing all cows and posts
        preview_rgb = img_rgb.copy()
        for cow in cows:
            cx1, cy1, cx2, cy2 = map(int, cow['bbox'])
            color = (0, 255, 0)
            cv2.rectangle(preview_rgb, (cx1, cy1), (cx2, cy2), color, 2)
            lbl = f"Vaca {cow['index']+1} ({cow['score']*100:.0f}%)"
            cv2.putText(preview_rgb, lbl, (cx1, max(15, cy1 - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Scan phase: solo bbox de postes + marca mínima de cinta roja.
        # NO dibujamos piso ni rectángulo acá - eso viene después de que el usuario
        # elija cuáles 2 postes usar como referencia.
        for post in posts:
            px1, py1, px2, py2 = map(int, post['bbox'])
            color = (255, 0, 255)
            cv2.rectangle(preview_rgb, (px1, py1), (px2, py2), color, 2)
            h_str = f"{post['measured_height_px']:.0f}px" if post['measured_height_px'] else '?'
            lbl = f"Poste {post['index']+1} ({h_str})"
            cv2.putText(preview_rgb, lbl, (px1, max(15, py1 - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # Marcador visual de la cinta detectada (línea roja central, 1px)
            # Esto ayuda al usuario a confirmar cuáles postes fueron bien detectados
            cx = int((px1 + px2) / 2)
            roi_rgb = preview_rgb[py1:py2, px1:px2]
            if roi_rgb.size > 0:
                roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
                tape = _find_red_tape_rows(roi_bgr)
                if tape is not None:
                    y_top_rel, y_bot_rel = tape
                    ly1 = py1 + y_top_rel
                    ly2 = py1 + y_bot_rel
                    cv2.line(preview_rgb, (cx, ly1), (cx, ly2), (255, 0, 0), 1)
                    cv2.circle(preview_rgb, (cx, ly1), 2, (255, 0, 0), -1)
                    cv2.circle(preview_rgb, (cx, ly2), 2, (255, 0, 0), -1)

        # Encode preview as base64 JPEG
        preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', preview_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        preview_b64 = base64.b64encode(buf).decode('utf-8')

        _log(f"scan complete: {len(cows)} cows, {len(posts)} posts")
        return {'cows': cows, 'posts': posts, 'preview_b64': preview_b64}

    def estimate_weight(self, img_path, visualize=True, debug=False, debug_context="", return_eye_coords=False, return_keypoint_coords=False, roi_offset=(0, 0), scale_method='both', breed="desconocido", category="desconocido", age_range="desconocido", override_cm_per_px=None, yolo_imgsz=None, cow_index=0, post_indices=None, locked_reference=None):
        """
        Estima el peso del ganado en una imagen
        
        Args:
            img_path: Ruta a la imagen del ganado
            visualize: Si True, retorna también la imagen procesada con visualizaciones
            debug: Si True, imprime logs de debug
            debug_context: Contexto adicional para los logs
            return_eye_coords: Si True, retorna también las coordenadas de los ojos detectados
            return_keypoint_coords: Si True, retorna también las coordenadas de los keypoints usados para dist1 y dist2
            scale_method: Método de escala a usar ('both', 'eyes', 'poste')
                - 'both': Busca ojos primero, luego poste rojo como fallback
                - 'eyes': Solo busca ojos
                - 'poste': Solo busca poste rojo (50cm)
        
        Returns:
            Si visualize=True y return_eye_coords=False y return_keypoint_coords=False: (imagen_procesada, peso_estimado)
            Si visualize=True y return_eye_coords=True: (imagen_procesada, peso_estimado, eye_coords, keypoint_coords)
            Si visualize=False y return_eye_coords=True: (peso_estimado, eye_coords, keypoint_coords) o (None, [], [])
            Si visualize=False y return_eye_coords=False: peso_estimado o None si no se puede calcular
        """
        def _log(msg: str):
            if debug:
                prefix = "[WEIGHT]"
                ctx = f" {debug_context}".strip()
                print(f"{prefix}{(' ' + ctx) if ctx else ''} {msg}")

        t0 = time.time()

        # Use shared helpers for image loading and cow detection
        img, resized_image, img_rgb, scale_factor, pad_x, pad_y, w_orig, h_orig = \
            self._load_and_resize(img_path, _log)
        _log(f"start img_path={img_path} shape={getattr(img, 'shape', None)} visualize={visualize}")

        new_width = 1040
        new_height = 640

        _cow_boxes_mapped, _cow_keypoints_mapped, _cow_scores, _cow_classes, _cow_detected = \
            self._detect_all_cows(img, resized_image, scale_factor, pad_x, pad_y, w_orig, h_orig, yolo_imgsz, _log)

        # Select which cow to analyze (cow_index param, clamped to valid range)
        _ci = 0
        if _cow_detected and _cow_boxes_mapped is not None and len(_cow_boxes_mapped) > 0:
            _ci = min(cow_index, len(_cow_boxes_mapped) - 1)
            _ci = max(0, _ci)
            if cow_index != 0:
                _log(f"cow_selection: cow_index={cow_index} -> using index {_ci} of {len(_cow_boxes_mapped)} detections")

        # Two-stage recovery: if selected cow has no keypoints, try cow.pt on its crop
        if (_cow_detected and _cow_boxes_mapped is not None and _ci < len(_cow_boxes_mapped)):
            _need_kp_recovery = (_cow_keypoints_mapped is None or
                                 _ci >= len(_cow_keypoints_mapped) or
                                 float(np.max(_cow_keypoints_mapped[_ci][:, 2])) < 0.01)
            if _need_kp_recovery:
                _log(f"cow_selection: no keypoints for cow {_ci}, trying two-stage recovery")
                _box_lb = _cow_boxes_mapped[_ci]
                # Reverse map letterbox → original image
                _ox1 = int((_box_lb[0] - pad_x) / scale_factor)
                _oy1 = int((_box_lb[1] - pad_y) / scale_factor)
                _ox2 = int((_box_lb[2] - pad_x) / scale_factor)
                _oy2 = int((_box_lb[3] - pad_y) / scale_factor)
                _ow, _oh = _ox2 - _ox1, _oy2 - _oy1
                _mx, _my = int(_ow * 0.25), int(_oh * 0.25)
                _cx1 = max(0, _ox1 - _mx)
                _cy1 = max(0, _oy1 - _my)
                _cx2 = min(img.shape[1], _ox2 + _mx)
                _cy2 = min(img.shape[0], _oy2 + _my)
                _cow_crop = img[_cy1:_cy2, _cx1:_cx2]
                if _cow_crop.size > 0:
                    try:
                        _rec_results = self.cow_model(_cow_crop, save=False, conf=self.keypoint_conf, iou=self.iou_threshold)
                        for _rr in _rec_results:
                            if _rr.boxes is not None and len(_rr.boxes) > 0 and _rr.keypoints is not None:
                                _rec_kps = _rr.keypoints.data.cpu().numpy()
                                _rec_boxes = _rr.boxes.xyxy.cpu().numpy()
                                # Map crop → original → letterbox
                                _rec_kps[..., 0] = (_rec_kps[..., 0] + _cx1) * scale_factor + pad_x
                                _rec_kps[..., 1] = (_rec_kps[..., 1] + _cy1) * scale_factor + pad_y
                                _rec_boxes[:, 0] = (_rec_boxes[:, 0] + _cx1) * scale_factor + pad_x
                                _rec_boxes[:, 1] = (_rec_boxes[:, 1] + _cy1) * scale_factor + pad_y
                                _rec_boxes[:, 2] = (_rec_boxes[:, 2] + _cx1) * scale_factor + pad_x
                                _rec_boxes[:, 3] = (_rec_boxes[:, 3] + _cy1) * scale_factor + pad_y
                                # Inject recovered keypoints into the arrays
                                if _cow_keypoints_mapped is None:
                                    n_kps = _rec_kps.shape[1]
                                    _cow_keypoints_mapped = np.zeros((len(_cow_boxes_mapped), n_kps, 3))
                                _cow_keypoints_mapped[_ci] = _rec_kps[0]
                                _cow_boxes_mapped[_ci] = _rec_boxes[0]
                                _log(f"cow_selection: two-stage recovery SUCCESS for cow {_ci}")
                                break
                    except Exception as e:
                        _log(f"cow_selection: two-stage recovery failed: {e}")

        # Obtener bbox del animal para definir región de cabeza
        head_region = None
        animal_bbox = None
        img_height, img_width = resized_image.shape[:2]

        if _cow_detected and _cow_boxes_mapped is not None and len(_cow_boxes_mapped) > 0:
            animal_x1, animal_y1, animal_x2, animal_y2 = map(int, _cow_boxes_mapped[_ci])
            animal_bbox = (animal_x1, animal_y1, animal_x2, animal_y2)
            animal_height = animal_y2 - animal_y1

            head_y1 = animal_y1
            head_y2 = animal_y1 + int(animal_height * 0.5)
            head_x1 = max(0, animal_x1 - int((animal_x2 - animal_x1) * 0.2))
            head_x2 = min(img_width, animal_x2 + int((animal_x2 - animal_x1) * 0.2))

            head_roi_height = head_y2 - head_y1
            if head_roi_height < 200:
                additional_height = 200 - head_roi_height
                head_y2 = min(img_height, head_y2 + additional_height)
                _log(f"head_region: expanded height from {head_roi_height} to {head_y2 - head_y1}px (min 200px)")

            head_region = (head_x1, head_y1, head_x2, head_y2)
            _log(f"head_region: detected x1={head_x1} y1={head_y1} x2={head_x2} y2={head_y2} height={head_y2-head_y1}px")

        # Si no se detecta cabeza, usar región superior de la imagen como fallback
        if head_region is None:
            head_region = (0, 0, img_width, int(img_height * 0.4))
            _log(f"head_region: fallback to upper 40% of image (no animal bbox detected)")
        
        head_x1, head_y1, head_x2, head_y2 = head_region
        
        # PASO 2: Detectar ojos dentro de la región de cabeza (solo si scale_method no es 'poste')
        dist = None
        dist1 = None
        dist2 = None
        _bbox_fallback_used = False
        eyes_masks_count = 0
        eye_coords = []  # Lista para almacenar coordenadas de ojos detectados
        eye_centers = []  # Lista para almacenar centros de ojos para calcular distancia
        
        if scale_method != 'poste':
            # Recortar región de cabeza para buscar ojos (más eficiente y preciso)
            head_roi = resized_image[head_y1:head_y2, head_x1:head_x2]
            
            if head_roi.size == 0:
                _log(f"head_roi: empty, using full image")
                head_roi = resized_image
                head_offset_x, head_offset_y = 0, 0
                head_x1, head_y1, head_x2, head_y2 = 0, 0, img_width, img_height
            else:
                head_offset_x, head_offset_y = head_x1, head_y1
                head_roi_height, head_roi_width = head_roi.shape[:2]
                _log(f"head_roi: size={head_roi.shape} (height={head_roi_height}px width={head_roi_width}px) offset=({head_offset_x}, {head_offset_y})")
                
                # Si el ROI es muy pequeño, usar toda la imagen como fallback
                if head_roi_height < 150 or head_roi_width < 200:
                    _log(f"head_roi: too small ({head_roi_height}x{head_roi_width}), using full image as fallback")
                    head_roi = resized_image
                    head_offset_x, head_offset_y = 0, 0
                    head_x1, head_y1, head_x2, head_y2 = 0, 0, img_width, img_height
            
            # Detección de ojos en la región de cabeza (o imagen completa si ROI es muy pequeño)
            _log(f"eye_detection: using ROI size={head_roi.shape} conf={self.eye_conf:.4f} scale_method={scale_method}")
            results1 = self.eye_model(head_roi, save=False, conf=self.eye_conf, iou=self.iou_threshold)
            
            # Obtener nombres de clases del modelo para filtrar
            eye_model_classes = self.eye_model.names if hasattr(self.eye_model, 'names') else {}
            _log(f"eye_model_classes: {eye_model_classes}")
            
            # Procesar resultados de segmentación de instancias (ojos)
            total_detections_before_filter = 0
            for result in results1:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    total_detections_before_filter += len(masks)
                    eyes_masks_count += len(masks)
                    
                    _log(f"eye_detection: found {len(masks)} detections before filtering")
                    
                    for mask, box, score, cls in zip(masks, boxes, scores, classes):
                        # VALIDACIÓN 0: Filtrar por clase - solo aceptar "Eye", rechazar "Nose" u otras clases
                        class_name = eye_model_classes.get(int(cls), f"class_{int(cls)}")
                        _log(f"detection: class={class_name} score={score:.3f} bbox={box}")
                        
                        # Rechazar si es "Nose" u otras clases que no sean "Eye"
                        if "nose" in class_name.lower() or "Nose" in class_name:
                            _log(f"eye_rejected: class={class_name} (not an eye)")
                            continue
                        
                        # Solo aceptar si es explícitamente "Eye" o si no hay clases definidas (asumir que son ojos)
                        if eye_model_classes and "eye" not in class_name.lower() and "Eye" not in class_name:
                            _log(f"eye_rejected: class={class_name} (not Eye)")
                            continue
                        
                        # Mapear coordenadas del ROI de cabeza al frame completo
                        x1_roi, y1_roi, x2_roi, y2_roi = map(int, box)
                        x1 = head_offset_x + x1_roi
                        y1 = head_offset_y + y1_roi
                        x2 = head_offset_x + x2_roi
                        y2 = head_offset_y + y2_roi
                        
                        # Asegurar que las coordenadas estén dentro de la imagen
                        x1 = max(0, min(x1, img_width))
                        y1 = max(0, min(y1, img_height))
                        x2 = max(0, min(x2, img_width))
                        y2 = max(0, min(y2, img_height))
                        
                        mask_resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), 
                                                 interpolation=cv2.INTER_NEAREST)
                        mask_resized = mask_resized.astype(bool)
                        
                        # Calcular centro del ojo (más preciso que usar esquinas)
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # VALIDACIÓN 1: El ojo debe estar dentro de la región de cabeza detectada
                        # (ya está filtrado por el ROI, pero validamos por seguridad)
                        if not (head_x1 <= center_x <= head_x2 and head_y1 <= center_y <= head_y2):
                            _log(f"eye_rejected: center=({center_x:.1f}, {center_y:.1f}) fuera de head_region")
                            continue
                        
                        # VALIDACIÓN 2: El tamaño del bbox debe ser razonable para un ojo
                        # Los ojos son pequeños comparados con el tamaño total del animal
                        eye_width = x2 - x1
                        eye_height = y2 - y1
                        eye_area = eye_width * eye_height
                        img_area = img_width * img_height
                        eye_area_ratio = eye_area / img_area
                        
                        # Rechazar si el área es demasiado grande (probablemente no es un ojo)
                        if eye_area_ratio > 0.15:  # Más del 15% de la imagen es demasiado grande para un ojo
                            _log(f"eye_rejected: area_ratio={eye_area_ratio:.3f} > 0.15 (demasiado grande)")
                            continue
                        
                        # Si pasa las validaciones, agregar a la lista
                        eye_centers.append((center_x, center_y))
                        _log(f"eye_accepted: class={class_name} center=({center_x:.1f}, {center_y:.1f}) score={score:.3f}")
                        
                        # Guardar coordenadas de ojos si se solicita (solo los válidos)
                        if return_eye_coords:
                            eye_coords.append({
                                'bbox': [x1, y1, x2, y2],
                                'center': [center_x, center_y],
                                'score': float(score),
                                'class': int(cls),
                                'class_name': self.eye_model.names[int(cls)],
                                'validated': True  # Marcar como validado
                            })
                        
                        if visualize:
                            img_mask = np.zeros_like(img_rgb)
                            img_mask[mask_resized] = [229, 22, 122]  # Color naranja para máscara
                            img_rgb = cv2.addWeighted(img_rgb, 1.0, img_mask, 1.0, 1)
                            
                            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (229, 22, 122), 2)
                            
                            # Dibujar centro del ojo como punto
                            cv2.circle(img_rgb, (int(center_x), int(center_y)), 5, (255, 255, 0), -1)  # Amarillo para centro
                            
                            label = f'{self.eye_model.names[int(cls)]} {score:.2f}'
                            cv2.putText(img_rgb, label, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Log resumen de detecciones
            _log(f"eye_detection_summary: total_detections={total_detections_before_filter} after_filtering={len(eye_centers)} valid_eyes={len(eye_centers)}")
            
            # Calcular distancia entre ojos usando los centros (más preciso)
            if len(eye_centers) >= 2:
                # VALIDACIÓN 3: Los ojos deben estar aproximadamente a la misma altura (horizontalmente alineados)
                # Filtrar pares de ojos que estén demasiado separados verticalmente
                valid_eye_pairs = []
                for i in range(len(eye_centers)):
                    for j in range(i + 1, len(eye_centers)):
                        pt1 = eye_centers[i]
                        pt2 = eye_centers[j]
                        
                        # Calcular diferencia vertical entre ojos
                        vertical_diff = abs(pt1[1] - pt2[1])
                        horizontal_diff = abs(pt1[0] - pt2[0])
                        
                        # Los ojos deben estar aproximadamente a la misma altura
                        # La diferencia vertical debe ser menor que el 20% de la diferencia horizontal
                        if vertical_diff < horizontal_diff * 0.3:  # Los ojos están alineados horizontalmente
                            dist_candidate = self.euclidean(pt1, pt2)
                            
                            # VALIDACIÓN 4: La distancia entre ojos debe ser razonable
                            # Para una vaca, la distancia entre ojos está entre 5-15% del ancho de la imagen
                            min_dist = img_width * 0.05
                            max_dist = img_width * 0.25
                            
                            if min_dist <= dist_candidate <= max_dist:
                                valid_eye_pairs.append((pt1, pt2, dist_candidate))
                                _log(f"eye_pair_valid: dist={dist_candidate:.1f}px vertical_diff={vertical_diff:.1f} horizontal_diff={horizontal_diff:.1f}")
                            else:
                                _log(f"eye_pair_rejected: dist={dist_candidate:.1f}px fuera de rango [{min_dist:.1f}, {max_dist:.1f}]")
                        else:
                            _log(f"eye_pair_rejected: vertical_diff={vertical_diff:.1f} > horizontal_diff*0.3={horizontal_diff*0.3:.1f} (no alineados)")
                
                # Usar el par de ojos más válido (menor diferencia vertical)
                if valid_eye_pairs:
                    # Ordenar por diferencia vertical (menor es mejor)
                    valid_eye_pairs.sort(key=lambda x: abs(x[0][1] - x[1][1]))
                    pt1, pt2, dist = valid_eye_pairs[0]
                    
                    # Filtrar eye_coords para solo incluir los ojos del par válido
                    if return_eye_coords:
                        # Encontrar los ojos que corresponden al par válido
                        valid_eye_coords = []
                        for eye in eye_coords:
                            eye_center = eye['center']
                            # Verificar si este ojo es parte del par válido (con tolerancia de 5 píxeles)
                            if (abs(eye_center[0] - pt1[0]) < 5 and abs(eye_center[1] - pt1[1]) < 5) or \
                               (abs(eye_center[0] - pt2[0]) < 5 and abs(eye_center[1] - pt2[1]) < 5):
                                valid_eye_coords.append(eye)
                        eye_coords = valid_eye_coords  # Reemplazar con solo los válidos
                        _log(f"eye_coords filtered: {len(eye_coords)} valid eyes from {len(eye_coords) + (eyes_masks_count - len(eye_coords))} total")
                    
                    _log(f"eyes: masks={eyes_masks_count} valid_pairs={len(valid_eye_pairs)} dist_ref=ok dist_px={dist:.2f}")
                else:
                    # Si no hay pares válidos, limpiar eye_coords
                    if return_eye_coords:
                        eye_coords = []
                    _log(f"eyes: masks={eyes_masks_count} valid_pairs=0 dist_ref=missing (no hay pares válidos)")
            else:
                _log(f"eyes: masks={eyes_masks_count} dist_ref=missing (need 2+ eyes after filtering, got {len(eye_centers)})")
        else:
            _log(f"eye_detection: skipped (scale_method='poste', solo usando poste rojo)")
        
        # Procesar resultados de detección de puntos clave (usando datos ya mapeados a letterbox)
        keypoints_found = False
        keypoint_coords = []  # Almacenar coordenadas de keypoints usados para dist1 y dist2
        if _cow_keypoints_mapped is not None and _cow_boxes_mapped is not None:
            keypoints = _cow_keypoints_mapped
            boxes = _cow_boxes_mapped
            scores = _cow_scores
            classes = _cow_classes

            if len(keypoints) > 0 and _ci < len(keypoints) and len(keypoints[_ci]) >= 5:
                # Keypoint mapping (from cow.pt training):
                #   KP0: head/poll          KP5: leg/hoof
                #   KP1: pinbone (hip)      KP6: back/topline
                #   KP2: shoulderbone       KP7: belly midpoint
                #   KP3: girth bottom       KP8: back near shoulder
                #   KP4: girth top (withers)
                #
                # dist1 = KP1→KP2 = Body Length (pinbone → shoulderbone)
                # dist2 = KP3→KP4 = Girth vertical (bottom → top)

                # Each keypoint has [x, y, confidence]. Filter by confidence.
                MIN_KP_CONF = 0.3  # Minimum per-keypoint confidence to trust its position
                kp1_conf = float(keypoints[_ci][1][2])
                kp2_conf = float(keypoints[_ci][2][2])
                kp3_conf = float(keypoints[_ci][3][2])
                kp4_conf = float(keypoints[_ci][4][2])

                _log(f"keypoint_confs: KP1(pinbone)={kp1_conf:.3f} KP2(shoulder)={kp2_conf:.3f} KP3(girth_bot)={kp3_conf:.3f} KP4(girth_top)={kp4_conf:.3f} min_required={MIN_KP_CONF}")

                # Check that all 4 keypoints used for measurement have sufficient confidence
                low_conf_kps = []
                if kp1_conf < MIN_KP_CONF:
                    low_conf_kps.append(f"KP1(pinbone)={kp1_conf:.3f}")
                if kp2_conf < MIN_KP_CONF:
                    low_conf_kps.append(f"KP2(shoulder)={kp2_conf:.3f}")
                if kp3_conf < MIN_KP_CONF:
                    low_conf_kps.append(f"KP3(girth_bot)={kp3_conf:.3f}")
                if kp4_conf < MIN_KP_CONF:
                    low_conf_kps.append(f"KP4(girth_top)={kp4_conf:.3f}")

                if low_conf_kps:
                    _log(f"keypoints: REJECTED low confidence: {', '.join(low_conf_kps)}")
                    # Don't set keypoints_found — dist1/dist2 remain None
                else:
                    keypoints_found = True

                point1 = keypoints[_ci][1][0], keypoints[_ci][1][1]
                point2 = keypoints[_ci][2][0], keypoints[_ci][2][1]
                point3 = keypoints[_ci][3][0], keypoints[_ci][3][1]
                point4 = keypoints[_ci][4][0], keypoints[_ci][4][1]

                if keypoints_found:
                    dist1 = self.euclidean(point1, point2)
                    dist2 = self.euclidean(point3, point4)

                # Validar que los 4 keypoints estén dentro del animal_bbox (con margen 15%)
                if animal_bbox and keypoints_found:
                    ax1, ay1, ax2, ay2 = animal_bbox
                    margin_x = (ax2 - ax1) * 0.15
                    margin_y = (ay2 - ay1) * 0.15
                    for kp_name, kp_pt in [('pinbone', point1), ('shoulder', point2),
                                            ('girth_bot', point3), ('girth_top', point4)]:
                        if not (ax1 - margin_x <= kp_pt[0] <= ax2 + margin_x and
                                ay1 - margin_y <= kp_pt[1] <= ay2 + margin_y):
                            _log(f"keypoints: REJECTED {kp_name}=({kp_pt[0]:.0f},{kp_pt[1]:.0f}) outside bbox [{ax1},{ay1},{ax2},{ay2}] margin=({margin_x:.0f},{margin_y:.0f})")
                            keypoints_found = False
                            dist1 = None
                            dist2 = None
                            break

                # Guardar coordenadas de keypoints si se solicita
                if return_keypoint_coords:
                    keypoint_coords = [
                        {'point': 'KP1_pinbone', 'coords': [float(point1[0]), float(point1[1])], 'conf': kp1_conf, 'used_for': 'dist1'},
                        {'point': 'KP2_shoulder', 'coords': [float(point2[0]), float(point2[1])], 'conf': kp2_conf, 'used_for': 'dist1'},
                        {'point': 'KP3_girth_bot', 'coords': [float(point3[0]), float(point3[1])], 'conf': kp3_conf, 'used_for': 'dist2'},
                        {'point': 'KP4_girth_top', 'coords': [float(point4[0]), float(point4[1])], 'conf': kp4_conf, 'used_for': 'dist2'},
                        {'dist1_px': float(dist1) if dist1 else None, 'dist2_px': float(dist2) if dist2 else None, 'keypoints_accepted': keypoints_found}
                    ]

                if visualize:
                    # Draw ALL cow bboxes: selected cow in green, others in gray
                    for det_idx, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                        x1, y1, x2, y2 = map(int, box)
                        if det_idx == _ci:
                            # Selected cow: green bbox
                            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cls_name = self.cow_model.names.get(int(cls), 'Cow')
                            label = f'{cls_name} {score:.2f}'
                            cv2.putText(img_rgb, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        else:
                            # Other cows: gray bbox with index label
                            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (160, 160, 160), 1)
                            cv2.putText(img_rgb, f'cow {det_idx+1}', (x1, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

                    # Draw keypoints only for the selected cow
                    if _ci < len(keypoints):
                        keypoint = keypoints[_ci]
                        for kp in keypoint:
                            kp_x, kp_y = int(kp[0]), int(kp[1])
                            cv2.circle(img_rgb, (kp_x, kp_y), 3, (0, 0, 255), -1)

                    # Resaltar los keypoints usados para dist1 y dist2 (only when accepted)
                    if keypoints_found and dist1 is not None and dist2 is not None:
                        # point1 y point2 para dist1 (azul)
                        cv2.circle(img_rgb, (int(point1[0]), int(point1[1])), 8, (255, 0, 0), 2)
                        cv2.circle(img_rgb, (int(point2[0]), int(point2[1])), 8, (255, 0, 0), 2)
                        cv2.line(img_rgb, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (255, 0, 0), 2)

                        # point3 y point4 para dist2 (cyan)
                        cv2.circle(img_rgb, (int(point3[0]), int(point3[1])), 8, (255, 255, 0), 2)
                        cv2.circle(img_rgb, (int(point4[0]), int(point4[1])), 8, (255, 255, 0), 2)
                        cv2.line(img_rgb, (int(point3[0]), int(point3[1])), (int(point4[0]), int(point4[1])), (255, 255, 0), 2)

                        # Etiquetas para dist1 y dist2
                        mid1_x, mid1_y = int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2)
                        mid2_x, mid2_y = int((point3[0] + point4[0]) / 2), int((point3[1] + point4[1]) / 2)
                        cv2.putText(img_rgb, f'dist1={dist1:.1f}px', (mid1_x, mid1_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        cv2.putText(img_rgb, f'dist2={dist2:.1f}px', (mid2_x, mid2_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        _log(f"keypoints: found={keypoints_found} dist1={'ok' if dist1 else 'missing'} dist2={'ok' if dist2 else 'missing'}")

        # Fallback: estimate dist1/dist2 from COCO bounding box when keypoints unavailable
        if not keypoints_found and animal_bbox and _cow_keypoints_mapped is None:
            ax1, ay1, ax2, ay2 = animal_bbox
            bbox_w = ax2 - ax1
            bbox_h = ay2 - ay1
            BBOX_BODY_LENGTH_RATIO = 0.80
            BBOX_GIRTH_VERT_RATIO = 0.60
            dist1 = bbox_w * BBOX_BODY_LENGTH_RATIO
            dist2 = bbox_h * BBOX_GIRTH_VERT_RATIO
            keypoints_found = True
            _bbox_fallback_used = True
            _log(f"keypoints: BBOX_FALLBACK dist1={dist1:.1f}px (bbox_w={bbox_w:.0f}*0.80) "
                 f"dist2={dist2:.1f}px (bbox_h={bbox_h:.0f}*0.60)")

            # Visualize bbox-estimated measurements mimicking keypoint style
            if visualize:
                ax1_i, ay1_i, ax2_i, ay2_i = int(ax1), int(ay1), int(ax2), int(ay2)
                # Draw cow bbox in green (like cow.pt detection)
                cv2.rectangle(img_rgb, (ax1_i, ay1_i), (ax2_i, ay2_i), (0, 255, 0), 2)

                # dist1 (body length): horizontal line at ~35% from top (spine level)
                # Same blue color as keypoint dist1: (255, 0, 0)
                spine_y = int(ay1_i + bbox_h * 0.35)
                margin_x = int(bbox_w * (1 - BBOX_BODY_LENGTH_RATIO) / 2)
                p1 = (ax1_i + margin_x, spine_y)  # "pinbone" (left)
                p2 = (ax2_i - margin_x, spine_y)  # "shoulder" (right)
                cv2.circle(img_rgb, p1, 8, (255, 0, 0), 2)
                cv2.circle(img_rgb, p2, 8, (255, 0, 0), 2)
                cv2.line(img_rgb, p1, p2, (255, 0, 0), 2)
                mid1_x, mid1_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                cv2.putText(img_rgb, f'dist1={dist1:.1f}px', (mid1_x, mid1_y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                # dist2 (girth vertical): vertical line at ~55% from left (ribcage area)
                # Same cyan color as keypoint dist2: (255, 255, 0)
                girth_x = int(ax1_i + bbox_w * 0.55)
                margin_y = int(bbox_h * (1 - BBOX_GIRTH_VERT_RATIO) / 2)
                p3 = (girth_x, ay1_i + margin_y)   # "girth top" (withers)
                p4 = (girth_x, ay2_i - margin_y)   # "girth bottom"
                cv2.circle(img_rgb, p3, 8, (255, 255, 0), 2)
                cv2.circle(img_rgb, p4, 8, (255, 255, 0), 2)
                cv2.line(img_rgb, p3, p4, (255, 255, 0), 2)
                mid2_x, mid2_y = (p3[0] + p4[0]) // 2, (p3[1] + p4[1]) // 2
                cv2.putText(img_rgb, f'dist2={dist2:.1f}px', (mid2_x + 5, mid2_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Detectar postes (solo si scale_method no es 'eyes')
        # Skip post detection entirely when override_cm_per_px is provided (batch mode)
        # — the calibrated scale replaces everything, so detecting posts is wasted work
        postes_all = []
        poste_selected = None
        poste_candidates = []
        poste_rejected = []
        if override_cm_per_px is not None:
            _log(f"postes: SKIP detección (override_cm_per_px={override_cm_per_px:.5f} provisto, usando calibración directa)")
        elif scale_method != 'eyes' and self.use_postes_reference and self.depth_estimator:
            _log(f"postes: iniciando detección (scale_method={scale_method})")
            postes_all = self.depth_estimator.detect_postes_all(resized_image)
            _log(f"postes: detectados {len(postes_all)} postes por YOLO (letterbox)")

            # Fallback: si se detectaron <2 postes en letterbox, reintentar con imagen original
            if len(postes_all) < 2:
                _log(f"postes: reintentando detección en imagen original ({w_orig}x{h_orig})")
                postes_orig = self.depth_estimator.detect_postes_all(img)
                _log(f"postes: detectados {len(postes_orig)} postes en imagen original")
                if len(postes_orig) > len(postes_all):
                    # Mapear coordenadas de la imagen original al espacio letterbox
                    postes_mapped = []
                    for p in postes_orig:
                        ox1, oy1, ox2, oy2 = p['bbox']
                        mx1 = ox1 * scale_factor + pad_x
                        my1 = oy1 * scale_factor + pad_y
                        mx2 = ox2 * scale_factor + pad_x
                        my2 = oy2 * scale_factor + pad_y
                        mapped_p = dict(p)
                        mapped_p['bbox'] = [mx1, my1, mx2, my2]
                        postes_mapped.append(mapped_p)
                    postes_all = postes_mapped
                    _log(f"postes: usando {len(postes_all)} postes de imagen original (mapeados a letterbox)")
            if postes_all:
                for i, p in enumerate(postes_all):
                    _log(f"postes: [{i}] bbox={p['bbox']} score={p.get('score', 0):.3f} red_ratio={p.get('yellow_ratio', 0):.3f}")

            # Filter by user-selected indices BEFORE reordering
            # post_indices refer to the score-ordered list from scan_detections,
            # which is the same order as postes_all (detect_postes_all returns by score desc).
            # We must filter BEFORE _select_post_for_scale which reorders by height/distance.
            if post_indices is not None and len(postes_all) > 0:
                _log(f"postes: filtering by post_indices={post_indices} (from {len(postes_all)} detected)")
                filtered_all = [p for idx, p in enumerate(postes_all) if idx in post_indices]
                if filtered_all:
                    _log(f"postes: after filter: {len(filtered_all)} postes kept: {[p['bbox'] for p in filtered_all]}")
                    postes_all = filtered_all
                else:
                    _log(f"postes: WARNING - post_indices filter resulted in 0 postes, keeping all {len(postes_all)}")

            poste_selected, poste_candidates, poste_rejected = self._select_post_for_scale(
                postes_all, resized_image, animal_bbox=animal_bbox, band_tolerance=0.5
            )
            _log(f"postes: selección completada - selected={poste_selected is not None} candidates={len(poste_candidates)} rejected={len(poste_rejected)}")
            
            if poste_selected:
                measured_h = poste_selected.get('measured_height_px')
                _log(f"postes: selected bbox={poste_selected['bbox']} score={poste_selected.get('score'):.3f} red_ratio={poste_selected.get('yellow_ratio', 0):.3f} measured_height_px={measured_h}")
                if measured_h is None or measured_h <= 0:
                    _log(f"postes: WARNING - altura medida no disponible o inválida (measured_h={measured_h})")
            else:
                _log(f"postes: no suitable post selected - total_detected={len(postes_all)} candidates={len(poste_candidates)} rejected={len(poste_rejected)}")
                if postes_all:
                    _log(f"postes: todos los postes fueron rechazados durante la selección")
                else:
                    _log(f"postes: ningún poste detectado por YOLO o filtrado por color rojo")
        elif scale_method == 'eyes':
            _log(f"postes: detección omitida (scale_method='eyes', solo usando ojos)")
        elif not self.use_postes_reference:
            _log(f"postes: detección omitida (use_postes_reference=False)")
        elif not self.depth_estimator:
            _log(f"postes: detección omitida (depth_estimator no inicializado)")
        # Nota: ya no dependemos de sticker_model; DepthEstimator tiene fallback por color.

        # Visualización de postes, franja del animal y selección
        if visualize:
            if animal_bbox:
                ax1, ay1, ax2, ay2 = map(int, animal_bbox)
                animal_height = max(1, ay2 - ay1)
                band_top = max(0, int(ay1 - animal_height * 0.5))
                band_bottom = img_height - 1  # Sin límite inferior
                cv2.rectangle(img_rgb, (ax1, ay1), (ax2, ay2), (0, 255, 255), 2)
                cv2.line(img_rgb, (0, band_top), (img_width, band_top), (255, 255, 0), 2)

            # Flow nuevo: cuando el usuario seleccionó postes, todo el render de escala
            # se hace via RECTÁNGULO (cinta → piso). Cuando no hay selección, render legacy.
            measured_heights = []
            rect_params_for_details = None  # para retornar en details (lock reference)

            # Si hay referencia fijada, dibujar el rectángulo fijo (no re-detectar postes)
            if locked_reference:
                _p1 = locked_reference.get('post1', {})
                _p2 = locked_reference.get('post2', {})
                try:
                    _rcx1 = int(_p1.get('cx', 0))
                    _rcx2 = int(_p2.get('cx', 0))
                    _rtt1 = int(_p1.get('top_tape', 0))
                    _rtt2 = int(_p2.get('top_tape', 0))
                    _rfl1 = int(_p1.get('floor', 0))
                    _rfl2 = int(_p2.get('floor', 0))
                    _rtp1 = float(_p1.get('tape_px', 0))
                    _rtp2 = float(_p2.get('tape_px', 0))
                    if _rcx1 > _rcx2:
                        _rcx1, _rcx2 = _rcx2, _rcx1
                        _rtt1, _rtt2 = _rtt2, _rtt1
                        _rfl1, _rfl2 = _rfl2, _rfl1
                        _rtp1, _rtp2 = _rtp2, _rtp1
                    rect_color_fix = (0, 255, 255)
                    for a, b in [((_rcx1, _rtt1), (_rcx2, _rtt2)),
                                 ((_rcx1, _rfl1), (_rcx2, _rfl2)),
                                 ((_rcx1, _rtt1), (_rcx1, _rfl1)),
                                 ((_rcx2, _rtt2), (_rcx2, _rfl2))]:
                        cv2.line(img_rgb, a, b, rect_color_fix, 1)
                    # Marca "REFERENCIA FIJA" en esquina
                    cv2.putText(img_rgb, 'REF FIJA', (_rcx1 + 4, _rtt1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color_fix, 1)
                    # Etiquetas cm por lado
                    for cxp, ttp, flp, tpp, side in [
                        (_rcx1, _rtt1, _rfl1, _rtp1, 'L'),
                        (_rcx2, _rtt2, _rfl2, _rtp2, 'R'),
                    ]:
                        if tpp > 0:
                            sc = 110.0 / tpp
                            h_cm = (flp - ttp) * sc
                            txt = f"{h_cm:.0f}cm"
                            ymid = (ttp + flp) // 2
                            x_txt = cxp + 12 if side == 'L' else cxp - 100
                            cv2.putText(img_rgb, txt, (x_txt, ymid),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color_fix, 2)
                except Exception as _e:
                    _log(f"locked_ref render failed: {_e}")

            # Si ya hay referencia fijada, NO dibujamos render de postes (ni legacy ni user-selected)
            # El rectángulo fijo ya se dibujó arriba con locked_reference.
            if locked_reference:
                user_selected_2 = False
                _log("postes: locked_reference activa - skipping post render")
                # Skip todo el bloque de render
                continue_render = False
            else:
                continue_render = True
                user_selected_2 = (post_indices is not None
                                   and len([p for p in poste_candidates
                                            if p.get('measured_height_px') and p.get('measured_height_px') > 0]) >= 2)

            _log(f"RECT_CHECK: post_indices={post_indices} candidates={len(poste_candidates)} "
                 f"user_selected_2={user_selected_2}")

            if not continue_render:
                pass  # locked_reference activa, saltamos todo el render de postes
            elif user_selected_2:
                # === RENDER NUEVO: sólo el rectángulo ===
                _valid_cands = [p for p in poste_candidates
                                if p.get('measured_height_px') and p.get('measured_height_px') > 0]
                for p in _valid_cands:
                    measured_heights.append(p['measured_height_px'])

                sel_post_infos = []
                for _p in _valid_cands[:2]:
                    # Bbox magenta mínimo para contexto visual
                    x1, y1, x2, y2 = map(int, _p['bbox'])
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 255), 1)
                    info = _draw_tape_and_floor_on(img_rgb, _p['bbox'])
                    _log(f"RECT_CHECK: _draw_tape_and_floor_on bbox={_p['bbox']} -> "
                         f"{'OK' if info else 'None (no tape detected)'}")
                    if info is not None:
                        sel_post_infos.append(info)

                if len(sel_post_infos) == 2:
                    p1, p2 = sel_post_infos[0], sel_post_infos[1]
                    if p1['cx'] > p2['cx']:
                        p1, p2 = p2, p1
                    # Guardar para exponer en details (lock reference)
                    rect_params_for_details = {
                        'post1': {'cx': int(p1['cx']), 'top_tape': int(p1['top_tape']),
                                  'floor': int(p1['floor']), 'tape_px': int(p1['tape_px'])},
                        'post2': {'cx': int(p2['cx']), 'top_tape': int(p2['top_tape']),
                                  'floor': int(p2['floor']), 'tape_px': int(p2['tape_px'])},
                    }
                    rect_color = (0, 255, 255)  # yellow en RGB
                    pts_rect = [
                        ((p1['cx'], p1['top_tape']), (p2['cx'], p2['top_tape'])),
                        ((p1['cx'], p1['floor']),    (p2['cx'], p2['floor'])),
                        ((p1['cx'], p1['top_tape']), (p1['cx'], p1['floor'])),
                        ((p2['cx'], p2['top_tape']), (p2['cx'], p2['floor'])),
                    ]
                    for a, b in pts_rect:
                        cv2.line(img_rgb, a, b, rect_color, 1)
                    for p, side in [(p1, 'L'), (p2, 'R')]:
                        if p['tape_px'] > 0:
                            scale = 110.0 / p['tape_px']
                            h_cm = (p['floor'] - p['top_tape']) * scale
                            txt = f"{h_cm:.0f}cm"
                            ymid = (p['top_tape'] + p['floor']) // 2
                            x_txt = p['cx'] + 12 if side == 'L' else p['cx'] - 100
                            cv2.putText(img_rgb, txt, (x_txt, ymid),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_color, 2)
                    _log(f"RECT_CHECK: rectangle drawn")
                else:
                    _log(f"RECT_CHECK: rectangle NOT drawn (need 2 post_infos, got {len(sel_post_infos)})")
            else:
                # === RENDER LEGACY: sólo cuando el usuario NO seleccionó 2 postes ===
                for idx_c, p in enumerate(poste_candidates):
                    x1, y1, x2, y2 = map(int, p['bbox'])
                    if animal_bbox and y2 < band_top:
                        continue
                    measured_h = p.get('measured_height_px')
                    if measured_h and measured_h > 0:
                        measured_heights.append(measured_h)
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        score = p.get('score', 0)
                        red_ratio = p.get('yellow_ratio', 0)
                        cv2.putText(img_rgb, f'POSTE {idx_c+1} (score:{score:.2f} rojo:{red_ratio:.2f})',
                                   (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 200, 0), 3)
                        label_ref = f'REF {idx_c+1}: {measured_h:.1f}px = 110cm'
                        (tw_r, th_r), _ = cv2.getTextSize(label_ref, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(img_rgb, (x1, max(0, y1 - th_r - 10)), (x1 + tw_r + 4, max(0, y1 - 2)), (0, 0, 0), -1)
                        cv2.putText(img_rgb, label_ref, (x1 + 2, max(th_r + 2, y1 - 4)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                        cx = int((x1 + x2) / 2)
                        line_y_bottom = y2
                        line_y_top = max(y1, y2 - int(measured_h))
                        cv2.line(img_rgb, (cx, line_y_top), (cx, line_y_bottom), (255, 255, 0), 3)
                        cv2.line(img_rgb, (cx - 8, line_y_top), (cx + 8, line_y_top), (255, 255, 0), 3)
                        cv2.line(img_rgb, (cx - 8, line_y_bottom), (cx + 8, line_y_bottom), (255, 255, 0), 3)
                        cv2.circle(img_rgb, (cx, line_y_top), 5, (255, 255, 0), -1)
                        cv2.circle(img_rgb, (cx, line_y_bottom), 5, (255, 255, 0), -1)
                        mid_post_y = (line_y_top + line_y_bottom) // 2
                        cv2.putText(img_rgb, f'{measured_h:.1f}px', (cx + 8, mid_post_y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Visualización antigua (línea naranja "SCALE@cow") REEMPLAZADA por el
            # rectángulo completo arriba (cinta → piso).

            # Mostrar promedio si hay 2+ postes visibles
            if len(measured_heights) >= 2:
                avg_h = sum(measured_heights) / len(measured_heights)
                avg_text = f'PROMEDIO: {avg_h:.1f}px = 110cm'
                (tw_a, th_a), _ = cv2.getTextSize(avg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                avg_x = img_width - tw_a - 10
                avg_y = 25
                cv2.rectangle(img_rgb, (avg_x - 4, avg_y - th_a - 6), (avg_x + tw_a + 4, avg_y + 6), (0, 0, 0), -1)
                cv2.putText(img_rgb, avg_text, (avg_x, avg_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calcular escala desde postes o usar override
        escala_postes = None
        _scale_direct_from_postes = False

        # ── Referencia fijada: computar cm/px usando el CRUCE GEOMÉTRICO
        #    del borde inferior del bbox con la línea inclinada del piso
        #    (no con la posición horizontal cow_cx — eso daba extrapolación errónea)
        if locked_reference and animal_bbox:
            _p1 = locked_reference.get('post1', {})
            _p2 = locked_reference.get('post2', {})
            cx1 = float(_p1.get('cx', 0))
            cx2 = float(_p2.get('cx', 0))
            floor1 = float(_p1.get('floor', 0))
            floor2 = float(_p2.get('floor', 0))
            tape_px_1 = float(_p1.get('tape_px', 0))
            tape_px_2 = float(_p2.get('tape_px', 0))
            if cx1 > cx2:
                cx1, cx2 = cx2, cx1
                floor1, floor2 = floor2, floor1
                tape_px_1, tape_px_2 = tape_px_2, tape_px_1
            ax1, ay1, ax2, ay2 = animal_bbox
            cow_cx = (ax1 + ax2) / 2.0
            if cx2 != cx1 and tape_px_1 > 0 and tape_px_2 > 0:
                # Misma lógica que /detect_cow_fast: cruce del segmento inferior del bbox
                # con el segmento del piso del rectángulo, para interpolar escala.
                X_lo_ = max(ax1, cx1)
                X_hi_ = min(ax2, cx2)
                t = None
                if X_lo_ <= X_hi_ and abs(floor2 - floor1) >= 0.5:
                    fy_lo = floor1 + (X_lo_ - cx1) / (cx2 - cx1) * (floor2 - floor1)
                    fy_hi = floor1 + (X_hi_ - cx1) / (cx2 - cx1) * (floor2 - floor1)
                    d_lo = fy_lo - ay2
                    d_hi = fy_hi - ay2
                    if d_lo * d_hi <= 1e-6 and abs(d_lo - d_hi) > 1e-6:
                        alpha = d_lo / (d_lo - d_hi)
                        x_cross = X_lo_ + alpha * (X_hi_ - X_lo_)
                        t = (x_cross - cx1) / (cx2 - cx1)
                        t = max(0.0, min(1.0, t))
                if t is None:
                    # Fallback: posición horizontal (clamped)
                    p_x = (cow_cx - cx1) / (cx2 - cx1)
                    t = max(0.0, min(1.0, p_x))
                scale_1 = 110.0 / tape_px_1
                scale_2 = 110.0 / tape_px_2
                escala_postes = (1 - t) * scale_1 + t * scale_2
                _scale_direct_from_postes = True
                _log(f"postes: LOCKED_REF escala={escala_postes:.5f} cm/px "
                     f"(cruce t={t:.3f}, cow_cx={cow_cx:.0f}, bbox_y2={ay2:.0f}, "
                     f"floor1={floor1:.0f}, floor2={floor2:.0f}, "
                     f"tape1_px={tape_px_1:.1f}, tape2_px={tape_px_2:.1f})")
            else:
                _log(f"postes: LOCKED_REF INVALID (cx1={cx1}, cx2={cx2}, tape1={tape_px_1}, tape2={tape_px_2})")

        if escala_postes is not None:
            # Ya calculada por locked_reference, saltar demás ramas
            pass
        elif override_cm_per_px is not None:
            # Calibración externa provista — usar directamente, sin detección de postes ni depth
            escala_postes = override_cm_per_px
            _log(f"postes: usando override_cm_per_px={override_cm_per_px:.5f} (calibración directa, skip depth)")
        elif post_indices is not None and len(poste_candidates) > 0:
            # User explicitly selected cintas — compute scale from their heights.
            # Skip depth estimator (which re-detects its own posts and ignores user selection).
            _valid = [c for c in poste_candidates
                      if c.get('measured_height_px') and c.get('measured_height_px') > 0]
            if len(_valid) == 2 and animal_bbox:
                # Intersección de la línea de escala (entre topes de los 2 postes)
                # con el borde superior del bbox de la vaca (y = ay1).
                # Se permite extrapolación si el tope del bbox está fuera del rango vertical del segmento.
                _posts_xh = []
                for _p in _valid:
                    _x1p, _y1p, _x2p, _y2p = _p['bbox']
                    _mhp = float(_p['measured_height_px'])
                    _top_yp = float(_y2p) - _mhp
                    _posts_xh.append(((_x1p + _x2p) / 2.0, _top_yp, _mhp))
                _posts_xh.sort(key=lambda v: v[0])
                (_cx1, _py1s, _h1), (_cx2, _py2s, _h2) = _posts_xh
                _ax1, _ay1, _ax2, _ay2 = animal_bbox
                if _py2s != _py1s:
                    _sample_x = _cx1 + (_cx2 - _cx1) * (_ay1 - _py1s) / (_py2s - _py1s)
                else:
                    _sample_x = (_cx1 + _cx2) / 2.0
                if _cx2 != _cx1:
                    _t = (_sample_x - _cx1) / (_cx2 - _cx1)  # sin clamp: permite extrapolación
                else:
                    _t = 0.5
                _h_at_cow = _h1 + _t * (_h2 - _h1)
                if _h_at_cow > 0:
                    escala_postes = self.poste_height_cm / _h_at_cow
                    _scale_direct_from_postes = True
                    _log(f"postes: USER_SELECTED_INTERP escala={escala_postes:.4f} cm/px "
                         f"(p1=(x={_cx1:.0f},py={_py1s:.0f},h={_h1:.1f}), "
                         f"p2=(x={_cx2:.0f},py={_py2s:.0f},h={_h2:.1f}), "
                         f"bbox_top_y={_ay1:.0f}, sample_x={_sample_x:.1f}, "
                         f"t={_t:.3f}, h_interp={_h_at_cow:.1f}px)")
                else:
                    _heights = [c['measured_height_px'] for c in _valid]
                    _avg_h = sum(_heights) / len(_heights)
                    escala_postes = self.poste_height_cm / _avg_h
                    _scale_direct_from_postes = True
                    _log(f"postes: USER_SELECTED_INTERP_INVALID (h_interp={_h_at_cow:.1f}<=0) "
                         f"→ fallback AVG escala={escala_postes:.4f} cm/px")
            elif _valid:
                _heights = [c['measured_height_px'] for c in _valid]
                _avg_h = sum(_heights) / len(_heights)
                escala_postes = self.poste_height_cm / _avg_h
                _scale_direct_from_postes = True
                _log(f"postes: USER_SELECTED_AVG escala={escala_postes:.4f} cm/px "
                     f"(poste_height={self.poste_height_cm}cm / avg_measured={_avg_h:.1f}px "
                     f"from {len(_heights)} cintas: {[f'{h:.0f}px' for h in _heights]})")
            else:
                _log(f"postes: WARNING - user-selected cintas have no valid heights")
        elif self.use_postes_reference and self.depth_estimator:
            _log(f"postes: calculando escala desde postes seleccionados")

            _scale_direct_from_postes = False

            # Solo promediar si hay más de 1 poste seleccionado, si no usar el único disponible
            if poste_candidates:
                _heights = [c.get('measured_height_px') for c in poste_candidates
                            if c.get('measured_height_px') and c.get('measured_height_px') > 0]
                _poste_cm = self.poste_height_cm
                if len(_heights) >= 2:
                    _avg_h = sum(_heights) / len(_heights)
                    escala_postes = _poste_cm / _avg_h
                    _log(f"postes: AVG_POSTS_FALLBACK escala={escala_postes:.4f} cm/px "
                         f"(poste_height={_poste_cm}cm / avg_measured={_avg_h:.1f}px "
                         f"from {len(_heights)} postes: {[f'{h:.0f}px' for h in _heights]})")
                elif len(_heights) == 1:
                    escala_postes = _poste_cm / _heights[0]
                    _log(f"postes: SINGLE_POST_FALLBACK escala={escala_postes:.4f} cm/px "
                         f"(poste_height={_poste_cm}cm / measured={_heights[0]:.1f}px)")
            elif escala_postes is None and poste_selected:
                _measured_h = poste_selected.get('measured_height_px')
                if _measured_h and _measured_h > 0:
                    _poste_cm = self.poste_height_cm
                    escala_postes = _poste_cm / _measured_h
                    _log(f"postes: SINGLE_POST_FALLBACK escala={escala_postes:.4f} cm/px "
                         f"(poste_height={_poste_cm}cm / measured={_measured_h:.1f}px)")
        else:
            if not self.use_postes_reference:
                _log(f"postes: deshabilitado (use_postes_reference=False)")
            elif not self.depth_estimator:
                _log(f"postes: no disponible (depth_estimator no inicializado)")

        # Corrección de escala por profundidad usando Depth Anything V2
        # (skip when using calibrated override — the calibration is already correct)
        _depth_correction_factor = 1.0
        if escala_postes is not None and override_cm_per_px is None and self.depth_pipe is not None and animal_bbox and poste_candidates:
            _depth_correction_factor = self._get_depth_scale_correction(
                resized_image, animal_bbox, poste_candidates, _log
            )
            if abs(_depth_correction_factor - 1.0) > 0.01:
                _escala_before = escala_postes
                escala_postes *= _depth_correction_factor
                _log(f"depth_correction: APPLIED factor={_depth_correction_factor:.4f} "
                     f"escala {_escala_before:.4f} → {escala_postes:.4f} cm/px")
            else:
                _log(f"depth_correction: factor={_depth_correction_factor:.4f} ~1.0, no adjustment needed")

        # Pre-compute available cm_per_px from VALID references only
        # (used for calibration in video — only trust 2-post or eye-based scale)
        _available_cm_per_px = None
        if escala_postes and self.use_postes_reference:
            _available_cm_per_px = escala_postes
        if _available_cm_per_px is None and dist:
            _available_cm_per_px = 20.0 / dist

        # Calcular peso si tenemos todas las medidas necesarias
        if dist1 and dist2:
            lb = 0.45359237  # Conversión de libras a kg
            
            # Usar el método de escala seleccionado por el usuario
            # Prioridad según scale_method:
            # - 'eyes': Solo ojos
            # - 'poste': Solo poste rojo
            # - 'both': Ojos primero, luego poste como fallback
            if scale_method == 'eyes':
                # Solo usar ojos
                if dist:
                    # Distancia inter-ocular real de ganado adulto (~18-25 cm).
                    # Valor conservador para Holando/cruza; ajustar según raza.
                    x = 20  # cm — distancia entre ojos asumida
                    dist1cm = (x * dist1) / dist
                    dist2cm = (x * dist2) / dist
                    escala_cm_per_px = x / dist
                    _log(f"weight_method=eyes dist_ref={dist:.2f}px eye_cm={x} escala={escala_cm_per_px:.6f}cm/px dist1cm={dist1cm:.2f} dist2cm={dist2cm:.2f}")
                else:
                    dist1cm = None
                    dist2cm = None
                    _log(f"weight_method=eyes: ojos no detectados, no se puede calcular peso")
            elif scale_method == 'poste':
                # Usar postes rojos (preferir 2 postes, fallback a 1 poste)
                _log(f"weight_method=poste: verificando disponibilidad - escala_postes={escala_postes is not None} use_postes_reference={self.use_postes_reference} direct_2post={_scale_direct_from_postes}")
                if escala_postes and self.use_postes_reference:
                    dist1cm = dist1 * escala_postes
                    dist2cm = dist2 * escala_postes
                    _scale_source = "2_postes" if _scale_direct_from_postes else "1_poste_fallback"
                    _log(f"weight_method=poste escala={escala_postes:.4f}cm/px source={_scale_source}")
                else:
                    dist1cm = None
                    dist2cm = None
                    if not self.use_postes_reference:
                        _log(f"weight_method=poste: use_postes_reference=False")
                    else:
                        _log(f"weight_method=poste: no se detectaron postes para calcular escala")
            else:  # scale_method == 'both'
                # Usar ojos primero, luego postes dobles como fallback
                # NOTA: Se requieren AMBOS postes visibles con la vaca entre ellos
                dist1cm = None
                dist2cm = None
                if dist:
                    # Método tradicional: usar ojos como referencia (preferido)
                    # Distancia inter-ocular real de ganado adulto (~18-25 cm).
                    x = 20  # cm — distancia entre ojos asumida
                    dist1cm = (x * dist1) / dist
                    dist2cm = (x * dist2) / dist
                    escala_cm_per_px = x / dist  # Escala: cm por píxel
                    _log(f"weight_method=eyes dist_ref={dist:.2f}px eye_cm={x} escala={escala_cm_per_px:.6f}cm/px dist1cm={dist1cm:.2f} dist2cm={dist2cm:.2f} (preferido)")

                    # Validar que las medidas sean razonables
                    if dist1cm < 20 or dist2cm < 20:
                        _log(f"WARNING: Medidas muy pequeñas! dist1cm={dist1cm:.2f} dist2cm={dist2cm:.2f} - posible error en escala")
                    if dist1cm > 300 or dist2cm > 300:
                        _log(f"WARNING: Medidas muy grandes! dist1cm={dist1cm:.2f} dist2cm={dist2cm:.2f} - posible error en escala")
                elif escala_postes and self.use_postes_reference:
                    # Fallback: postes dobles (ambos visibles, vaca entre ellos)
                    dist1cm = dist1 * escala_postes
                    dist2cm = dist2 * escala_postes
                    _log(f"weight_method=postes_dobles escala={escala_postes:.4f} (fallback, ojos no detectados, ambos postes visibles)")
                else:
                    # No hay referencia de escala válida
                    if not dist and not escala_postes:
                        _log(f"weight=missing: no hay referencia de escala (ojos no detectados, postes: se requieren ambos visibles con vaca entre ellos)")
                    elif not dist and not self.use_postes_reference:
                        _log(f"weight=missing: ojos no detectados y postes deshabilitados")
            
            if dist1cm is None or dist2cm is None:
                _no_scale_details = {
                    'missing_points': {
                        'eyes': not bool(dist) if scale_method != 'poste' else False,
                        'dist1': not bool(dist1),
                        'dist2': not bool(dist2),
                        'scale_reference': True,
                    },
                    'message': 'No se encontró referencia de escala',
                    'cm_per_px': _available_cm_per_px,
                    'animal_bbox_height_px': float(animal_bbox[3] - animal_bbox[1]) if animal_bbox else None,
                    'dist1_px': float(dist1) if dist1 else None,
                    'dist2_px': float(dist2) if dist2 else None,
                    'scale_method_selected': scale_method,
                    'has_eyes': bool(dist),
                    'has_dist1': bool(dist1),
                    'has_dist2': bool(dist2),
                    'has_scale_reference': False,
                    'poste_selected': poste_selected['bbox'] if poste_selected else None,
                    'postes_heights_px': [c.get('measured_height_px') for c in poste_candidates if c.get('measured_height_px') and c.get('measured_height_px') > 0] if 'poste_candidates' in locals() else [],
                    'keypoints_found': True,
                    'cow_score': float(_cow_scores[_ci]) if _cow_scores is not None and _ci < len(_cow_scores) else None,
                }
                if visualize:
                    if return_eye_coords and return_keypoint_coords:
                        return img_rgb, None, eye_coords, keypoint_coords, _no_scale_details
                    elif return_eye_coords:
                        return img_rgb, None, eye_coords, keypoint_coords if return_keypoint_coords else [], _no_scale_details
                    elif return_keypoint_coords:
                        return img_rgb, None, [], keypoint_coords, _no_scale_details
                    else:
                        return img_rgb, None, _no_scale_details
                else:
                    if return_eye_coords and return_keypoint_coords:
                        return None, eye_coords, keypoint_coords, _no_scale_details
                    elif return_eye_coords:
                        return None, eye_coords, keypoint_coords if return_keypoint_coords else [], _no_scale_details
                    elif return_keypoint_coords:
                        return None, [], keypoint_coords, _no_scale_details
                    else:
                        return None, _no_scale_details

            # ── Schaeffer-based weight formula ──
            # Standard Schaeffer: Weight_lbs = (HG^2 × BL) / 300
            #   HG = Heart Girth (full circumference around body behind front legs)
            #   BL = Body Length  (shoulder point to pin bone)
            #
            # With monocular depth (Depth Anything V2):
            #   We estimate the full elliptical circumference of the girth zone,
            #   then use the standard Schaeffer formula:
            #   W_kg = (HG_cm^2 × BL_cm) / 10838
            #   where 10838 = 300 × 2.54^3 / 0.4536
            #
            # Fallback (without depth):
            #   dist1cm = Body Length  (KP1 pinbone → KP2 shoulderbone)
            #   dist2cm = Girth vertical diameter (KP3 girth bottom → KP4 girth top)
            #   Formula: Weight_kg = (BL × GirthVert^2 × lb) / 300

            # Unificar cm_per_px desde la escala disponible
            cm_per_px = dist1cm / dist1 if dist1 and dist1 > 0 else None

            # Visualización: línea de altura del animal (muestra la escala aplicada al animal)
            if visualize and animal_bbox and cm_per_px is not None and cm_per_px > 0:
                ax1_h, ay1_h, ax2_h, ay2_h = map(int, animal_bbox)
                animal_h_px = ay2_h - ay1_h
                animal_h_cm = animal_h_px * cm_per_px
                # Línea vertical a la derecha del bbox del animal
                line_x = min(ax2_h + 25, img_width - 30)
                # Línea principal (verde brillante)
                cv2.line(img_rgb, (line_x, ay1_h), (line_x, ay2_h), (0, 220, 0), 3)
                # Ticks horizontales en los extremos
                cv2.line(img_rgb, (line_x - 12, ay1_h), (line_x + 12, ay1_h), (0, 220, 0), 3)
                cv2.line(img_rgb, (line_x - 12, ay2_h), (line_x + 12, ay2_h), (0, 220, 0), 3)
                # Flechas (triángulos) en los extremos
                cv2.fillPoly(img_rgb, [np.array([[line_x, ay1_h], [line_x - 6, ay1_h + 10], [line_x + 6, ay1_h + 10]])], (0, 220, 0))
                cv2.fillPoly(img_rgb, [np.array([[line_x, ay2_h], [line_x - 6, ay2_h - 10], [line_x + 6, ay2_h - 10]])], (0, 220, 0))
                # Etiqueta con altura en cm
                mid_y_h = (ay1_h + ay2_h) // 2
                # Fondo semitransparente para el texto
                label_text = f'Altura: {animal_h_cm:.1f}cm'
                label_text2 = f'({animal_h_px}px)'
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_rgb, (line_x + 14, mid_y_h - th - 5), (line_x + 18 + tw, mid_y_h + 25), (0, 0, 0), -1)
                cv2.putText(img_rgb, label_text, (line_x + 16, mid_y_h),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img_rgb, label_text2, (line_x + 16, mid_y_h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
                _log(f"viz: animal_height_cm={animal_h_cm:.1f} ({animal_h_px}px * {cm_per_px:.5f}cm/px)")

            # Visualización: indicador de corrección por profundidad
            if visualize and abs(_depth_correction_factor - 1.0) > 0.01:
                _dc_text = f'Depth corr: {_depth_correction_factor:.2f}x'
                _dc_color = (0, 200, 255)  # naranja
                (tw_dc, th_dc), _ = cv2.getTextSize(_dc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                _dc_x = img_width - tw_dc - 10
                _dc_y = 50
                cv2.rectangle(img_rgb, (_dc_x - 4, _dc_y - th_dc - 4), (_dc_x + tw_dc + 4, _dc_y + 4), (0, 0, 0), -1)
                cv2.putText(img_rgb, _dc_text, (_dc_x, _dc_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, _dc_color, 1)

            # Intentar circunferencia con profundidad monocular
            girth_circumference_cm = None
            _girth_left_edge = None
            _girth_right_edge = None
            if self.depth_pipe is not None and cm_per_px is not None:
                girth_result = self._estimate_girth_circumference(
                    resized_image, point3, point4, cm_per_px, _log
                )
                if girth_result is not None:
                    girth_circumference_cm, _girth_left_edge, _girth_right_edge = girth_result

            if girth_circumference_cm is not None:
                # Fórmula Schaeffer estándar en métrico:
                # W_kg = (HG_cm² × BL_cm) / 10838
                # donde 10838 = 300 × 2.54³ / 0.4536
                raw_weight = (girth_circumference_cm ** 2 * dist1cm) / 10838
                _log(f"weight_formula=schaeffer_standard HG={girth_circumference_cm:.2f}cm BL={dist1cm:.2f}cm divisor=10838")
            else:
                # Fallback: fórmula actual con diámetro vertical
                raw_weight = (dist1cm * dist2cm * dist2cm * lb) / 300
                _log(f"weight_formula=fallback_vertical BL={dist1cm:.2f}cm GirthVert={dist2cm:.2f}cm divisor=300")

            # Visualización: línea horizontal verde mostrando ancho detectado por profundidad
            if girth_circumference_cm is not None and _girth_left_edge is not None and visualize:
                girth_mid_y = int((point3[1] + point4[1]) / 2)
                cv2.line(img_rgb, (_girth_left_edge, girth_mid_y), (_girth_right_edge, girth_mid_y), (0, 255, 0), 2)
                cv2.putText(img_rgb, f'HG={girth_circumference_cm:.0f}cm',
                           (_girth_right_edge + 5, girth_mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            multiplier = get_weight_multiplier(breed, category, age_range)
            weight = raw_weight * multiplier
            _bbox_approx_tag = " (bbox_approx)" if _bbox_fallback_used else ""
            _depth_tag = f" depth_corr={_depth_correction_factor:.3f}" if abs(_depth_correction_factor - 1.0) > 0.01 else ""
            _log(f"weight=ok{_bbox_approx_tag}{_depth_tag} value={weight:.4f}kg raw={raw_weight:.4f}kg multiplier={multiplier:.4f} breed={breed} category={category} age={age_range} BL(dist1cm)={dist1cm:.2f} GirthVert(dist2cm)={dist2cm:.2f} girth_circ={'%.2f' % girth_circumference_cm if girth_circumference_cm else 'N/A'}cm elapsed_ms={(time.time()-t0)*1000:.1f}")
            
            # Determinar qué método de escala se usó
            scale_method_used = None
            if scale_method == 'eyes' and bool(dist):
                scale_method_used = 'eyes'
            elif scale_method == 'poste':
                if escala_postes is not None:
                    scale_method_used = 'postes_dobles'
                elif poste_selected is not None:
                    scale_method_used = 'poste'
            elif scale_method == 'both':
                if bool(dist):
                    scale_method_used = 'eyes'
                elif poste_selected is not None:
                    scale_method_used = 'poste'
                elif escala_postes is not None:
                    scale_method_used = 'postes_dobles'
            
            # Crear mensaje descriptivo según el método usado
            if scale_method_used == 'eyes':
                scale_message = 'Peso calculado usando ojos como referencia de escala'
            elif scale_method_used == 'poste':
                scale_message = 'Peso calculado usando poste rojo (altura) como referencia de escala'
            elif scale_method_used == 'postes_dobles':
                scale_message = 'Peso calculado usando postes dobles como referencia de escala'
            else:
                scale_message = 'Peso calculado con referencia válida'
            
            # Calcular cm_per_px que se usó para esta estimación
            _cm_per_px_used = None
            if dist1cm and dist1:
                _cm_per_px_used = dist1cm / dist1

            # Altura del bbox del animal en px (para calibración en video)
            _animal_bbox_height_px = float(animal_bbox[3] - animal_bbox[1]) if animal_bbox else None

            # Crear objeto de éxito (todos los puntos detectados)
            success_details = {
                'missing_points': {
                    'eyes': not bool(dist) if scale_method != 'poste' else False,  # Solo relevante si no es 'poste'
                    'dist1': not bool(dist1),
                    'dist2': not bool(dist2),
                    'scale_reference': not bool(dist) and not (poste_selected is not None or (escala_postes is not None and self.use_postes_reference))
                },
                'message': scale_message,
                'scale_method_used': scale_method_used,
                'scale_method_selected': scale_method,
                'has_eyes': bool(dist),
                'has_dist1': bool(dist1),
                'has_dist2': bool(dist2),
                'has_scale_reference': bool(dist) or (poste_selected is not None) or (escala_postes is not None and self.use_postes_reference),
                'eyes_count': len(eye_centers) if 'eye_centers' in locals() else 0,
                'keypoints_found': True,
                'poste_selected': poste_selected['bbox'] if poste_selected else None,
                'postes_detected': len(postes_all) if 'postes_all' in locals() else 0,
                'scale_from_postes': _scale_direct_from_postes,
                'postes_heights_px': [c.get('measured_height_px') for c in poste_candidates if c.get('measured_height_px') and c.get('measured_height_px') > 0] if 'poste_candidates' in locals() else [],
                # ── Campos para calibración de altura en video ──
                'cm_per_px': _cm_per_px_used,
                'animal_bbox_height_px': _animal_bbox_height_px,
                # Bbox de la vaca en coords RESIZED (letterbox) + orig para overlay
                'animal_bbox_resized': (
                    [float(animal_bbox[0]), float(animal_bbox[1]),
                     float(animal_bbox[2]), float(animal_bbox[3])]
                    if animal_bbox else None
                ),
                'animal_bbox_original': (
                    [float((animal_bbox[0] - pad_x) / scale_factor),
                     float((animal_bbox[1] - pad_y) / scale_factor),
                     float((animal_bbox[2] - pad_x) / scale_factor),
                     float((animal_bbox[3] - pad_y) / scale_factor)]
                    if animal_bbox and scale_factor else None
                ),
                'video_w': int(w_orig) if 'w_orig' in locals() else None,
                'video_h': int(h_orig) if 'h_orig' in locals() else None,
                # ── Corrección por raza/categoría/edad ──
                'breed': breed,
                'category': category,
                'age_range': age_range,
                'weight_multiplier': multiplier,
                'raw_weight': round(raw_weight, 2),
                'cow_score': float(_cow_scores[_ci]) if _cow_scores is not None and _ci < len(_cow_scores) else None,
                'rectangle_ref': rect_params_for_details if 'rect_params_for_details' in locals() else None,
            }
            
            # Manejar diferentes combinaciones de retorno
            if visualize and return_eye_coords and return_keypoint_coords:
                return img_rgb, weight, eye_coords, keypoint_coords, success_details
            elif visualize and return_eye_coords:
                return img_rgb, weight, eye_coords, keypoint_coords if return_keypoint_coords else [], success_details
            elif visualize and return_keypoint_coords:
                return img_rgb, weight, [], keypoint_coords, success_details
            elif visualize:
                return img_rgb, weight, success_details
            elif return_eye_coords and return_keypoint_coords:
                return weight, eye_coords, keypoint_coords, success_details
            elif return_eye_coords:
                return weight, eye_coords, keypoint_coords if return_keypoint_coords else [], success_details
            elif return_keypoint_coords:
                return weight, [], keypoint_coords, success_details
            else:
                return weight, success_details

        # Crear diccionario detallado de qué puntos faltan (según el método de escala)
        # Solo marcar 'eyes' como faltante si es relevante para el método seleccionado
        missing_points = {
            'eyes': not bool(dist) if scale_method != 'poste' else False,  # Solo relevante si no es 'poste'
            'dist1': not bool(dist1),
            'dist2': not bool(dist2),
            'scale_reference': False  # Se calculará según el método
        }
        
        # Calcular scale_reference según el método seleccionado
        if scale_method == 'eyes':
            missing_points['scale_reference'] = not bool(dist)
        elif scale_method == 'poste':
            missing_points['scale_reference'] = poste_selected is None
        else:  # 'both'
            missing_points['scale_reference'] = not bool(dist) and not (poste_selected is not None or (escala_postes is not None and self.use_postes_reference))
        
        # Crear mensaje descriptivo de qué falta (según el método de escala seleccionado)
        missing_list = []
        if missing_points['scale_reference']:
            if scale_method == 'eyes':
                missing_list.append('ojos (referencia de escala)')
            elif scale_method == 'poste':
                # Mensaje más detallado para postes
                postes_count = len(postes_all) if 'postes_all' in locals() else 0
                if postes_count == 0:
                    missing_list.append('poste rojo (no se detectaron postes en la imagen)')
                elif poste_selected is None:
                    candidates = len(poste_candidates) if 'poste_candidates' in locals() else 0
                    rejected = len(poste_rejected) if 'poste_rejected' in locals() else 0
                    missing_list.append(f'poste rojo (detectados {postes_count} postes, {candidates} candidatos, {rejected} rechazados - ninguno seleccionado)')
                else:
                    missing_list.append('poste rojo (referencia de escala de 50cm)')
            else:  # 'both'
                if not bool(dist) and not (poste_selected is not None):
                    missing_list.append('ojos o poste rojo (referencia de escala)')
                elif not bool(dist):
                    missing_list.append('ojos (poste rojo disponible como fallback)')
                elif poste_selected is None:
                    missing_list.append('poste rojo (ojos disponibles como fallback)')
        if missing_points['dist1']:
            missing_list.append('dist1 (ancho del cuerpo)')
        if missing_points['dist2']:
            missing_list.append('dist2 (alto del cuerpo)')
        
        missing_message = f"Puntos faltantes: {', '.join(missing_list) if missing_list else 'ninguno'}"
        _log(f"weight=missing reasons: dist_ref={bool(dist)} dist1={bool(dist1)} dist2={bool(dist2)} escala_postes={escala_postes is not None} elapsed_ms={(time.time()-t0)*1000:.1f}")
        _log(f"weight=missing_details: {missing_message}")
        
        # Altura del bbox del animal en px (para calibración en video incluso cuando falla el peso)
        _animal_bbox_height_px = float(animal_bbox[3] - animal_bbox[1]) if animal_bbox else None

        # Crear objeto de error detallado
        error_details = {
            'missing_points': missing_points,
            'message': missing_message,
            'scale_method_selected': scale_method,
            'has_eyes': bool(dist),
            'has_dist1': bool(dist1),
            'has_dist2': bool(dist2),
            'has_scale_reference': bool(dist) or (escala_postes is not None and self.use_postes_reference) or (poste_selected is not None),
            'eyes_count': len(eye_centers) if 'eye_centers' in locals() else 0,
            'keypoints_found': keypoints_found if 'keypoints_found' in locals() else False,
            'poste_selected': poste_selected['bbox'] if poste_selected else None,
            'postes_detected': len(postes_all) if 'postes_all' in locals() else 0,
            'scale_from_postes': _scale_direct_from_postes if '_scale_direct_from_postes' in locals() else False,
            'postes_candidates': len(poste_candidates) if 'poste_candidates' in locals() else 0,
            'postes_rejected': len(poste_rejected) if 'poste_rejected' in locals() else 0,
            'postes_heights_px': [c.get('measured_height_px') for c in poste_candidates if c.get('measured_height_px') and c.get('measured_height_px') > 0] if 'poste_candidates' in locals() else [],
            # ── Campos para calibración de altura en video ──
            'cm_per_px': _available_cm_per_px,  # Escala disponible de postes u ojos (puede ser None)
            'animal_bbox_height_px': _animal_bbox_height_px,
            'dist1_px': float(dist1) if dist1 else None,
            'dist2_px': float(dist2) if dist2 else None,
            'cow_score': float(_cow_scores[_ci]) if '_cow_scores' in locals() and _cow_scores is not None and '_ci' in locals() and _ci < len(_cow_scores) else None,
        }

        # Manejar diferentes combinaciones de retorno cuando no hay peso
        # Incluir error_details en el retorno cuando sea posible
        if visualize and return_eye_coords and return_keypoint_coords:
            return img_rgb, None, eye_coords, keypoint_coords, error_details
        elif visualize and return_eye_coords:
            return img_rgb, None, eye_coords, keypoint_coords if return_keypoint_coords else [], error_details
        elif visualize and return_keypoint_coords:
            return img_rgb, None, [], keypoint_coords, error_details
        elif visualize:
            return img_rgb, None, error_details
        elif return_eye_coords and return_keypoint_coords:
            return None, eye_coords, keypoint_coords, error_details
        elif return_eye_coords:
            return None, eye_coords, keypoint_coords if return_keypoint_coords else [], error_details
        elif return_keypoint_coords:
            return None, [], keypoint_coords, error_details
        else:
            return None, error_details

