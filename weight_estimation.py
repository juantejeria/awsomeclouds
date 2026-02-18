"""
Estimación de peso de ganado a partir de imágenes usando visión por computadora.

Clase principal: WeightEstimator
  - Detecta el cuerpo del animal y sus keypoints con YOLO (cow.pt)
  - Detecta ojos con YOLO de segmentación (eye.pt) para escala por distancia inter-ocular
  - Detecta postes rojos (sticker.pt / color) para escala por altura conocida (122 cm)
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

try:
    from depth_estimation import DepthEstimator
    DEPTH_ESTIMATOR_AVAILABLE = True
except ImportError:
    DEPTH_ESTIMATOR_AVAILABLE = False
    DepthEstimator = None

try:
    from transformers import pipeline as hf_pipeline
    HF_DEPTH_AVAILABLE = True
except ImportError:
    HF_DEPTH_AVAILABLE = False

from breed_coefficients import get_weight_multiplier

# HSV del rojo de referencia (rojo tiene dos rangos: 0-10 y 170-180)
# Rangos más permisivos para detectar rojo puro (bandas rojas de 122cm)
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
    
    def _select_post_for_scale(self, postes, image, animal_bbox=None, band_tolerance=0.2):
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
            band_bottom = ay2 + int(animal_height * band_tolerance)
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
                in_band = y1 >= band_top and y2 <= band_bottom
            
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

    def estimate_weight(self, img_path, visualize=True, debug=False, debug_context="", return_eye_coords=False, return_keypoint_coords=False, roi_offset=(0, 0), scale_method='both', breed="desconocido", category="desconocido", age_range="desconocido", override_cm_per_px=None, yolo_imgsz=None):
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
                - 'poste': Solo busca poste rojo (122cm)
        
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

        # Leer imagen
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {img_path}")
        _log(f"start img_path={img_path} shape={getattr(img, 'shape', None)} visualize={visualize}")
        
        # Asegurar formato BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        img = np.ascontiguousarray(img)
        
        # Redimensionar imagen preservando aspecto (letterbox)
        new_width = 1040
        new_height = 640
        h_orig, w_orig = img.shape[:2]
        scale_factor = min(new_width / w_orig, new_height / h_orig)
        scaled_w = int(w_orig * scale_factor)
        scaled_h = int(h_orig * scale_factor)
        scaled_img = cv2.resize(img, (scaled_w, scaled_h))
        # Centrar en canvas del tamaño objetivo (padding gris)
        resized_image = np.full((new_height, new_width, 3), 114, dtype=np.uint8)
        pad_x = (new_width - scaled_w) // 2
        pad_y = (new_height - scaled_h) // 2
        resized_image[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = scaled_img
        _log(f"resize: orig={w_orig}x{h_orig} scale={scale_factor:.4f} scaled={scaled_w}x{scaled_h} pad=({pad_x},{pad_y})")
        
        # Convertir a RGB para visualización
        img_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # PASO 1: Detectar animal usando cow_model
        # Primero intentar en IMAGEN ORIGINAL (mayor resolución, YOLO maneja su propio resize)
        # Si falla, fallback a imagen letterbox
        _cow_kwargs = dict(save=False, conf=self.keypoint_conf, iou=self.iou_threshold)
        if yolo_imgsz:
            _cow_kwargs['imgsz'] = yolo_imgsz

        # Detección en imagen original
        _log(f"cow_detection: intentando en imagen original ({w_orig}x{h_orig}) conf={self.keypoint_conf:.4f}")
        results2_orig = self.cow_model(img, **_cow_kwargs)

        # Extraer y mapear coordenadas de original → letterbox
        _cow_boxes_mapped = None
        _cow_keypoints_mapped = None
        _cow_scores = None
        _cow_classes = None
        _cow_detected = False

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
            # Fallback 1: intentar en imagen letterbox
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
            # Fallback 2: cow_model con augment=True (test-time augmentation: multi-escala + flips)
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
            # Fallback 3: modelo COCO preentrenado (solo bbox, sin keypoints)
            # COCO class 19 = "cow", 21 = "bear", 17 = "horse" - filtramos solo bovinos
            _log(f"cow_detection: reintentando con modelo COCO (yolov8n) en imagen original")
            try:
                _coco_results = self.coco_model(img, save=False, conf=0.1, iou=self.iou_threshold, classes=[19])
                _best_coco_box = None
                _best_coco_score = 0
                for result in _coco_results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes_coco = result.boxes.xyxy.cpu().numpy()
                        scores_coco = result.boxes.conf.cpu().numpy()
                        for box, score in zip(boxes_coco, scores_coco):
                            if score > _best_coco_score:
                                _best_coco_score = score
                                _best_coco_box = box
                if _best_coco_box is not None:
                    _cow_detected = True
                    # Mapear a letterbox
                    mapped = _best_coco_box.copy()
                    mapped[0] = _best_coco_box[0] * scale_factor + pad_x
                    mapped[1] = _best_coco_box[1] * scale_factor + pad_y
                    mapped[2] = _best_coco_box[2] * scale_factor + pad_x
                    mapped[3] = _best_coco_box[3] * scale_factor + pad_y
                    _cow_boxes_mapped = np.array([mapped])
                    _cow_scores = np.array([_best_coco_score])
                    _cow_classes = np.array([19.0])
                    _cow_keypoints_mapped = None  # COCO no tiene keypoints de vaca
                    _log(f"cow_detection: COCO detectó vaca score={_best_coco_score:.3f} bbox={_best_coco_box.tolist()} (fallback 3 - solo bbox, sin keypoints)")
            except Exception as e:
                _log(f"cow_detection: COCO fallback falló: {e}")

        if not _cow_detected:
            _log(f"cow_detection: no se detectó animal con ninguna estrategia")

        # Obtener bbox del animal para definir región de cabeza
        head_region = None
        animal_bbox = None
        img_height, img_width = resized_image.shape[:2]

        if _cow_detected and _cow_boxes_mapped is not None and len(_cow_boxes_mapped) > 0:
            animal_x1, animal_y1, animal_x2, animal_y2 = map(int, _cow_boxes_mapped[0])
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

            if len(keypoints) > 0 and len(keypoints[0]) >= 5:
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
                kp1_conf = float(keypoints[0][1][2])
                kp2_conf = float(keypoints[0][2][2])
                kp3_conf = float(keypoints[0][3][2])
                kp4_conf = float(keypoints[0][4][2])

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

                point1 = keypoints[0][1][0], keypoints[0][1][1]
                point2 = keypoints[0][2][0], keypoints[0][2][1]
                point3 = keypoints[0][3][0], keypoints[0][3][1]
                point4 = keypoints[0][4][0], keypoints[0][4][1]

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
                    for keypoint, box, score, cls in zip(keypoints, boxes, scores, classes):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Dibujar todos los keypoints
                        for kp in keypoint:
                            kp_x, kp_y = int(kp[0]), int(kp[1])
                            cv2.circle(img_rgb, (kp_x, kp_y), 3, (0, 0, 255), -1)

                        # Resaltar los keypoints usados para dist1 y dist2 con colores diferentes
                        # point1 y point2 para dist1 (azul)
                        cv2.circle(img_rgb, (int(point1[0]), int(point1[1])), 8, (255, 0, 0), 2)
                        cv2.circle(img_rgb, (int(point2[0]), int(point2[1])), 8, (255, 0, 0), 2)
                        cv2.line(img_rgb, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (255, 0, 0), 2)

                        # point3 y point4 para dist2 (cyan)
                        cv2.circle(img_rgb, (int(point3[0]), int(point3[1])), 8, (255, 255, 0), 2)
                        cv2.circle(img_rgb, (int(point4[0]), int(point4[1])), 8, (255, 255, 0), 2)
                        cv2.line(img_rgb, (int(point3[0]), int(point3[1])), (int(point4[0]), int(point4[1])), (255, 255, 0), 2)

                        label = f'{self.cow_model.names[int(cls)]} {score:.2f}'
                        cv2.putText(img_rgb, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Etiquetas para dist1 y dist2
                        mid1_x, mid1_y = int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2)
                        mid2_x, mid2_y = int((point3[0] + point4[0]) / 2), int((point3[1] + point4[1]) / 2)
                        cv2.putText(img_rgb, f'dist1={dist1:.1f}px', (mid1_x, mid1_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        cv2.putText(img_rgb, f'dist2={dist2:.1f}px', (mid2_x, mid2_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        _log(f"keypoints: found={keypoints_found} dist1={'ok' if dist1 else 'missing'} dist2={'ok' if dist2 else 'missing'}")

        # Detectar postes (solo si scale_method no es 'eyes')
        postes_all = []
        poste_selected = None
        poste_candidates = []
        poste_rejected = []
        if scale_method != 'eyes' and self.use_postes_reference and self.depth_estimator:
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
            
            poste_selected, poste_candidates, poste_rejected = self._select_post_for_scale(
                postes_all, resized_image, animal_bbox=animal_bbox, band_tolerance=0.2
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
                band_top = max(0, int(ay1 - animal_height * 0.2))
                band_bottom = min(img_height - 1, int(ay2 + animal_height * 0.2))
                cv2.rectangle(img_rgb, (ax1, ay1), (ax2, ay2), (0, 255, 255), 2)
                cv2.line(img_rgb, (0, band_top), (img_width, band_top), (255, 255, 0), 2)
                cv2.line(img_rgb, (0, band_bottom), (img_width, band_bottom), (255, 255, 0), 2)

            # Mostrar TODOS los postes detectados inicialmente (en magenta/rosa para distinguirlos)
            for i, p in enumerate(postes_all):
                x1, y1, x2, y2 = map(int, p['bbox'])
                # Color magenta/rosa para todos los postes detectados
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 255), 2)
                # Etiqueta con número e información
                label = f'POSTE {i+1}'
                score = p.get('score', 0)
                red_ratio = p.get('yellow_ratio', 0)
                cv2.putText(img_rgb, f'{label} (score:{score:.2f} rojo:{red_ratio:.2f})', 
                           (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Mostrar candidatos (en azul)
            for p in poste_candidates:
                x1, y1, x2, y2 = map(int, p['bbox'])
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Azul para candidatos
                cv2.putText(img_rgb, 'CANDIDATO', (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Mostrar rechazados (en rojo claro)
            for p in poste_rejected:
                x1, y1, x2, y2 = map(int, p['bbox'])
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 200), 2)  # Rojo más claro para rechazados
                reason = p.get('reason', 'rejected')
                cv2.putText(img_rgb, f'RECHAZADO ({reason})', (x1, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
            
            # Mostrar línea de medición en cada candidato (todos son referencia, usamos promedio)
            measured_heights = []
            for idx_c, p in enumerate(poste_candidates):
                x1, y1, x2, y2 = map(int, p['bbox'])
                measured_h = p.get('measured_height_px')
                if measured_h and measured_h > 0:
                    measured_heights.append(measured_h)
                    # Recuadro verde para candidatos de referencia
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 200, 0), 3)
                    # Etiqueta
                    label_ref = f'REF {idx_c+1}: {measured_h:.1f}px = 122cm'
                    (tw_r, th_r), _ = cv2.getTextSize(label_ref, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(img_rgb, (x1, max(0, y1 - th_r - 10)), (x1 + tw_r + 4, max(0, y1 - 2)), (0, 0, 0), -1)
                    cv2.putText(img_rgb, label_ref, (x1 + 2, max(th_r + 2, y1 - 4)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    # Línea de medición
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

            # Mostrar promedio si hay 2+ candidatos
            if len(measured_heights) >= 2:
                avg_h = sum(measured_heights) / len(measured_heights)
                avg_text = f'PROMEDIO: {avg_h:.1f}px = 122cm'
                (tw_a, th_a), _ = cv2.getTextSize(avg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                # Mostrar en la esquina superior derecha
                avg_x = img_width - tw_a - 10
                avg_y = 25
                cv2.rectangle(img_rgb, (avg_x - 4, avg_y - th_a - 6), (avg_x + tw_a + 4, avg_y + 6), (0, 0, 0), -1)
                cv2.putText(img_rgb, avg_text, (avg_x, avg_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Intentar usar postes como referencia alternativa SOLO si está habilitado explícitamente
        escala_postes = None
        
        # IMPORTANTE: Solo intentar usar postes si está explícitamente habilitado
        if self.use_postes_reference and self.depth_estimator:
            _log(f"postes: intentando usar postes como referencia (habilitado)")
            # animal_bbox ya fue extraído y mapeado a letterbox en PASO 1
            
            resultado_depth = None
            if animal_bbox:
                # Pasar measured_height_px del poste seleccionado para mayor precisión
                _poste_measured_h = poste_selected.get('measured_height_px') if poste_selected else None
                resultado_depth = self.depth_estimator.estimar_escala_con_postes(
                    resized_image,
                    animal_bbox=animal_bbox,
                    debug=debug,
                    measured_height_px=_poste_measured_h
                )
                escala_postes = resultado_depth.get('escala')

                if escala_postes:
                    _log(f"postes: escala={escala_postes:.4f} cm/px profundidad_postes={resultado_depth.get('profundidad_postes'):.2f}cm profundidad_animal={resultado_depth.get('profundidad_animal'):.2f}cm")

                    # Validar que la vaca esté entre ambos postes (horizontalmente)
                    _p1_bbox = resultado_depth.get('poste1_bbox')
                    _p2_bbox = resultado_depth.get('poste2_bbox')
                    if _p1_bbox and _p2_bbox and animal_bbox:
                        _p1_cx = (_p1_bbox[0] + _p1_bbox[2]) / 2
                        _p2_cx = (_p2_bbox[0] + _p2_bbox[2]) / 2
                        _post_left = min(_p1_cx, _p2_cx)
                        _post_right = max(_p1_cx, _p2_cx)
                        _animal_cx = (animal_bbox[0] + animal_bbox[2]) / 2
                        if _post_left <= _animal_cx <= _post_right:
                            _log(f"postes: vaca entre postes OK (animal_cx={_animal_cx:.1f} entre [{_post_left:.1f}, {_post_right:.1f}])")
                        else:
                            _log(f"postes: RECHAZADO - vaca NO está entre postes (animal_cx={_animal_cx:.1f} fuera de [{_post_left:.1f}, {_post_right:.1f}])")
                            escala_postes = None
                else:
                    _log(f"postes: no se pudo calcular escala con postes (se requieren ambos postes visibles)")
            else:
                _log(f"postes: no se puede usar postes (no hay bbox de animal)")
        else:
            if not self.use_postes_reference:
                _log(f"postes: deshabilitado (use_postes_reference=False)")
            elif not self.depth_estimator:
                _log(f"postes: no disponible (depth_estimator no inicializado)")

        # Flag: la escala fue calculada directamente de 2 postes visibles en ESTE frame
        _scale_direct_from_postes = escala_postes is not None and self.use_postes_reference

        # Fallback: usar escala cacheada externamente (ej. de frames anteriores del video)
        if escala_postes is None and override_cm_per_px is not None:
            escala_postes = override_cm_per_px
            _log(f"postes: usando override_cm_per_px={override_cm_per_px:.5f} (escala cacheada de frames anteriores)")

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
                # Solo usar postes rojos (se requieren AMBOS postes visibles y vaca entre ellos)
                _log(f"weight_method=poste: verificando disponibilidad - escala_postes={escala_postes is not None} use_postes_reference={self.use_postes_reference}")
                if escala_postes and self.use_postes_reference:
                    dist1cm = dist1 * escala_postes
                    dist2cm = dist2 * escala_postes
                    _log(f"weight_method=postes_dobles escala={escala_postes:.4f}cm/px (ambos postes visibles, vaca entre ellos)")
                else:
                    dist1cm = None
                    dist2cm = None
                    if not self.use_postes_reference:
                        _log(f"weight_method=poste: use_postes_reference=False")
                    else:
                        _log(f"weight_method=poste: se requieren ambos postes visibles con la vaca entre ellos para calcular escala")
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
            _log(f"weight=ok value={weight:.4f}kg raw={raw_weight:.4f}kg multiplier={multiplier:.4f} breed={breed} category={category} age={age_range} BL(dist1cm)={dist1cm:.2f} GirthVert(dist2cm)={dist2cm:.2f} girth_circ={'%.2f' % girth_circumference_cm if girth_circumference_cm else 'N/A'}cm elapsed_ms={(time.time()-t0)*1000:.1f}")
            
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
                # ── Corrección por raza/categoría/edad ──
                'breed': breed,
                'category': category,
                'age_range': age_range,
                'weight_multiplier': multiplier,
                'raw_weight': round(raw_weight, 2),
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
                    missing_list.append('poste rojo (referencia de escala de 122cm)')
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

