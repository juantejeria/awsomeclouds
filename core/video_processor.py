"""
Procesamiento de video para reconocimiento y estimación de peso de múltiples vacas.

Clase principal: VideoProcessor
  - Recorre frames del video a una tasa configurable (sample_rate)
  - Detecta vacas con YOLO y las rastrea entre frames usando IoU (tracking simple)
  - Para cada vaca trackeada: ejecuta reconocimiento CNN y estimación de peso
  - Calibra la escala cm/px con postes y la reutiliza en frames sin referencia
  - Agrega resultados por vaca: votación ponderada (identidad), media/mediana (peso)

Dependencias: opencv-python, numpy, weight_estimation, breed_coefficients, testing
"""
import cv2
import numpy as np
from collections import defaultdict
from core.weight_estimation import WeightEstimator
from core.breed_coefficients import get_weight_multiplier
from core.testing import ImageScore
import os
import base64
import math

class VideoProcessor:
    """Procesador de videos para reconocimiento y estimación de peso de múltiples vacas"""
    
    def __init__(self, recognition_model=None, weight_estimator=None, farm=None, version=2, confidence_threshold=0.5, debug=False,
                 yolo_conf=0.25, yolo_iou=0.45, enhance_image=True,
                 min_weight_kg=50, max_weight_kg=3000, eye_distance_max_gap=10, eye_distance_max_samples=3,
                 breed="desconocido", category="desconocido", age_range="desconocido",
                 scale_method="both",
                 output_video_path=None):
        """
        Inicializa el procesador de video
        
        Args:
            recognition_model: Modelo de reconocimiento de ganado
            weight_estimator: Estimador de peso
            farm: Nombre de la granja
            version: Versión del modelo (1 para VGG16, 2 para ResNet50/SENet50)
            confidence_threshold: Umbral de confianza para reconocimiento
            yolo_conf: Umbral de confianza para YOLO (0.0-1.0). Más bajo = más detecciones
            yolo_iou: Umbral de IoU para YOLO (0.0-1.0). Más bajo = permite detecciones más cercanas
            enhance_image: Si True, mejora contraste/brillo de frames antes de detección
        """
        self.recognition_model = recognition_model
        self.weight_estimator = weight_estimator
        self.farm = farm
        self.version = version
        self.confidence_threshold = confidence_threshold
        self.debug = debug
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou
        self.enhance_image = enhance_image
        self.min_weight_kg = min_weight_kg
        self.max_weight_kg = max_weight_kg
        self.eye_distance_max_gap = eye_distance_max_gap
        self.eye_distance_max_samples = eye_distance_max_samples
        self.breed = breed
        self.category = category
        self.age_range = age_range
        self.scale_method = scale_method

        # Video anotado de salida
        self.output_video_path = output_video_path
        self.video_writer = None
        self.output_frame_size = (640, 360)

        # Almacenar resultados por vaca (tracking ID)
        self.cow_results = defaultdict(lambda: {
            'recognitions': [],
            'weights': [],
            'weight_errors': [],
            'frames': [],
            'bboxes': [],
            'track_history': [],  # Lista de {'frame': num, 'bbox': [x1,y1,x2,y2]}
            'eye_distance_px_history': [],  # Lista de {'frame': num, 'eye_distance_px': float}
            'frame_measurements': [],  # Lista de {'frame': num, 'dist1_px': float, 'dist2_px': float, 'eye_distance_px': float|None}
            'reference_frames': [],  # Frames con postes/ojos/keypoints aunque no haya peso
            'detection_frame_image': None,  # Frame donde se detectó por primera vez
            'first_detection_frame': None,   # Número del frame de primera detección
            'last_seen_frame': None,         # Último frame donde se vio la vaca
            'max_weight_frame_image': None,  # Frame del peso máximo
            'max_weight_frame_number': None,  # Número del frame del peso máximo
            'weight_frames': [],  # Lista de todos los frames donde se calculó peso: [{'frame': num, 'weight': kg, 'image': base64}]
            # ── Calibración de altura para fallback de escala ──
            # Cuando un frame tiene un poste visible (escala conocida), guardamos la
            # altura real de la vaca en cm.  Esto permite re-usar la vaca como "regla"
            # en frames donde no hay poste ni ojos.
            'calibrated_height_cm_history': [],  # [{'frame': int, 'height_cm': float, 'cm_per_px': float}]
            'calibration_frames': [],  # Frames where posts established scale: [{'frame': int, 'image': base64, 'cm_per_px': float, ...}]
        })
        
        # Tracking: mantener último bbox por cada vaca para asociar entre frames
        self.tracked_cows = {}  # {track_id: {'last_bbox': bbox, 'last_frame': frame_num}}
        self.next_track_id = 0

        # Caché de escala de postes (detectada en frames tempranos sin vacas)
        self.cached_cm_per_px = None
        self.cached_scale_frame = None       # base64 image of the frame where posts were detected
        self.cached_scale_frame_number = None
        self.cached_poste1_bbox = None       # [x1, y1, x2, y2] in original frame coords
        self.cached_poste2_bbox = None       # [x1, y1, x2, y2] in original frame coords
        
        # Estadísticas globales del video
        self.stats = {
            'total_frames_processed': 0,
            'frames_with_detections': 0,           # Frames donde YOLO detectó al menos 1 vaca
            'frames_without_detections': 0,        # Frames donde YOLO NO detectó NINGUNA vaca
            'total_detections': 0,                 # Total de detecciones YOLO (puede haber múltiples por frame)
            'detections_without_recognition': 0     # Detecciones YOLO que no pudieron ser reconocidas (no están en dataset o confianza baja)
        }
    
    @staticmethod
    def _is_post_visible(frame_bgr, post_bbox, min_red_ratio=0.10):
        """Verifica si un poste rojo es visible en el frame chequeando píxeles rojos
        en la posición cacheada del poste.

        Args:
            frame_bgr: Frame en BGR (coords originales)
            post_bbox: [x1, y1, x2, y2] del poste en coords originales
            min_red_ratio: Porcentaje mínimo de píxeles rojos para considerar visible

        Returns:
            (visible: bool, red_ratio: float)
        """
        h, w = frame_bgr.shape[:2]
        px1 = max(0, int(post_bbox[0]))
        py1 = max(0, int(post_bbox[1]))
        px2 = min(w, int(post_bbox[2]))
        py2 = min(h, int(post_bbox[3]))
        roi = frame_bgr[py1:py2, px1:px2]
        if roi.size == 0:
            return False, 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Rojo en HSV: H=0-10 o H=170-180, S>50, V>50
        mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        mask = mask1 | mask2
        red_ratio = np.count_nonzero(mask) / mask.size
        return red_ratio >= min_red_ratio, float(red_ratio)

    @staticmethod
    def _calculate_iou(bbox1, bbox2):
        """Calcula Intersection over Union (IoU) entre dos bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calcular intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcular áreas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _assign_track_id(self, bbox, frame_number, iou_threshold=0.3, max_frame_gap=10):
        """
        Asigna un track_id a una detección basándose en tracking con IoU
        
        Args:
            bbox: Bounding box de la detección [x1, y1, x2, y2]
            frame_number: Número del frame actual
            iou_threshold: Umbral mínimo de IoU para considerar que es la misma vaca
        
        Returns:
            track_id: ID de tracking asignado
        """
        best_iou = 0
        best_track_id = None
        
        # Buscar vaca existente con mayor IoU
        for track_id, track_data in self.tracked_cows.items():
            # Solo considerar vacas vistas recientemente (gap configurable)
            if frame_number - track_data['last_frame'] <= max_frame_gap:
                iou = self._calculate_iou(bbox, track_data['last_bbox'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
        
        # Si encontramos match, actualizar tracking
        if best_track_id is not None:
            self.tracked_cows[best_track_id]['last_bbox'] = bbox
            self.tracked_cows[best_track_id]['last_frame'] = frame_number
            return best_track_id
        
        # Si no hay match, crear nuevo track
        track_id = f"cow_{self.next_track_id}"
        self.next_track_id += 1
        self.tracked_cows[track_id] = {
            'last_bbox': bbox,
            'last_frame': frame_number
        }
        return track_id

    @staticmethod
    def _calculate_weight_from_scale(dist1_px, dist2_px, eye_distance_px, eye_distance_cm=20.0):
        """Calcula peso usando distancias en píxeles y escala por distancia entre ojos.
        
        eye_distance_cm: distancia inter-ocular real (~18-25 cm para ganado adulto).
        """
        lb = 0.45359237  # libras a kg
        dist1cm = (eye_distance_cm * dist1_px) / eye_distance_px
        dist2cm = (eye_distance_cm * dist2_px) / eye_distance_px
        return (dist1cm * dist2cm * dist2cm * lb) / 300

    @staticmethod
    def _nearest_eye_distance(history, frame_number, max_gap, max_samples=3):
        """Busca distancia entre ojos más cercana en el tiempo (usa mediana de N más cercanas)."""
        if not history:
            return None
        candidates = [h for h in history if abs(h['frame'] - frame_number) <= max_gap and h.get('eye_distance_px')]
        if not candidates:
            return None
        candidates.sort(key=lambda h: abs(h['frame'] - frame_number))
        values = [c['eye_distance_px'] for c in candidates[:max_samples]]
        if not values:
            return None
        return float(np.median(values))
    
    def process_video(self, video_path, frames_per_second=1, max_frames=None):
        """
        Procesa un video frame por frame
        
        Args:
            video_path: Ruta al archivo de video
            frames_per_second: Cuántos frames procesar por segundo (1 = cada segundo)
            max_frames: Máximo número de frames a procesar (None = todos)
        
        Returns:
            Diccionario con resultados agregados por vaca
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Proteger contra FPS inválido (0/None) y asegurar intervalo >= 1
        if not fps or fps <= 0:
            fps = 30.0
        frame_interval = int(fps / frames_per_second) if frames_per_second > 0 else 1
        frame_interval = max(frame_interval, 1)

        # Inicializar VideoWriter para video anotado de salida
        if self.output_video_path:
            output_fps = min(frames_per_second, fps)
            output_fps = max(output_fps, 1.0)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 para browser
            self.video_writer = cv2.VideoWriter(
                self.output_video_path, fourcc, output_fps, self.output_frame_size
            )
            if not self.video_writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # fallback
                self.video_writer = cv2.VideoWriter(
                    self.output_video_path, fourcc, output_fps, self.output_frame_size
                )

        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar solo frames seleccionados
            if frame_count % frame_interval == 0:
                if max_frames and processed_frames >= max_frames:
                    break
                
                # Procesar frame
                self._process_frame(frame, processed_frames)
                processed_frames += 1
                self.stats['total_frames_processed'] = processed_frames
            
            frame_count += 1
        
        cap.release()

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        # Agregar resultados temporales con estadísticas
        return self._aggregate_results()
    
    def _process_frame(self, frame, frame_number):
        """
        Procesa un frame individual
        
        Args:
            frame: Frame del video (numpy array)
            frame_number: Número del frame procesado
        """
        # Guardar frame temporalmente para procesamiento
        temp_path = f'temp_frame_{frame_number}.jpg'
        cv2.imwrite(temp_path, frame)
        
        try:
            # Frame acumulativo para video anotado (recibe anotaciones de TODAS las vacas)
            annotated_video_frame = frame.copy() if self.video_writer is not None else None

            # Mejorar imagen antes de detección (contraste/brillo)
            processed_frame = frame.copy()
            if self.enhance_image:
                # Convertir a LAB para mejorar contraste sin afectar colores
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                processed_frame = cv2.merge([l, a, b])
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_LAB2BGR)
            
            # Pre-detección de postes en cada frame para cachear escala (cm/px)
            # Esto permite detectar postes en frames tempranos sin vacas
            if self.cached_cm_per_px is None and self.weight_estimator and hasattr(self.weight_estimator, 'depth_estimator') and self.weight_estimator.depth_estimator:
                try:
                    poste1_bbox, poste2_bbox = self.weight_estimator.depth_estimator.detect_postes(frame)
                    if poste1_bbox is not None and poste2_bbox is not None:
                        # Calcular escala usando la altura medida del poste
                        h1 = abs(poste1_bbox[3] - poste1_bbox[1])
                        h2 = abs(poste2_bbox[3] - poste2_bbox[1])
                        avg_h_px = (h1 + h2) / 2.0
                        if avg_h_px > 0:
                            poste_height_cm = self.weight_estimator.depth_estimator.poste1_height_cm
                            # Escala en espacio original
                            cm_per_px_orig = poste_height_cm / avg_h_px
                            # Convertir al espacio letterbox (1040x640) que usa estimate_weight
                            frame_h, frame_w = frame.shape[:2]
                            lb_scale = min(1040.0 / frame_w, 640.0 / frame_h)
                            # En letterbox los píxeles son más grandes -> cm_per_px_letterbox = cm_per_px_orig / lb_scale
                            self.cached_cm_per_px = cm_per_px_orig / lb_scale
                            self.cached_scale_frame_number = frame_number
                            # Guardar posiciones de postes para dibujar en frames de calibración
                            self.cached_poste1_bbox = [int(c) for c in poste1_bbox]
                            self.cached_poste2_bbox = [int(c) for c in poste2_bbox]

                            # Draw posts on frame and save as base64
                            scale_vis = frame.copy()
                            p1x1, p1y1, p1x2, p1y2 = map(int, poste1_bbox)
                            p2x1, p2y1, p2x2, p2y2 = map(int, poste2_bbox)
                            poste_ref_cm = self.weight_estimator.depth_estimator.poste1_height_cm

                            # Rectángulos rojos gruesos alrededor de los postes
                            cv2.rectangle(scale_vis, (p1x1, p1y1), (p1x2, p1y2), (0, 0, 255), 4)
                            cv2.rectangle(scale_vis, (p2x1, p2y1), (p2x2, p2y2), (0, 0, 255), 4)

                            # Líneas de medición de altura dentro de cada poste (amarillo)
                            p1_cx = (p1x1 + p1x2) // 2
                            cv2.line(scale_vis, (p1_cx, p1y1), (p1_cx, p1y2), (0, 255, 255), 3)
                            cv2.line(scale_vis, (p1_cx - 10, p1y1), (p1_cx + 10, p1y1), (0, 255, 255), 3)
                            cv2.line(scale_vis, (p1_cx - 10, p1y2), (p1_cx + 10, p1y2), (0, 255, 255), 3)

                            p2_cx = (p2x1 + p2x2) // 2
                            cv2.line(scale_vis, (p2_cx, p2y1), (p2_cx, p2y2), (0, 255, 255), 3)
                            cv2.line(scale_vis, (p2_cx - 10, p2y1), (p2_cx + 10, p2y1), (0, 255, 255), 3)
                            cv2.line(scale_vis, (p2_cx - 10, p2y2), (p2_cx + 10, p2y2), (0, 255, 255), 3)

                            # Etiquetas con altura en píxeles y referencia en cm
                            cv2.putText(scale_vis, f'Poste 1: {h1:.0f}px = {poste_ref_cm:.0f}cm', (p1x1, max(20, p1y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(scale_vis, f'Poste 2: {h2:.0f}px = {poste_ref_cm:.0f}cm', (p2x1, max(20, p2y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            # Escala prominente con fondo
                            scale_text = f'Escala: {self.cached_cm_per_px:.4f} cm/px'
                            (stw, sth), _ = cv2.getTextSize(scale_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                            cv2.rectangle(scale_vis, (15, 5), (25 + stw, 15 + sth), (0, 0, 0), -1)
                            cv2.putText(scale_vis, scale_text,
                                        (20, 10 + sth), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                            # Mantener resolución alta para ver detalles (max 1280x720)
                            sh, sw = scale_vis.shape[:2]
                            scale_ratio = min(1280 / sw, 720 / sh, 1.0)
                            new_sw = int(sw * scale_ratio)
                            new_sh = int(sh * scale_ratio)
                            scale_resized = cv2.resize(scale_vis, (new_sw, new_sh)) if scale_ratio < 1.0 else scale_vis
                            _, scale_buf = cv2.imencode('.jpg', scale_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            self.cached_scale_frame = f"data:image/jpeg;base64,{base64.b64encode(scale_buf).decode('utf-8')}"

                            if self.debug:
                                print(f"[VIDEO][frame={frame_number}] POSTES DETECTADOS: h1={h1:.0f}px h2={h2:.0f}px avg={avg_h_px:.0f}px -> cm_per_px_orig={cm_per_px_orig:.5f} cm_per_px_letterbox={self.cached_cm_per_px:.5f} (lb_scale={lb_scale:.4f})")
                    elif self.debug and frame_number < 5:
                        print(f"[VIDEO][frame={frame_number}] postes: p1={'SI' if poste1_bbox else 'NO'} p2={'SI' if poste2_bbox else 'NO'}")
                except Exception as e:
                    if self.debug:
                        print(f"[VIDEO][frame={frame_number}] Error en pre-detección de postes: {e}")

            # Detectar vacas en el frame usando YOLO
            if self.weight_estimator:
                # Usar modelo YOLO para detectar vacas con umbrales configurables
                results = self.weight_estimator.cow_model(
                    processed_frame, 
                    save=False,
                    conf=self.yolo_conf,
                    iou=self.yolo_iou
                )

                # Contar detecciones en este frame
                total_boxes = 0
                for r in results:
                    if r.boxes is not None:
                        total_boxes += len(r.boxes)
                
                if self.debug:
                    print(f"[VIDEO][frame={frame_number}] cow_model boxes={total_boxes}")
                
                # Actualizar estadísticas
                if total_boxes > 0:
                    self.stats['frames_with_detections'] += 1
                    self.stats['total_detections'] += total_boxes
                else:
                    self.stats['frames_without_detections'] += 1
                    # Overlay "Sin detecciones" en video anotado para frames vacíos
                    if annotated_video_frame is not None:
                        cv2.putText(annotated_video_frame, 'Sin detecciones',
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)

                for i, result in enumerate(results):
                    if result.boxes is not None and len(result.boxes) > 0:
                        # Extraer bounding boxes (puede haber múltiples detecciones)
                        boxes = result.boxes.xyxy.cpu().numpy()
                        for j, bbox in enumerate(boxes):
                            # Asignar track_id usando tracking con IoU
                            track_id = self._assign_track_id(bbox.tolist(), frame_number)
                            
                            # Recortar región de la vaca con EXPANSIÓN para incluir la cabeza
                            x1, y1, x2, y2 = map(int, bbox)

                            # Dibujar bbox + track_id en video anotado
                            if annotated_video_frame is not None:
                                cv2.rectangle(annotated_video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_video_frame, f'{track_id}',
                                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            # Calcular dimensiones del bbox
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            
                            # EXPANDIR el ROI para incluir la cabeza que puede estar fuera del bbox
                            # Expandir hacia arriba (30% más) para incluir cabeza
                            # Expandir hacia los lados (20% más) para incluir cabeza cuando está de lado
                            # Expandir hacia abajo (10% más) por seguridad
                            expansion_top = int(bbox_height * 0.3)  # 30% hacia arriba
                            expansion_sides = int(bbox_width * 0.2)  # 20% hacia los lados
                            expansion_bottom = int(bbox_height * 0.1)  # 10% hacia abajo
                            
                            # Aplicar expansión con límites del frame
                            frame_height, frame_width = frame.shape[:2]
                            expanded_x1 = max(0, x1 - expansion_sides)
                            expanded_y1 = max(0, y1 - expansion_top)
                            expanded_x2 = min(frame_width, x2 + expansion_sides)
                            expanded_y2 = min(frame_height, y2 + expansion_bottom)
                            
                            # Recortar ROI expandido
                            cow_roi = frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
                            
                            # Guardar offset de expansión para mapear coordenadas correctamente
                            roi_offset_x = expanded_x1
                            roi_offset_y = expanded_y1
                            
                            if cow_roi.size > 0:
                                # Guardar frame de detección si es la primera vez que vemos esta vaca
                                if track_id not in self.cow_results or len(self.cow_results[track_id]['frames']) == 0:
                                    # Dibujar bbox en el frame completo para mostrar el contexto
                                    frame_with_bbox = frame.copy()
                                    cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    cv2.putText(frame_with_bbox, f'{track_id}', (x1, y1 - 10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    # Guardar frame de detección (redimensionado para no ocupar mucho espacio)
                                    detection_frame = cv2.resize(frame_with_bbox, (640, 360))  # Mantener aspect ratio aproximado
                                    self.cow_results[track_id]['detection_frame_image'] = detection_frame
                                    self.cow_results[track_id]['first_detection_frame'] = frame_number
                                
                                # Guardar tracking en todos los frames detectados
                                self.cow_results[track_id]['track_history'].append({
                                    'frame': frame_number,
                                    'bbox': bbox.tolist()
                                })
                                self.cow_results[track_id]['last_seen_frame'] = frame_number
                                
                                # Guardar ROI temporalmente
                                roi_path = f'temp_roi_{track_id}_{frame_number}.jpg'
                                cv2.imwrite(roi_path, cow_roi)

                                # Full frame for weight estimation (posts are outside ROI crop)
                                frame_path = f'temp_frame_{track_id}_{frame_number}.jpg'
                                cv2.imwrite(frame_path, frame)
                                
                                # Reconocimiento de ganado
                                if self.recognition_model:
                                    try:
                                        preds = ImageScore(
                                            model=self.recognition_model,
                                            img=roi_path,
                                            farm=self.farm,
                                            version=self.version,
                                            confidence_threshold=self.confidence_threshold
                                        ).scores()
                                        
                                        self.cow_results[track_id]['recognitions'].append({
                                            'predictions': preds['predictions'],
                                            'metadata': preds['metadata'],
                                            'frame': frame_number
                                        })
                                    except Exception as e:
                                        if self.debug:
                                            print(f"Error en reconocimiento frame {frame_number} track={track_id}: {e}")
                                
                                # Estimación de peso
                                if self.weight_estimator:
                                    try:
                                        # Siempre activar debug para diagnosticar problemas de peso
                                        # Solicitar también coordenadas de ojos y keypoints para visualización
                                        result = self.weight_estimator.estimate_weight(
                                            frame_path,
                                            visualize=True,
                                            debug=True,  # Siempre activar para ver qué falta
                                            debug_context=f"[frame={frame_number} track={track_id}]",
                                            return_eye_coords=True,
                                            return_keypoint_coords=True,
                                            roi_offset=(0, 0),  # Full frame: no offset needed
                                            scale_method=self.scale_method,
                                            breed=self.breed,
                                            category=self.category,
                                            age_range=self.age_range,
                                            override_cm_per_px=self.cached_cm_per_px
                                        )
                                        
                                        # Manejar diferentes tipos de retorno (ahora incluye error_details)
                                        weight = None
                                        eye_coords = []
                                        keypoint_coords = []
                                        error_details = None
                                        
                                        weight_img = None
                                        if isinstance(result, tuple):
                                            if len(result) == 5:
                                                # img_rgb, weight, eye_coords, keypoint_coords, details
                                                weight_img, weight, eye_coords, keypoint_coords, error_details = result
                                            elif len(result) == 4:
                                                # img_rgb, weight, error_details (con visualize)
                                                if isinstance(result[0], np.ndarray):
                                                    weight_img, weight = result[0], result[1]
                                                    error_details = result[3] if len(result) > 3 else None
                                                else:
                                                    weight, eye_coords, keypoint_coords, error_details = result
                                            elif len(result) == 3:
                                                weight, eye_coords, keypoint_coords = result[0], result[1], result[2]
                                            elif len(result) == 2:
                                                # Puede ser (weight, error_details) o (weight, eye_coords)
                                                if isinstance(result[1], dict) and 'missing_points' in result[1]:
                                                    weight, error_details = result[0], result[1]
                                                else:
                                                    weight, eye_coords = result[0], result[1]
                                                    keypoint_coords = []
                                            else:
                                                weight = result[0] if len(result) > 0 else None
                                        else:
                                            weight = result

                                        # Guardar frame de referencia con visualizaciones (postes/ojos/keypoints)
                                        if 'weight_img' in locals() and weight_img is not None:
                                            ref_img = cv2.resize(weight_img, (640, 360))
                                            _, ref_buffer = cv2.imencode('.jpg', cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
                                            ref_base64 = base64.b64encode(ref_buffer).decode('utf-8')
                                            self.cow_results[track_id]['reference_frames'].append({
                                                'frame': frame_number,
                                                'image': f"data:image/jpeg;base64,{ref_base64}"
                                            })

                                        # Obtener distancia entre ojos (px) si está disponible en eye_coords
                                        eye_distance_px_current = None
                                        if eye_coords and len(eye_coords) >= 2:
                                            c1 = eye_coords[0].get('center')
                                            c2 = eye_coords[1].get('center')
                                            if c1 and c2:
                                                eye_distance_px_current = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
                                                self.cow_results[track_id]['eye_distance_px_history'].append({
                                                    'frame': frame_number,
                                                    'eye_distance_px': float(eye_distance_px_current)
                                                })

                                        # Guardar medidas de keypoints por frame (dist1/dist2)
                                        dist1_px = None
                                        dist2_px = None
                                        if keypoint_coords and isinstance(keypoint_coords[-1], dict):
                                            dist1_px = keypoint_coords[-1].get('dist1_px')
                                            dist2_px = keypoint_coords[-1].get('dist2_px')
                                        # ── Calibración de altura: guardar la altura real de la vaca ──
                                        # cuando tenemos una escala válida (poste o ojos).
                                        details_for_calib = error_details if error_details and isinstance(error_details, dict) else None
                                        if weight is not None and isinstance(result, tuple) and len(result) >= 3:
                                            # El último elemento de la tupla es success_details
                                            _sd = result[-1] if isinstance(result[-1], dict) else None
                                            if _sd:
                                                details_for_calib = _sd

                                        # Extraer datos crudos del frame para recálculo en frontend
                                        _frame_bbox_h = None
                                        _frame_postes = 0
                                        _scale_from_postes = False

                                        # ── DIAGNÓSTICO: qué tenemos en details_for_calib ──
                                        print(f"[CALIB-DIAG][frame={frame_number} track={track_id}] details_for_calib is None: {details_for_calib is None}, weight={weight}, result type={type(result).__name__}, result len={len(result) if isinstance(result, tuple) else 'N/A'}")
                                        if details_for_calib and isinstance(details_for_calib, dict):
                                            print(f"[CALIB-DIAG][frame={frame_number} track={track_id}] cm_per_px={details_for_calib.get('cm_per_px')}, animal_bbox_height_px={details_for_calib.get('animal_bbox_height_px')}, scale_from_postes={details_for_calib.get('scale_from_postes')}")

                                        if details_for_calib and isinstance(details_for_calib, dict):
                                            _frame_bbox_h = details_for_calib.get('animal_bbox_height_px')
                                            _frame_postes = details_for_calib.get('postes_detected', 0)
                                            _scale_from_postes = details_for_calib.get('scale_from_postes', False)

                                        self.cow_results[track_id]['frame_measurements'].append({
                                            'frame': frame_number,
                                            'dist1_px': float(dist1_px) if dist1_px is not None else None,
                                            'dist2_px': float(dist2_px) if dist2_px is not None else None,
                                            'eye_distance_px': float(eye_distance_px_current) if eye_distance_px_current is not None else None,
                                            'animal_bbox_height_px': float(_frame_bbox_h) if _frame_bbox_h else None,
                                            'postes_detected': _frame_postes,
                                        })

                                        if details_for_calib and isinstance(details_for_calib, dict):
                                            _cm_px = details_for_calib.get('cm_per_px')
                                            _anim_h_px = details_for_calib.get('animal_bbox_height_px')
                                            print(f"[CALIB-DIAG][frame={frame_number} track={track_id}] CHECK: _cm_px={_cm_px}, _anim_h_px={_anim_h_px}, pass={bool(_cm_px and _anim_h_px and _cm_px > 0 and _anim_h_px > 0)}")
                                            if _cm_px and _anim_h_px and _cm_px > 0 and _anim_h_px > 0:
                                                _cow_h_cm = _anim_h_px * _cm_px
                                                self.cow_results[track_id]['calibrated_height_cm_history'].append({
                                                    'frame': frame_number,
                                                    'height_cm': float(_cow_h_cm),
                                                    'cm_per_px': float(_cm_px),
                                                })
                                                if self.debug:
                                                    print(f"[CALIB][frame={frame_number} track={track_id}] cow_height_cm={_cow_h_cm:.1f} cm_per_px={_cm_px:.5f} animal_h_px={_anim_h_px:.1f}")

                                                # ── Filtro por COLOR: verificar que ambos postes rojos son
                                                # realmente visibles en el frame actual (no tapados por la vaca).
                                                # Miramos los píxeles en las posiciones cacheadas de los postes
                                                # y chequeamos que haya suficiente rojo.
                                                _both_posts_visible = False
                                                if self.cached_poste1_bbox and self.cached_poste2_bbox:
                                                    _p1_vis, _p1_red = VideoProcessor._is_post_visible(frame, self.cached_poste1_bbox)
                                                    _p2_vis, _p2_red = VideoProcessor._is_post_visible(frame, self.cached_poste2_bbox)
                                                    _both_posts_visible = _p1_vis and _p2_vis
                                                    print(f"[CALIB-DIAG][frame={frame_number}] poste1_visible={_p1_vis}({_p1_red:.2f}) poste2_visible={_p2_vis}({_p2_red:.2f}) both={_both_posts_visible}")

                                                if _both_posts_visible:
                                                    # Save calibration frame image (sin dibujar postes artificiales,
                                                    # los postes reales ya son visibles en el frame)
                                                    calib_img = weight_img if weight_img is not None else frame.copy()
                                                    if weight_img is not None:
                                                        calib_final = cv2.cvtColor(calib_img, cv2.COLOR_RGB2BGR)
                                                    else:
                                                        calib_final = cv2.resize(calib_img, (1040, 640))
                                                    _, calib_buffer = cv2.imencode('.jpg', calib_final, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                                    calib_base64 = base64.b64encode(calib_buffer).decode('utf-8')
                                                    self.cow_results[track_id]['calibration_frames'].append({
                                                        'frame': frame_number,
                                                        'image': f"data:image/jpeg;base64,{calib_base64}",
                                                        'cm_per_px': float(_cm_px),
                                                        'cow_height_cm': float(_cow_h_cm),
                                                        'animal_bbox_height_px': float(_anim_h_px),
                                                        'postes_detected': _frame_postes,
                                                        'scale_from_postes': _scale_from_postes,
                                                    })

                                        # ── Fallback: usar altura calibrada como escala ──
                                        # Si NO tenemos peso pero SÍ tenemos keypoints (dist1_px, dist2_px)
                                        # y hay una calibración previa → calcular peso usando la vaca como regla.
                                        if weight is None and dist1_px and dist2_px:
                                            calib_history = self.cow_results[track_id]['calibrated_height_cm_history']
                                            if calib_history:
                                                # Usar mediana de alturas calibradas para robustez
                                                calib_heights = [c['height_cm'] for c in calib_history]
                                                median_cow_h_cm = float(np.median(calib_heights))
                                                
                                                # Altura actual de la vaca en px (del bbox en este frame)
                                                cow_h_px_now = float(bbox[3] - bbox[1]) if len(bbox) >= 4 else None
                                                
                                                if cow_h_px_now and cow_h_px_now > 0:
                                                    # Derivar escala: cm/px = cow_height_cm / cow_height_px
                                                    fallback_cm_per_px = median_cow_h_cm / cow_h_px_now
                                                    
                                                    fb_dist1cm = dist1_px * fallback_cm_per_px
                                                    fb_dist2cm = dist2_px * fallback_cm_per_px
                                                    
                                                    lb = 0.45359237
                                                    raw_weight = (fb_dist1cm * fb_dist2cm * fb_dist2cm * lb) / 300
                                                    multiplier = get_weight_multiplier(self.breed, self.category, self.age_range)
                                                    weight = raw_weight * multiplier

                                                    if self.debug:
                                                        print(f"[FALLBACK][frame={frame_number} track={track_id}] "
                                                              f"weight={weight:.2f}kg raw={raw_weight:.2f}kg multiplier={multiplier:.4f} "
                                                              f"using cow-height calibration "
                                                              f"(median_h_cm={median_cow_h_cm:.1f} cow_h_px_now={cow_h_px_now:.1f} "
                                                              f"cm_per_px={fallback_cm_per_px:.5f} calib_frames={len(calib_history)})")

                                        # Validar rango de peso (evitar valores absurdos)
                                        if weight is not None and (weight < self.min_weight_kg or weight > self.max_weight_kg):
                                            if self.debug:
                                                print(f"[WEIGHT][frame={frame_number} track={track_id}] peso fuera de rango ({weight:.2f}kg). Se ignora.")
                                            weight = None
                                            error_details = error_details or {}
                                            error_details['message'] = f"Peso fuera de rango ({self.min_weight_kg}-{self.max_weight_kg} kg)"

                                        # roi_offset_x y roi_offset_y ya están definidos arriba (líneas 265-266)

                                        if weight is not None:
                                            # Extraer distancia entre ojos si está disponible en error_details o success_details
                                            eye_distance_cm = None
                                            eye_distance_px = None
                                            if error_details and isinstance(error_details, dict):
                                                # Si hay error_details, no hay peso, pero podría tener info de ojos
                                                pass
                                            elif len(result) >= 4:
                                                # Buscar detalles en el resultado (success_details)
                                                details = result[3] if len(result) > 3 else None
                                                if details and isinstance(details, dict):
                                                    # Los detalles no tienen la distancia directamente, necesitamos calcularla
                                                    pass
                                            
                                            self.cow_results[track_id]['weights'].append({
                                                'weight': weight,
                                                'frame': frame_number,
                                                'eye_distance_px': eye_distance_px,  # Se calculará después si está disponible
                                                'eye_distance_cm': eye_distance_cm,
                                                'estimated': False
                                            })
                                            self.cow_results[track_id]['bboxes'].append({
                                                'bbox': bbox.tolist(),
                                                'frame': frame_number
                                            })
                                            
                                            # Guardar frame completo con peso y ojos para visualización
                                            frame_with_bbox = frame.copy()
                                            
                                            # Dibujar bbox de la vaca (verde)
                                            cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                            cv2.putText(frame_with_bbox, f'{track_id} - {weight:.2f}kg',
                                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                                            # Dibujar peso en video anotado
                                            if annotated_video_frame is not None:
                                                cv2.putText(annotated_video_frame, f'{weight:.1f}kg',
                                                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                                            # Obtener dimensiones del ROI original para mapeo de coordenadas
                                            roi_height, roi_width = frame.shape[:2]
                                            
                                            # weight_estimation.py usa letterbox (preserva aspecto).
                                            # Necesitamos invertir: las coordenadas vienen en el canvas 1040x640 con padding.
                                            resized_width = 1040
                                            resized_height = 640
                                            lb_scale = min(resized_width / roi_width, resized_height / roi_height)
                                            lb_scaled_w = int(roi_width * lb_scale)
                                            lb_scaled_h = int(roi_height * lb_scale)
                                            lb_pad_x = (resized_width - lb_scaled_w) // 2
                                            lb_pad_y = (resized_height - lb_scaled_h) // 2
                                            
                                            # Factor para convertir coordenadas del canvas → ROI original:
                                            # roi_coord = (canvas_coord - pad) / lb_scale
                                            scale_x = 1.0 / lb_scale  # multiply after subtracting pad
                                            scale_y = 1.0 / lb_scale
                                            
                                            # Dibujar detecciones de ojos (naranja/rosa) en el frame completo
                                            # Las coordenadas de eye_coords están en el ROI redimensionado (1040x640)
                                            if eye_coords and len(eye_coords) > 0:
                                                eye_centers_frame = []
                                                for eye in eye_coords:
                                                    eye_bbox_resized = eye['bbox']  # [x1_resized, y1_resized, x2_resized, y2_resized]
                                                    eye_center_resized = eye.get('center', [(eye_bbox_resized[0] + eye_bbox_resized[2])/2, 
                                                                                            (eye_bbox_resized[1] + eye_bbox_resized[3])/2])
                                                    
                                                    # Mapear coordenadas del canvas letterbox → ROI original
                                                    # canvas_coord → subtract pad → multiply scale
                                                    eye_x1_roi = int((eye_bbox_resized[0] - lb_pad_x) * scale_x)
                                                    eye_y1_roi = int((eye_bbox_resized[1] - lb_pad_y) * scale_y)
                                                    eye_x2_roi = int((eye_bbox_resized[2] - lb_pad_x) * scale_x)
                                                    eye_y2_roi = int((eye_bbox_resized[3] - lb_pad_y) * scale_y)
                                                    
                                                    # Mapear centro del ojo
                                                    eye_center_x_roi = (eye_center_resized[0] - lb_pad_x) * scale_x
                                                    eye_center_y_roi = (eye_center_resized[1] - lb_pad_y) * scale_y
                                                    
                                                    # estimate_weight runs on the full frame, so letterbox→original
                                                    # mapping already yields full-frame coordinates (no ROI offset needed)
                                                    eye_x1_frame = eye_x1_roi
                                                    eye_y1_frame = eye_y1_roi
                                                    eye_x2_frame = eye_x2_roi
                                                    eye_y2_frame = eye_y2_roi
                                                    eye_center_x_frame = eye_center_x_roi
                                                    eye_center_y_frame = eye_center_y_roi
                                                    
                                                    eye_centers_frame.append((eye_center_x_frame, eye_center_y_frame))
                                                    
                                                    # Dibujar rectángulo de ojo (color naranja/rosa: BGR 22, 22, 229)
                                                    cv2.rectangle(frame_with_bbox, 
                                                                (eye_x1_frame, eye_y1_frame), 
                                                                (eye_x2_frame, eye_y2_frame), 
                                                                (22, 22, 229), 2)
                                                    
                                                    # Dibujar centro del ojo como punto amarillo (usado para calcular distancia)
                                                    cv2.circle(frame_with_bbox, (int(eye_center_x_frame), int(eye_center_y_frame)), 
                                                              5, (0, 255, 255), -1)  # Amarillo para centro
                                                    
                                                    # Etiqueta con confianza
                                                    eye_label = f"Ojo {eye['score']:.2f}"
                                                    cv2.putText(frame_with_bbox, eye_label,
                                                               (eye_x1_frame, eye_y1_frame - 5),
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (22, 22, 229), 2)

                                                    # Duplicar ojos en video anotado
                                                    if annotated_video_frame is not None:
                                                        cv2.rectangle(annotated_video_frame,
                                                                      (eye_x1_frame, eye_y1_frame),
                                                                      (eye_x2_frame, eye_y2_frame),
                                                                      (22, 22, 229), 2)
                                                        cv2.circle(annotated_video_frame,
                                                                   (int(eye_center_x_frame), int(eye_center_y_frame)),
                                                                   4, (0, 255, 255), -1)
                                                
                                                # Dibujar línea entre centros de ojos si hay 2 o más ojos (distancia entre ojos)
                                                if len(eye_centers_frame) >= 2:
                                                    pt1_frame = eye_centers_frame[0]
                                                    pt2_frame = eye_centers_frame[1]
                                                    cv2.line(frame_with_bbox, 
                                                            (int(pt1_frame[0]), int(pt1_frame[1])), 
                                                            (int(pt2_frame[0]), int(pt2_frame[1])), 
                                                            (0, 255, 255), 2)  # Amarillo para línea de distancia entre ojos
                                                    # Etiqueta de distancia
                                                    mid_x = int((pt1_frame[0] + pt2_frame[0]) / 2)
                                                    mid_y = int((pt1_frame[1] + pt2_frame[1]) / 2)
                                                    eye_dist_px = math.sqrt((pt1_frame[0] - pt2_frame[0])**2 + (pt1_frame[1] - pt2_frame[1])**2)
                                                    # Mostrar distancia en píxeles y centímetros (usando referencia de 20cm)
                                                    cv2.putText(frame_with_bbox, f'dist_ojos={eye_dist_px:.1f}px (ref: 20cm)',
                                                               (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                                    
                                                    # Guardar distancia entre ojos para el peso actual si está disponible
                                                    if weight is not None and self.cow_results[track_id]['weights']:
                                                        last_weight_entry = self.cow_results[track_id]['weights'][-1]
                                                        if last_weight_entry['frame'] == frame_number:
                                                            last_weight_entry['eye_distance_px'] = eye_dist_px
                                                            last_weight_entry['eye_distance_cm'] = 20.0  # Distancia inter-ocular ganado adulto
                                            
                                            # Dibujar keypoints usados para dist1 y dist2
                                            if keypoint_coords and len(keypoint_coords) > 0:
                                                # Los primeros 4 elementos son los puntos, el último tiene las distancias
                                                dist_info = keypoint_coords[-1] if isinstance(keypoint_coords[-1], dict) and 'dist1_px' in keypoint_coords[-1] else {}
                                                
                                                for kp_info in keypoint_coords[:4]:  # Solo los primeros 4 son puntos
                                                    if isinstance(kp_info, dict) and 'coords' in kp_info:
                                                        kp_coords_resized = kp_info['coords']
                                                        used_for = kp_info.get('used_for', '')
                                                        
                                                        # Mapear coordenadas del canvas letterbox → ROI original
                                                        kp_x_roi = (kp_coords_resized[0] - lb_pad_x) * scale_x
                                                        kp_y_roi = (kp_coords_resized[1] - lb_pad_y) * scale_y
                                                        
                                                        # Already in full-frame coords (no ROI offset needed)
                                                        kp_x_frame = kp_x_roi
                                                        kp_y_frame = kp_y_roi
                                                        
                                                        # Dibujar punto según su uso
                                                        if used_for == 'dist1':
                                                            # Azul para dist1
                                                            cv2.circle(frame_with_bbox, (int(kp_x_frame), int(kp_y_frame)), 
                                                                      8, (255, 0, 0), 2)  # Azul
                                                        elif used_for == 'dist2':
                                                            # Cyan para dist2
                                                            cv2.circle(frame_with_bbox, (int(kp_x_frame), int(kp_y_frame)), 
                                                                      8, (255, 255, 0), 2)  # Cyan
                                                
                                                # Dibujar líneas entre puntos de dist1 y dist2
                                                if len(keypoint_coords) >= 4:
                                                    # dist1: point1 y point2
                                                    p1 = keypoint_coords[0]['coords']
                                                    p2 = keypoint_coords[1]['coords']
                                                    p1_frame = (int((p1[0] - lb_pad_x) * scale_x), int((p1[1] - lb_pad_y) * scale_y))
                                                    p2_frame = (int((p2[0] - lb_pad_x) * scale_x), int((p2[1] - lb_pad_y) * scale_y))
                                                    cv2.line(frame_with_bbox, p1_frame, p2_frame, (255, 0, 0), 2)  # Azul
                                                    
                                                    # dist2: point3 y point4
                                                    p3 = keypoint_coords[2]['coords']
                                                    p4 = keypoint_coords[3]['coords']
                                                    p3_frame = (int((p3[0] - lb_pad_x) * scale_x), int((p3[1] - lb_pad_y) * scale_y))
                                                    p4_frame = (int((p4[0] - lb_pad_x) * scale_x), int((p4[1] - lb_pad_y) * scale_y))
                                                    cv2.line(frame_with_bbox, p3_frame, p4_frame, (255, 255, 0), 2)  # Cyan
                                                    
                                                    # Etiquetas de distancias
                                                    if dist_info:
                                                        mid1_x = (p1_frame[0] + p2_frame[0]) // 2
                                                        mid1_y = (p1_frame[1] + p2_frame[1]) // 2
                                                        mid2_x = (p3_frame[0] + p4_frame[0]) // 2
                                                        mid2_y = (p3_frame[1] + p4_frame[1]) // 2
                                                        cv2.putText(frame_with_bbox, f'dist1={dist_info.get("dist1_px", 0):.1f}px',
                                                                   (mid1_x, mid1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                                        cv2.putText(frame_with_bbox, f'dist2={dist_info.get("dist2_px", 0):.1f}px',
                                                                   (mid2_x, mid2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                                                    # Duplicar keypoints en video anotado
                                                    if annotated_video_frame is not None:
                                                        cv2.line(annotated_video_frame, p1_frame, p2_frame, (255, 0, 0), 2)
                                                        cv2.line(annotated_video_frame, p3_frame, p4_frame, (255, 255, 0), 2)

                                            # Redimensionar para no ocupar mucho espacio
                                            weight_frame_resized = cv2.resize(frame_with_bbox, (640, 360))
                                            
                                            # Convertir a base64 para enviar al frontend
                                            _, buffer = cv2.imencode('.jpg', weight_frame_resized)
                                            img_base64 = base64.b64encode(buffer).decode('utf-8')
                                            
                                            # Guardar en lista de frames con peso
                                            self.cow_results[track_id]['weight_frames'].append({
                                                'frame': frame_number,
                                                'weight': weight,
                                                'image': f"data:image/jpeg;base64,{img_base64}"
                                            })
                                            
                                            # Guardar frame del peso máximo (para compatibilidad)
                                            current_max = self.cow_results[track_id].get('max_weight_frame_number')
                                            current_max_weight = None
                                            if current_max is not None:
                                                # Buscar peso máximo actual
                                                for w in self.cow_results[track_id]['weights']:
                                                    if w['frame'] == current_max:
                                                        current_max_weight = w['weight']
                                                        break
                                            
                                            if current_max_weight is None or weight > current_max_weight:
                                                # Este es el nuevo peso máximo, guardar frame
                                                max_weight_frame = cv2.resize(frame_with_bbox, (640, 360))
                                                self.cow_results[track_id]['max_weight_frame_image'] = max_weight_frame
                                                self.cow_results[track_id]['max_weight_frame_number'] = frame_number
                                        else:
                                            # Guardar información de error detallada
                                            if error_details:
                                                # Inicializar weight_errors si no existe
                                                if 'weight_errors' not in self.cow_results[track_id]:
                                                    self.cow_results[track_id]['weight_errors'] = []
                                                
                                                self.cow_results[track_id]['weight_errors'].append({
                                                    'frame': frame_number,
                                                    'error_details': error_details,
                                                    'message': error_details.get('message', 'No se pudo calcular peso')
                                                })
                                                print(f"[WEIGHT][frame={frame_number} track={track_id}] {error_details.get('message', 'No se pudo calcular peso')}")
                                            else:
                                                print(f"[WEIGHT][frame={frame_number} track={track_id}] No se pudo calcular peso - revisa logs anteriores")
                                    except Exception as e:
                                        print(f"Error en estimación de peso frame {frame_number} track={track_id}: {e}")
                                
                                # Marcar frame procesado (para estadísticas)
                                self.cow_results[track_id]['frames'].append(frame_number)
                                
                                # Limpiar archivo temporal
                                if os.path.exists(roi_path):
                                    os.remove(roi_path)
                                if os.path.exists(frame_path):
                                    os.remove(frame_path)
            
            # Escribir frame anotado en el video de salida
            if annotated_video_frame is not None and self.video_writer is not None:
                resized = cv2.resize(annotated_video_frame, self.output_frame_size)
                self.video_writer.write(resized)

            # Limpiar frame temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            print(f"Error procesando frame {frame_number}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _aggregate_results(self):
        """
        Agrega resultados temporales para cada vaca con estadísticas detalladas
        
        Returns:
            Diccionario con resultados agregados por vaca y estadísticas globales
        """
        aggregated = {}
        
        for cow_id, data in self.cow_results.items():
            # Asegurar dict base aunque solo haya peso (evita KeyError)
            if cow_id not in aggregated:
                aggregated[cow_id] = {}

            # Estadísticas de detección
            frames_detected = len(data['frames'])
            frames_recognized = len(data['recognitions'])
            frames_with_weight = len(data['weights'])
            
            aggregated[cow_id]['frames_detected'] = frames_detected
            aggregated[cow_id]['frames_recognized'] = frames_recognized
            aggregated[cow_id]['frames_with_weight'] = frames_with_weight
            aggregated[cow_id]['first_detection_frame'] = data.get('first_detection_frame')
            aggregated[cow_id]['last_seen_frame'] = data.get('last_seen_frame')
            aggregated[cow_id]['track_history'] = data.get('track_history', [])
            aggregated[cow_id]['reference_frames'] = data.get('reference_frames', [])
            # Limitar calibration_frames a ~12 muestras distribuidas uniformemente
            _all_calib = data.get('calibration_frames', [])
            _max_calib = 12
            if len(_all_calib) > _max_calib:
                _step = len(_all_calib) / _max_calib
                _sampled = [_all_calib[int(i * _step)] for i in range(_max_calib)]
                aggregated[cow_id]['calibration_frames'] = _sampled
            else:
                aggregated[cow_id]['calibration_frames'] = _all_calib
            print(f"[CALIB-DIAG][AGGREGATE] cow_id={cow_id}: calibration_frames={len(_all_calib)} (sent={len(aggregated[cow_id]['calibration_frames'])}), frame_measurements={len(data.get('frame_measurements', []))}, weights={len(data.get('weights', []))}")
            aggregated[cow_id]['frame_measurements'] = data.get('frame_measurements', [])
            
            # Convertir frame de detección a base64 para enviarlo al frontend
            if data.get('detection_frame_image') is not None:
                detection_frame = data['detection_frame_image']
                # Codificar imagen como JPEG en base64
                _, buffer = cv2.imencode('.jpg', detection_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                aggregated[cow_id]['detection_frame'] = f"data:image/jpeg;base64,{img_base64}"
            else:
                aggregated[cow_id]['detection_frame'] = None

            # Promediar reconocimientos (votación por mayoría ponderada)
            if data['recognitions']:
                all_predictions = {}
                weighted_votes = {}
                total_weight = 0.0
                filtered_count = 0
                for rec in data['recognitions']:
                    pred_class = rec['metadata'].get('predicted_class')
                    confidence = rec['metadata'].get('max_confidence', 0)
                    if pred_class:
                        if pred_class not in all_predictions:
                            all_predictions[pred_class] = []
                        all_predictions[pred_class].append(confidence)
                        # Usar confidence_threshold como filtro mínimo para identidad
                        if confidence >= self.confidence_threshold:
                            weighted_votes[pred_class] = weighted_votes.get(pred_class, 0.0) + confidence
                            total_weight += confidence
                            filtered_count += 1
                
                # Calcular promedio de confianza por clase
                avg_predictions = {}
                for cls, confidences in all_predictions.items():
                    avg_predictions[cls] = np.mean(confidences)
                
                # Clase más frecuente y con mayor confianza promedio
                if avg_predictions:
                    best_class = max(avg_predictions, key=avg_predictions.get)
                    # También calcular voto ponderado para estabilizar identidad por tracking
                    best_weighted_class = None
                    identity_confidence = None
                    if weighted_votes:
                        best_weighted_class = max(weighted_votes, key=weighted_votes.get)
                        identity_confidence = weighted_votes[best_weighted_class] / max(total_weight, 1e-6)
                    aggregated[cow_id].update({
                        'predicted_class': best_weighted_class or best_class,
                        'avg_confidence': avg_predictions[best_class],
                        'recognition_frames': frames_recognized,
                        'identity': {
                            'weighted_class': best_weighted_class,
                            'weighted_confidence': identity_confidence,
                            'weighted_samples': filtered_count,
                            'threshold_used': self.confidence_threshold
                        }
                    })
            else:
                # Vaca detectada por YOLO pero nunca reconocida por CNN (no está en dataset o confianza baja)
                aggregated[cow_id]['predicted_class'] = None
                aggregated[cow_id]['recognition_frames'] = 0
                aggregated[cow_id]['identity'] = None
                # Contar como detección sin reconocimiento
                self.stats['detections_without_recognition'] += frames_detected
            
            # Promediar pesos
            if data['weights']:
                weights = [w['weight'] for w in data['weights']]
                # Encontrar frame del peso máximo y mínimo
                max_weight_entry = max(data['weights'], key=lambda x: x['weight'])
                min_weight_entry = min(data['weights'], key=lambda x: x['weight'])
                
                # Calcular estadísticas de distancia entre ojos
                eye_distances_px = [w.get('eye_distance_px') for w in data['weights'] if w.get('eye_distance_px') is not None]
                
                aggregated[cow_id]['weight'] = {
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights),
                    'max_frame': max_weight_entry['frame'],  # Frame del peso máximo
                    'min_frame': min_weight_entry['frame'],  # Frame del peso mínimo
                    'measurements': frames_with_weight,
                    'eye_distance_px_mean': float(np.mean(eye_distances_px)) if eye_distances_px else None,
                    'eye_distance_px_min': float(np.min(eye_distances_px)) if eye_distances_px else None,
                    'eye_distance_px_max': float(np.max(eye_distances_px)) if eye_distances_px else None,
                    'eye_distance_cm_assumed': 4.0  # Valor asumido en el código
                }
                
                # Guardar imagen del frame del peso máximo si está disponible
                if data.get('max_weight_frame_image') is not None:
                    # Convertir a base64 para enviar al frontend
                    _, buffer = cv2.imencode('.jpg', data['max_weight_frame_image'])
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    aggregated[cow_id]['max_weight_frame_image'] = f"data:image/jpeg;base64,{img_base64}"
                    aggregated[cow_id]['max_weight_frame_number'] = data.get('max_weight_frame_number')
                
                # Incluir todos los frames con peso (ya están en base64)
                aggregated[cow_id]['weight_frames'] = data.get('weight_frames', [])
            else:
                # Vaca detectada pero nunca se pudo calcular peso
                aggregated[cow_id]['weight'] = None

            # Estimación estabilizada usando distancia entre ojos de frames cercanos
            estimated_weights = []
            measurements = data.get('frame_measurements', [])
            eye_history = data.get('eye_distance_px_history', [])
            weight_frames = {w['frame'] for w in data.get('weights', [])}
            for m in measurements:
                frame_num = m.get('frame')
                if frame_num in weight_frames:
                    continue
                dist1_px = m.get('dist1_px')
                dist2_px = m.get('dist2_px')
                if dist1_px is None or dist2_px is None:
                    continue
                eye_distance_px = m.get('eye_distance_px')
                if not eye_distance_px:
                    eye_distance_px = self._nearest_eye_distance(
                        eye_history,
                        frame_num,
                        self.eye_distance_max_gap,
                        self.eye_distance_max_samples
                    )
                if not eye_distance_px:
                    continue
                estimated = self._calculate_weight_from_scale(dist1_px, dist2_px, eye_distance_px)
                if self.min_weight_kg <= estimated <= self.max_weight_kg:
                    estimated_weights.append({
                        'frame': frame_num,
                        'weight': float(estimated),
                        'eye_distance_px': float(eye_distance_px),
                        'estimated': True,
                        'source': 'nearest_eye_distance'
                    })

            if estimated_weights:
                aggregated[cow_id]['estimated_weight_frames'] = estimated_weights
            else:
                aggregated[cow_id]['estimated_weight_frames'] = []

            all_weight_values = [w['weight'] for w in data.get('weights', [])] + [w['weight'] for w in estimated_weights]
            if all_weight_values:
                aggregated[cow_id]['weight_stabilized'] = {
                    'mean': float(np.mean(all_weight_values)),
                    'median': float(np.median(all_weight_values)),
                    'min': float(np.min(all_weight_values)),
                    'max': float(np.max(all_weight_values)),
                    'measurements': len(all_weight_values),
                    'direct_measurements': len(data.get('weights', [])),
                    'estimated_measurements': len(estimated_weights),
                    'eye_distance_window_frames': self.eye_distance_max_gap
                }
            else:
                aggregated[cow_id]['weight_stabilized'] = None
            
            # Incluir información de errores de peso (qué puntos faltaron en cada frame)
            aggregated[cow_id]['weight_errors'] = data.get('weight_errors', [])
        
        # Agregar estadísticas globales al resultado
        result = {
            'cows': aggregated,
            'stats': {
                'total_frames_processed': self.stats['total_frames_processed'],
                'frames_with_detections': self.stats['frames_with_detections'],
                'frames_without_detections': self.stats['frames_without_detections'],
                'total_detections': self.stats['total_detections'],
                'detections_without_recognition': self.stats['detections_without_recognition'],
                'total_cows_tracked': len(aggregated)
            },
            'output_video': self.output_video_path,
            'scale_frame': {
                'image': self.cached_scale_frame,
                'frame': self.cached_scale_frame_number,
                'cm_per_px': self.cached_cm_per_px
            } if self.cached_scale_frame else None,
            'weight_params': {
                'breed': self.breed,
                'category': self.category,
                'age_range': self.age_range,
                'weight_multiplier': get_weight_multiplier(self.breed, self.category, self.age_range),
                'min_weight_kg': self.min_weight_kg,
                'max_weight_kg': self.max_weight_kg,
            }
        }

        return result
    
    def process_video_simple(self, video_path, sample_rate=3):
        """
        Versión simplificada: procesa cada N frames
        
        Args:
            video_path: Ruta al video
            sample_rate: Procesar 1 frame cada N frames (3 ≈ cada 0.1 segundos en video 30fps = 10 fps procesados)
        
        Returns:
            Resultados agregados
        """
        return self.process_video(video_path, frames_per_second=30/sample_rate if sample_rate > 0 else 1)

