"""
Estimación de profundidad y escala usando postes de referencia.

Clase principal: DepthEstimator
  - Detecta postes rojos en la imagen usando YOLO (sticker.pt) y fallback por color HSV
  - Mide la altura en píxeles del tramo rojo de cada poste
  - Calcula la escala cm/px usando la altura real conocida (por defecto 122 cm)
  - Opcionalmente calcula profundidad relativa animal-postes con triangulación

Dependencias: ultralytics, opencv-python, numpy
"""
import cv2
import numpy as np
import math
import os
from ultralytics import YOLO


class DepthEstimator:
    """
    Calcula profundidad usando dos postes como referencia mediante triangulación.
    
    Con dos postes de altura conocida y distancia conocida entre ellos,
    puede calcular la profundidad del animal y ajustar la escala correctamente.
    """
    
    def __init__(self, sticker_model_path="models_yolo/sticker.pt",
                 poste1_height_cm=100, poste2_height_cm=100,
                 distancia_postes_cm=200, focal_length_px=None,
                 conf_threshold=0.25, iou_threshold=0.45,
                 yellow_ratio_threshold=0.08, yellow_ratio_threshold_high=0.18,
                 yellow_global_threshold=0.12):
        """
        Inicializa el estimador de profundidad con dos postes.
        
        Args:
            sticker_model_path: Ruta al modelo YOLO para detectar postes/stickers
            poste1_height_cm: Altura real del poste 1 en cm
            poste2_height_cm: Altura real del poste 2 en cm
            distancia_postes_cm: Distancia real entre los dos postes en cm
            focal_length_px: Longitud focal de la cámara en píxeles (None para estimar)
            conf_threshold: Umbral de confianza para detección de postes
            iou_threshold: Umbral de IoU para NMS
        """
        self.sticker_model = YOLO(sticker_model_path) if os.path.exists(sticker_model_path) else None
        self.poste1_height_cm = poste1_height_cm
        self.poste2_height_cm = poste2_height_cm
        self.distancia_postes_cm = distancia_postes_cm
        self.focal_length_px = focal_length_px
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # Filtros por color (rojo)
        self.yellow_ratio_threshold = yellow_ratio_threshold  # Mantener nombre por compatibilidad
        self.yellow_ratio_threshold_high = yellow_ratio_threshold_high
        self.yellow_global_threshold = yellow_global_threshold

        # Rangos HSV para rojo (dos bandas). Más tolerantes para sombra.
        # Nota: mantenemos el nombre "yellow" por compatibilidad con el resto del código.
        self._red_hsv_lower1 = np.array([0, 60, 40])
        self._red_hsv_upper1 = np.array([15, 255, 255])
        self._red_hsv_lower2 = np.array([165, 60, 40])
        self._red_hsv_upper2 = np.array([180, 255, 255])

        # HSV del pasto para excluirlo (evita falsos positivos en verde)
        self._grass_hsv_lower = np.array([21, 38, 123])
        self._grass_hsv_upper = np.array([30, 115, 255])
    
    @staticmethod
    def euclidean(pt1, pt2):
        """Calcula la distancia euclidiana entre dos puntos"""
        return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    @staticmethod
    def calcular_centro_bbox(bbox):
        """Calcula el centro de un bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def detect_postes(self, image):
        """
        Detecta ambos postes en la imagen usando el modelo YOLO.
        
        Args:
            image: Imagen numpy array (BGR)
            
        Returns:
            (poste1_bbox, poste2_bbox) o (None, None) si no se detectan ambos
        """
        postes = self.detect_postes_all(image)

        if len(postes) >= 2:
            # Elegir el par "más separado" en X entre los mejores candidatos.
            # Esto ayuda cuando hay varios fragmentos del mismo poste.
            postes.sort(key=lambda p: (p.get('score', 0.0), p.get('yellow_ratio', 0.0)), reverse=True)
            top = postes[: min(8, len(postes))]

            best_pair = None
            best_value = -1.0
            w = float(image.shape[1]) if image is not None and image.size > 0 else 1.0

            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    ci = self.calcular_centro_bbox(top[i]['bbox'])
                    cj = self.calcular_centro_bbox(top[j]['bbox'])
                    sep_norm = abs(ci[0] - cj[0]) / max(1.0, w)
                    score_mean = (float(top[i].get('score', 0.0)) + float(top[j].get('score', 0.0))) / 2.0
                    red_mean = (float(top[i].get('yellow_ratio', 0.0)) + float(top[j].get('yellow_ratio', 0.0))) / 2.0
                    value = 0.55 * score_mean + 0.25 * red_mean + 0.20 * sep_norm
                    if value > best_value:
                        best_value = value
                        best_pair = (top[i], top[j])

            if best_pair is not None:
                return best_pair[0]['bbox'], best_pair[1]['bbox']

            # Fallback simple
            return top[0]['bbox'], top[1]['bbox']
        elif len(postes) == 1:
            # Solo un poste detectado
            return postes[0]['bbox'], None

        return None, None

    def detect_postes_all(self, image):
        """
        Detecta todos los postes en la imagen usando el modelo YOLO.

        Returns:
            Lista de dicts: [{'bbox': [...], 'score': float, 'yellow_ratio': float}]
        """
        postes = []

        # 1) Intentar YOLO si está disponible
        results = None
        if self.sticker_model is not None:
            results = self.sticker_model(image, save=False, conf=self.conf_threshold, iou=self.iou_threshold)

        # Detectar cuánto rojo hay en la imagen completa para ajustar umbral
        hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Rojo tiene dos rangos HSV, combinarlos (tolerante a sombra)
        lower_red1, upper_red1 = self._red_hsv_lower1, self._red_hsv_upper1
        lower_red2, upper_red2 = self._red_hsv_lower2, self._red_hsv_upper2
        grass_lower, grass_upper = self._grass_hsv_lower, self._grass_hsv_upper
        red_mask_full1 = cv2.inRange(hsv_full, lower_red1, upper_red1)
        red_mask_full2 = cv2.inRange(hsv_full, lower_red2, upper_red2)
        red_mask_full = cv2.bitwise_or(red_mask_full1, red_mask_full2)
        grass_mask_full = cv2.inRange(hsv_full, grass_lower, grass_upper)
        red_mask_full = cv2.bitwise_and(red_mask_full, cv2.bitwise_not(grass_mask_full))
        global_red_ratio = float(np.mean(red_mask_full > 0))
        # Umbral más permisivo: usar el umbral bajo por defecto, solo usar el alto si hay MUCHO rojo en la imagen
        yellow_ratio_threshold = self.yellow_ratio_threshold_high if global_red_ratio > self.yellow_global_threshold * 2 else self.yellow_ratio_threshold
        # Reducir aún más el umbral mínimo para detectar bandas rojas pequeñas pero puras
        yellow_ratio_threshold = max(0.03, yellow_ratio_threshold * 0.6)  # Mínimo 3% de rojo, o 60% del umbral original

        if results is not None:
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = map(int, box.tolist())
                        # Limitar bbox a la imagen
                        x1 = max(0, min(x1, image.shape[1] - 1))
                        x2 = max(0, min(x2, image.shape[1]))
                        y1 = max(0, min(y1, image.shape[0] - 1))
                        y2 = max(0, min(y2, image.shape[0]))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Validar proporción de rojo dentro del bbox
                        roi = image[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        red_mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
                        red_mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
                        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                        grass_mask = cv2.inRange(hsv_roi, grass_lower, grass_upper)
                        red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(grass_mask))
                        red_ratio = float(np.mean(red_mask > 0))

                        bbox_area = (x2 - x1) * (y2 - y1)
                        img_area = image.shape[0] * image.shape[1]
                        bbox_ratio = bbox_area / img_area if img_area > 0 else 0

                        adjusted_threshold = yellow_ratio_threshold
                        if score > 0.5 and bbox_ratio < 0.1:
                            adjusted_threshold = yellow_ratio_threshold * 0.5

                        if red_ratio < adjusted_threshold:
                            if red_ratio > adjusted_threshold * 0.7:
                                print(
                                    f"[DEPTH] Poste YOLO rechazado por rojo: "
                                    f"bbox={[x1, y1, x2, y2]} score={score:.3f} red_ratio={red_ratio:.3f} threshold={adjusted_threshold:.3f}"
                                )
                            continue

                        postes.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'score': float(score),
                            'yellow_ratio': red_ratio  # Mantener nombre por compatibilidad
                        })

        # 2) Fallback por color (especialmente útil cuando YOLO falla o no existe)
        # Si YOLO ya detectó suficientes, aún añadimos color pero con NMS para evitar duplicados.
        postes_color = self._detect_postes_by_color(image)
        if postes_color:
            postes.extend(postes_color)

        # 3) Deduplicar con NMS/IoU simple
        postes = self._nms_postes(postes, iou_threshold=0.35)
        postes.sort(key=lambda p: (p.get('score', 0.0), p.get('yellow_ratio', 0.0)), reverse=True)
        return postes

    def _detect_postes_by_color(self, image):
        """
        Detecta candidatos a "postes rojos" solo por color/forma (HSV + contornos).
        Devuelve dicts compatibles: {'bbox': [x1,y1,x2,y2], 'score': float, 'yellow_ratio': float}
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, self._red_hsv_lower1, self._red_hsv_upper1)
        red2 = cv2.inRange(hsv, self._red_hsv_lower2, self._red_hsv_upper2)
        red_mask = cv2.bitwise_or(red1, red2)

        grass = cv2.inRange(hsv, self._grass_hsv_lower, self._grass_hsv_upper)
        red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(grass))

        # Morfología: unir segmentos verticales y limpiar ruido
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        img_area = float(h * w)
        min_height = max(40, int(h * 0.18))
        min_area = max(600.0, img_area * 0.0008)  # ~0.08% del área

        postes = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh < min_height or bw <= 0:
                continue

            aspect = float(bh) / float(max(1, bw))
            if aspect < 1.6:
                continue

            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            roi = red_mask[y1:y2, x1:x2]
            red_ratio = float(np.mean(roi > 0)) if roi.size > 0 else 0.0

            # Para postes con tira roja fina, el ratio puede ser bajo si bbox captura parte negra.
            # Aceptar ratios modestos si el candidato es alto y angosto.
            if red_ratio < 0.04 and aspect < 3.0:
                continue

            area_ratio = (float((x2 - x1) * (y2 - y1)) / max(1.0, img_area))
            # Score heurístico (0-1): mezcla de "rojez" y tamaño relativo.
            score = float(np.clip(0.75 * red_ratio + 0.25 * min(1.0, area_ratio * 6.0), 0.0, 1.0))

            postes.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'score': score,
                'yellow_ratio': red_ratio
            })

        return postes

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        denom = area_a + area_b - inter
        return float(inter / denom) if denom > 0 else 0.0

    def _nms_postes(self, postes, iou_threshold=0.35):
        if not postes:
            return []
        postes_sorted = sorted(postes, key=lambda p: (p.get('score', 0.0), p.get('yellow_ratio', 0.0)), reverse=True)
        kept = []
        for p in postes_sorted:
            if all(self._bbox_iou(p['bbox'], k['bbox']) < iou_threshold for k in kept):
                kept.append(p)
        return kept
    
    def calcular_profundidad_postes(self, poste1_bbox, poste2_bbox, image_shape=None):
        """
        Calcula la profundidad del plano donde están los postes.
        
        Usa la altura de los postes en píxeles y sus alturas reales conocidas
        para calcular la profundidad usando la fórmula:
        profundidad = (altura_real * focal_length) / altura_px
        
        Args:
            poste1_bbox: Bounding box del poste 1 [x1, y1, x2, y2]
            poste2_bbox: Bounding box del poste 2 [x1, y1, x2, y2]
            image_shape: (height, width) de la imagen para estimar focal_length si no está definido
            
        Returns:
            Profundidad promedio de los postes en cm, o None si no se puede calcular
        """
        if poste1_bbox is None or poste2_bbox is None:
            return None
        
        # Calcular altura de cada poste en píxeles
        poste1_height_px = abs(poste1_bbox[3] - poste1_bbox[1])  # y2 - y1
        poste2_height_px = abs(poste2_bbox[3] - poste2_bbox[1])
        
        # Estimar focal_length si no está definido
        focal_length = self.focal_length_px
        if focal_length is None and image_shape is not None:
            # Estimación simple: focal_length ≈ width (asumiendo cámara estándar)
            focal_length = image_shape[1] * 0.7  # Factor empírico
        
        if focal_length is None:
            # Método alternativo: usar distancia entre postes
            poste1_center = self.calcular_centro_bbox(poste1_bbox)
            poste2_center = self.calcular_centro_bbox(poste2_bbox)
            distancia_postes_px = self.euclidean(poste1_center, poste2_center)
            
            if distancia_postes_px > 0:
                # Estimar profundidad usando proporción
                # Asumiendo que los postes están en el mismo plano
                factor_escala = self.distancia_postes_cm / distancia_postes_px
                # Profundidad estimada (requiere calibración inicial)
                profundidad_estimada = factor_escala * 100  # Factor empírico, ajustar según calibración
                return profundidad_estimada
            return None
        
        # Calcular profundidad usando altura de postes
        profundidad1 = (self.poste1_height_cm * focal_length) / poste1_height_px if poste1_height_px > 0 else None
        profundidad2 = (self.poste2_height_cm * focal_length) / poste2_height_px if poste2_height_px > 0 else None
        
        if profundidad1 is not None and profundidad2 is not None:
            return (profundidad1 + profundidad2) / 2
        elif profundidad1 is not None:
            return profundidad1
        elif profundidad2 is not None:
            return profundidad2
        
        return None
    
    def calcular_profundidad_animal(self, animal_bbox, profundidad_postes, animal_height_expected_cm=120):
        """
        Calcula la profundidad del animal usando la profundidad de los postes como referencia.
        
        En el setup típico, el animal camina ENTRE los dos postes, por lo que está
        aproximadamente en el mismo plano de profundidad.  Sin una focal_length_px
        calibrada no hay forma confiable de estimar una diferencia de profundidad,
        así que asumimos que el animal está en el mismo plano que los postes
        (factor_profundidad = 1.0).
        
        Args:
            animal_bbox: Bounding box del animal [x1, y1, x2, y2]
            profundidad_postes: Profundidad calculada de los postes en cm
            animal_height_expected_cm: Altura esperada del animal en cm (ajustar según raza)
            
        Returns:
            Profundidad del animal en cm, o None si no se puede calcular
        """
        if animal_bbox is None or profundidad_postes is None:
            return None
        
        animal_height_px = abs(animal_bbox[3] - animal_bbox[1])  # y2 - y1
        
        if animal_height_px == 0:
            return None
        
        if self.focal_length_px:
            # Método preciso con focal_length calibrada
            profundidad_animal = (animal_height_expected_cm * self.focal_length_px) / animal_height_px
        else:
            # Sin focal_length calibrada, asumir que el animal está en el mismo plano
            # que los postes (setup estándar: el animal pasa entre los postes).
            # Anteriormente se usaba un factor hardcodeado de 1.2 que inflaba la
            # escala incorrectamente (~47% de error acumulado en peso).
            profundidad_animal = profundidad_postes  # mismo plano
        
        return profundidad_animal
    
    def calcular_escala_ajustada(self, poste_bbox, profundidad_postes, profundidad_animal,
                                 measured_height_px=None):
        """
        Calcula la escala ajustada (cm por píxel) según la profundidad calculada.
        
        Args:
            poste_bbox: Bounding box de uno de los postes
            profundidad_postes: Profundidad de los postes en cm
            profundidad_animal: Profundidad del animal en cm
            measured_height_px: Altura medida de la marca roja en px (más preciso
                                que la altura del bbox completo). Si se proporciona,
                                se usa en lugar de la altura del bbox.
            
        Returns:
            Escala en cm/píxel ajustada para la profundidad del animal
        """
        if poste_bbox is None or profundidad_postes is None or profundidad_animal is None:
            return None
        
        # Usar altura medida de la marca roja si está disponible;
        # de lo contrario, usar la altura del bbox completo (menos preciso).
        if measured_height_px and measured_height_px > 0:
            poste_height_px = measured_height_px
        else:
            poste_height_px = abs(poste_bbox[3] - poste_bbox[1])
        
        if poste_height_px == 0:
            return None
        
        # Escala base usando el poste
        altura_poste_cm = (self.poste1_height_cm + self.poste2_height_cm) / 2
        escala_poste = altura_poste_cm / poste_height_px
        
        # Ajustar escala según profundidad
        # Si el animal está más lejos, necesita escala mayor (más cm por píxel)
        factor_profundidad = profundidad_animal / profundidad_postes
        escala_animal = escala_poste * factor_profundidad
        
        return escala_animal
    
    def estimar_escala_con_postes(self, image, animal_bbox=None, debug=False,
                                  measured_height_px=None):
        """
        Método principal: detecta postes y calcula escala ajustada.
        
        Args:
            image: Imagen numpy array (BGR)
            animal_bbox: Bounding box del animal [x1, y1, x2, y2] (opcional)
            debug: Si True, imprime información de debug
            measured_height_px: Altura medida de la marca roja del poste en px.
                                Si se proporciona, se usa para mayor precisión en la escala.
            
        Returns:
            dict con:
                - 'escala': Escala en cm/píxel (None si no se puede calcular)
                - 'profundidad_postes': Profundidad de postes en cm
                - 'profundidad_animal': Profundidad del animal en cm (si animal_bbox proporcionado)
                - 'poste1_bbox': Bounding box del poste 1
                - 'poste2_bbox': Bounding box del poste 2
        """
        resultado = {
            'escala': None,
            'profundidad_postes': None,
            'profundidad_animal': None,
            'poste1_bbox': None,
            'poste2_bbox': None
        }
        
        # Detectar postes
        poste1_bbox, poste2_bbox = self.detect_postes(image)
        resultado['poste1_bbox'] = poste1_bbox
        resultado['poste2_bbox'] = poste2_bbox
        
        if poste1_bbox is None or poste2_bbox is None:
            if debug:
                print(f"[DEPTH] No se detectaron ambos postes. Poste1: {poste1_bbox is not None}, Poste2: {poste2_bbox is not None}")
            return resultado
        
        if debug:
            print(f"[DEPTH] Postes detectados: Poste1={poste1_bbox}, Poste2={poste2_bbox}")
        
        # Calcular profundidad de postes
        image_shape = image.shape[:2]  # (height, width)
        profundidad_postes = self.calcular_profundidad_postes(poste1_bbox, poste2_bbox, image_shape)
        resultado['profundidad_postes'] = profundidad_postes
        
        if profundidad_postes is None:
            if debug:
                print("[DEPTH] No se pudo calcular profundidad de postes")
            return resultado
        
        if debug:
            print(f"[DEPTH] Profundidad postes: {profundidad_postes:.2f} cm")
        
        # Calcular profundidad del animal si se proporciona bbox
        if animal_bbox is not None:
            profundidad_animal = self.calcular_profundidad_animal(animal_bbox, profundidad_postes)
            resultado['profundidad_animal'] = profundidad_animal
            
            if profundidad_animal is not None:
                if debug:
                    print(f"[DEPTH] Profundidad animal: {profundidad_animal:.2f} cm")
                
                # Calcular escala ajustada (usar measured_height_px si disponible)
                escala = self.calcular_escala_ajustada(
                    poste1_bbox, profundidad_postes, profundidad_animal,
                    measured_height_px=measured_height_px
                )
                resultado['escala'] = escala
                
                if debug and escala is not None:
                    print(f"[DEPTH] Escala ajustada: {escala:.4f} cm/píxel")
        else:
            # Sin animal_bbox, calcular escala base usando postes
            altura_poste_cm = (self.poste1_height_cm + self.poste2_height_cm) / 2
            poste_height_px = abs(poste1_bbox[3] - poste1_bbox[1])
            if poste_height_px > 0:
                resultado['escala'] = altura_poste_cm / poste_height_px
                if debug:
                    print(f"[DEPTH] Escala base (postes): {resultado['escala']:.4f} cm/píxel")
        
        return resultado

