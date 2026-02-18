from flask import Flask, redirect, url_for, request, render_template, jsonify, session
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from testing import ModelLoad, ImageScore
from weight_estimation import WeightEstimator
from depth_estimation import DepthEstimator
from breed_coefficients import BREED_OPTIONS, CATEGORY_OPTIONS, AGE_OPTIONS
from video_processor import VideoProcessor
import operator
import configparser
import tempfile
import uuid
from tensorflow.keras import backend as K
import base64
import cv2 
import numpy as np
import math

# HSV del rojo de referencia (rojo tiene dos rangos: 0-15 y 165-180)
# Más tolerante a sombras y variaciones de tono del rojo.
RED_HSV_LOWER1 = np.array([0, 60, 40])    # Rojo bajo
RED_HSV_UPPER1 = np.array([15, 255, 255])
RED_HSV_LOWER2 = np.array([165, 60, 40])  # Rojo alto
RED_HSV_UPPER2 = np.array([180, 255, 255])
# HSV del pasto para excluirlo
GRASS_HSV_LOWER = np.array([21, 38, 123])
GRASS_HSV_UPPER = np.array([30, 115, 255])


# Configuration loading
config = configparser.ConfigParser()
config.read("config.ini")

# Define a flask app
app = Flask(__name__)
app.secret_key = config["app"]["app-secret-key"]

# Umbral de confianza para detectar animales desconocidos
try:
    CONFIDENCE_THRESHOLD = float(config["detection"]["confidence_threshold"])
except (KeyError, ValueError):
    CONFIDENCE_THRESHOLD = 0.5  # Valor por defecto

# Inicializar estimador de peso con parámetros configurables para mejor detección
try:
    # ESCENARIO 1: Máxima Detección (Muchas Vacas, Condiciones Difíciles)
    # Parámetros YOLO: conf muy bajo = máxima detección (puede tener más falsos positivos)
    YOLO_CONF = 0.05          # Muy permisivo para detectar más vacas
    YOLO_IOU = 0.45           # Default, permite detecciones cercanas
    EYE_CONF_MULTIPLIER = 0.1  # 0.05 * 0.1 = 0.005 (muy permisivo para ojos)
    KEYPOINT_CONF_MULTIPLIER = 0.1  # 0.05 * 0.1 = 0.005 (muy permisivo para keypoints)
    weight_estimator = WeightEstimator(
        conf_threshold=YOLO_CONF, 
        iou_threshold=YOLO_IOU,
        eye_conf_multiplier=EYE_CONF_MULTIPLIER,
        keypoint_conf_multiplier=KEYPOINT_CONF_MULTIPLIER,
        use_postes_reference=True,
        poste1_height_cm=122,
        poste2_height_cm=122
    )
except Exception as e:
    print(f"Advertencia: No se pudieron cargar los modelos de estimación de peso: {e}")
    weight_estimator = None

# Inicializar detector de postes/stickers para referencia de escala
try:
    depth_estimator = DepthEstimator(
        conf_threshold=YOLO_CONF,
        iou_threshold=YOLO_IOU
    )
except Exception as e:
    print(f"Advertencia: No se pudo cargar el modelo de postes: {e}")
    depth_estimator = None

def get_file_path_and_save(request, is_video=False):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    if is_video:
        upload_dir = os.path.join(basepath, 'static', 'videos')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, secure_filename(f.filename))
    else:
        file_path = os.path.join(basepath, 'static/img', secure_filename(f.filename))
    f.save(file_path)
    return file_path

def loading_model(facility):
    """Cargamos el modelo fuera del testeo"""
    model_loaded = False
    global model
    # TensorFlow 2.x no requiere graph explícito, modo eager está habilitado por defecto
    model = ModelLoad(filepath=os.path.join('./checkpoints', facility, 'chckpt.best.h5')).model_loader()

    model_loaded = True

@app.route('/')
def index():
    # Por defecto limpiamos sesión por si no lanzamos testeo tras cargar el modelo al 
    # volver a la ruto "/" desde index.html.
    K.clear_session()
    # Leemos todas las granjas habilitadas en el sistema
    farms = next(os.walk(os.path.join(os.getcwd(), 'checkpoints')))[1]
    # Sólo nos quedamos con aquellas que tienen un modelo entrenado:
    # Nos ahorramos problemas en la vista choose.html
    data = list()
    for i in farms:
        for fname in os.listdir(os.path.join(os.getcwd(), 'checkpoints', i)):
            if fname.endswith('.h5'):
                data.append({"granja": i})
    return render_template('chooser.html', data=data)

@app.route('/load_model', methods=['GET', 'POST'])
def load_model():
    facility = request.form.get('comp_select')

    # Pasamos la granja elegida como elemento de sesión y 
    session['facility'] = facility
    # Cargamos el modelo para esa granja
    loading_model(facility)

    return render_template('index.html',
                         breed_options=BREED_OPTIONS,
                         category_options=CATEGORY_OPTIONS,
                         age_options=AGE_OPTIONS)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)

        # Obtener opciones del usuario
        enable_recognition = request.form.get('enable_recognition', 'true') == 'true'
        enable_weight = request.form.get('enable_weight', 'true') == 'true'
        scale_method = request.form.get('scale_method', 'both')  # 'both', 'eyes', 'poste'
        breed = request.form.get('breed', 'desconocido')
        category = request.form.get('category', 'desconocido')
        age_range = request.form.get('age_range', 'desconocido')

        result = {}

        # Verificar que el modelo esté cargado si se necesita reconocimiento
        if enable_recognition:
            try:
                # Intentar acceder a 'model' para verificar si está definido
                try:
                    current_model = model
                    if current_model is None:
                        raise NameError("model is None")
                except NameError:
                    # 'model' no está definido, intentar cargar desde sesión
                    facility = session.get('facility')
                    if facility:
                        loading_model(facility)
                    else:
                        result['recognition_error'] = 'No se ha seleccionado una granja. Por favor, selecciona una granja primero desde la página principal.'
                        enable_recognition = False
            except Exception as e:
                result['recognition_error'] = f'Error al cargar el modelo de reconocimiento: {str(e)}'
                enable_recognition = False
        
        # Reconocimiento de ganado
        if enable_recognition:
            try:
                # TensorFlow 2.x ejecuta en modo eager por defecto, no necesita graph.as_default()
                # version=2 para ResNet50/SENet50, version=1 para VGG16
                preds = ImageScore(model=model, 
                                img=file_path, 
                                farm=session['facility'],
                                version=2,
                                confidence_threshold=CONFIDENCE_THRESHOLD).scores()
        
                result['recognition'] = preds['predictions'].copy()
                result['recognition_metadata'] = preds['metadata']
                print(f"Predicción: {preds['metadata']}")
            except Exception as e:
                result['recognition_error'] = str(e)
                print(f"Error en reconocimiento: {e}")
        
        # Estimación de peso
        if enable_weight and weight_estimator:
            try:
                # estimate_weight ahora retorna (img_rgb, weight, details) cuando visualize=True
                # Pasar el método de escala seleccionado
                result_tuple = weight_estimator.estimate_weight(
                    file_path,
                    visualize=True,
                    debug=True,
                    debug_context="IMG_PREDICT",
                    scale_method=scale_method,
                    breed=breed,
                    category=category,
                    age_range=age_range
                )
                
                if isinstance(result_tuple, tuple) and len(result_tuple) >= 2:
                    weight_img = result_tuple[0]
                    weight = result_tuple[1]
                    details = result_tuple[2] if len(result_tuple) > 2 else None
                else:
                    # Fallback para compatibilidad
                    weight_img = result_tuple[0] if isinstance(result_tuple, tuple) else None
                    weight = result_tuple[1] if isinstance(result_tuple, tuple) and len(result_tuple) > 1 else result_tuple
                    details = None
                
                if weight is not None:
                    result['weight'] = round(weight, 2)
                else:
                    result['weight'] = None
                    # Usar mensaje detallado si está disponible
                    if details and isinstance(details, dict) and 'message' in details:
                        result['weight_error'] = details['message']
                    else:
                        result['weight_error'] = "No se pudieron detectar todos los puntos necesarios para calcular el peso"

                # Convertir imagen procesada a base64 para enviarla al frontend
                if weight_img is not None:
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(weight_img, cv2.COLOR_RGB2BGR))
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    result['weight_image'] = f"data:image/jpeg;base64,{img_base64}"
            except Exception as e:
                result['weight_error'] = str(e)
                print(f"Error en estimación de peso: {e}")
        elif enable_weight and not weight_estimator:
            result['weight_error'] = "Los modelos de estimación de peso no están disponibles"
        
        # Eliminamos la imagen que hemos cargado desde el navegador
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify(result)

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Procesa un video con múltiples vacas"""
    if request.method == 'POST':
        file_path = get_file_path_and_save(request, is_video=True)
        
        # Obtener opciones del usuario
        enable_recognition = request.form.get('enable_recognition', 'true') == 'true'
        enable_weight = request.form.get('enable_weight', 'true') == 'true'
        scale_method = request.form.get('scale_method', 'both')  # 'both', 'eyes', 'poste'
        breed = request.form.get('breed', 'desconocido')
        category = request.form.get('category', 'desconocido')
        age_range = request.form.get('age_range', 'desconocido')
        # sample_rate: procesar 1 frame cada N frames
        # Default: 1 = todos los frames (mejor para videos cortos)
        sample_rate = int(request.form.get('sample_rate', 1))
        debug_video = request.form.get('debug', 'false') == 'true'
        
        result = {
            'type': 'video',
            'cows': {}
        }
        
        # Verificar que el modelo esté cargado si se necesita reconocimiento
        recognition_model_to_use = None
        if enable_recognition:
            try:
                # Intentar acceder a 'model' para verificar si está definido
                try:
                    current_model = model
                    if current_model is not None:
                        recognition_model_to_use = current_model
                    else:
                        raise NameError("model is None")
                except NameError:
                    # 'model' no está definido, intentar cargar desde sesión
                    facility = session.get('facility')
                    if facility:
                        loading_model(facility)
                        recognition_model_to_use = model
                    else:
                        result['success'] = False
                        result['error'] = 'No se ha seleccionado una granja. Por favor, selecciona una granja primero desde la página principal.'
                        # Limpiar archivo de video antes de retornar
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except:
                            pass
                        return jsonify(result)
            except Exception as e:
                result['success'] = False
                result['error'] = f'Error al cargar el modelo de reconocimiento: {str(e)}'
                # Limpiar archivo de video antes de retornar
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass
                return jsonify(result)
        
        try:
            # Inicializar procesador de video con parámetros optimizados
            processor = VideoProcessor(
                recognition_model=recognition_model_to_use,
                weight_estimator=weight_estimator if enable_weight else None,
                farm=session.get('facility'),
                version=2,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                debug=debug_video,
                yolo_conf=YOLO_CONF,  # Usar mismo conf que WeightEstimator
                yolo_iou=YOLO_IOU,   # Usar mismo iou que WeightEstimator
                enhance_image=True,   # Mejorar contraste/brillo antes de detección
                breed=breed,
                category=category,
                age_range=age_range,
                scale_method=scale_method
            )
            
            # Procesar video (ahora retorna dict con 'cows' y 'stats')
            aggregated_results = processor.process_video_simple(file_path, sample_rate=sample_rate)
            
            # El resultado ahora incluye 'cows' y 'stats'
            if isinstance(aggregated_results, dict) and 'cows' in aggregated_results:
                result['cows'] = aggregated_results['cows']
                result['stats'] = aggregated_results.get('stats', {})
                result['total_cows'] = len(aggregated_results['cows'])
                result['scale_frame'] = aggregated_results.get('scale_frame')
                result['weight_params'] = aggregated_results.get('weight_params', {})

                # Diagnóstico: contar calibration_frames por vaca
                for _cid, _cdata in aggregated_results.get('cows', {}).items():
                    _cf_count = len(_cdata.get('calibration_frames', []))
                    _fm_count = len(_cdata.get('frame_measurements', []))
                    print(f"[CALIB-DIAG][RESPONSE] cow={_cid}: calibration_frames={_cf_count}, frame_measurements={_fm_count}")
            else:
                # Compatibilidad con formato antiguo
                result['cows'] = aggregated_results
                result['total_cows'] = len(aggregated_results) if isinstance(aggregated_results, dict) else 0
            
            result['success'] = True
            
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            print(f"Error procesando video: {e}")
        
        finally:
            # Limpiar archivo de video
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        
        return jsonify(result)

@app.route('/detect_reference_points', methods=['POST'])
def detect_reference_points():
    """
    Detecta los dos postes/stickers de referencia en una imagen
    y devuelve sus bounding boxes y una imagen con anotaciones.
    """
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)

        result = {
            'success': False,
            'poste1': None,
            'poste2': None
        }

        if depth_estimator is None:
            result['error'] = 'El detector de referencias (postes) no está disponible'
        else:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError('No se pudo leer la imagen')

                # DepthEstimator ahora incluye fallback por color/forma (no depende solo de YOLO)
                poste1_bbox, poste2_bbox = depth_estimator.detect_postes(image)

                def _bbox_to_dict(bbox):
                    x1, y1, x2, y2 = bbox
                    center = [(x1 + x2) / 2, (y1 + y2) / 2]
                    return {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float(center[0]), float(center[1])]
                    }

                if poste1_bbox is not None:
                    result['poste1'] = _bbox_to_dict(poste1_bbox)
                if poste2_bbox is not None:
                    result['poste2'] = _bbox_to_dict(poste2_bbox)

                # Dibujar anotaciones para visualizar
                annotated = image.copy()
                highlight_color = (0, 0, 255)  # Rojo (BGR)
                line_color = (0, 0, 255)  # Rojo (BGR)
                def _draw_poste_with_yellow_line(bbox, label):
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), highlight_color, 2)

                    # Buscar el tramo rojo real dentro del bbox
                    roi = annotated[y1:y2, x1:x2]
                    if roi.size > 0:
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        # Rojo tiene dos rangos HSV, combinarlos
                        red_mask1 = cv2.inRange(hsv_roi, RED_HSV_LOWER1, RED_HSV_UPPER1)
                        red_mask2 = cv2.inRange(hsv_roi, RED_HSV_LOWER2, RED_HSV_UPPER2)
                        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                        grass_mask = cv2.inRange(hsv_roi, GRASS_HSV_LOWER, GRASS_HSV_UPPER)
                        red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(grass_mask))
                        kernel = np.ones((3, 3), np.uint8)
                        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

                        # Overlay para visualizar qué píxeles cuentan como rojo
                        overlay = roi.copy()
                        overlay[red_mask > 0] = (0, 0, 255)  # Rojo
                        roi_blend = cv2.addWeighted(roi, 0.7, overlay, 0.3, 0)
                        annotated[y1:y2, x1:x2] = roi_blend

                        # Usar columnas centrales para evitar bordes
                        col_start = int(red_mask.shape[1] * 0.4)
                        col_end = int(red_mask.shape[1] * 0.6)
                        center_strip = red_mask[:, col_start:col_end]

                        # Ratio de rojo por fila (evita puntos sueltos)
                        row_ratio = np.mean(center_strip > 0, axis=1)
                        rows_with_red = np.where(row_ratio >= 0.2)[0]

                        if rows_with_red.size > 0:
                            # Buscar el tramo rojo continuo empezando desde abajo
                            y_bottom = int(rows_with_red.max())
                            min_gap = 4  # tolerancia de filas sin rojo
                            gap_count = 0
                            y_top = y_bottom

                            for row in range(y_bottom, -1, -1):
                                if row_ratio[row] >= 0.2:
                                    y_top = row
                                    gap_count = 0
                                else:
                                    gap_count += 1
                                    if gap_count >= min_gap:
                                        break

                            line_y1 = y1 + y_top
                            line_y2 = y1 + y_bottom
                        else:
                            # Fallback al bbox completo si no hay rojo detectado
                            line_y1, line_y2 = y1, y2
                    else:
                        line_y1, line_y2 = y1, y2

                    # Línea roja de medición (altura real roja)
                    cx = int((x1 + x2) / 2)
                    if line_y2 - line_y1 < 3:
                        line_y1 = max(y1, line_y1 - 2)
                        line_y2 = min(y2, line_y2 + 2)
                    # Dibujo doble para que siempre se vea (borde blanco + rojo)
                    cv2.line(annotated, (cx, line_y1), (cx, line_y2), (255, 255, 255), 4)
                    cv2.line(annotated, (cx, line_y1), (cx, line_y2), line_color, 3)
                    # Marcar extremos
                    cv2.circle(annotated, (cx, line_y1), 4, line_color, -1)
                    cv2.circle(annotated, (cx, line_y2), 4, line_color, -1)
                    height_px = abs(line_y2 - line_y1)
                    cv2.putText(annotated, f'{label} ({height_px}px)', (x1, max(0, y1 - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, highlight_color, 2)

                if poste1_bbox is not None:
                    _draw_poste_with_yellow_line(poste1_bbox, 'POSTE 1')
                if poste2_bbox is not None:
                    _draw_poste_with_yellow_line(poste2_bbox, 'POSTE 2')

                # Convertir imagen anotada a base64
                _, buffer = cv2.imencode('.jpg', annotated)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                result['image'] = f"data:image/jpeg;base64,{img_base64}"

                if poste1_bbox is not None and poste2_bbox is not None:
                    result['success'] = True
                else:
                    result['error'] = 'No se detectaron ambos postes en la imagen'
            except Exception as e:
                result['error'] = str(e)
            finally:
                try:
                    os.remove(file_path)
                except:
                    pass

        return jsonify(result)

@app.route('/detect_posts_all', methods=['POST'])
def detect_posts_all():
    """
    Detecta todos los postes del alambrado en una imagen y los marca en rojo.
    """
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)

        result = {
            'success': False,
            'count': 0,
            'boxes': []
        }

        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError('No se pudo leer la imagen')

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
            )

            # Resaltar estructuras verticales
            vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 35))
            vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=1)

            contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            h, w = image.shape[:2]
            boxes = []
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), angle = rect
                if rw == 0 or rh == 0:
                    continue

                length = max(rw, rh)
                thickness = min(rw, rh)

                if length < h * 0.10:
                    continue
                if thickness > w * 0.12:
                    continue
                aspect = length / max(thickness, 1.0)
                if aspect < 2.0:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw > w * 0.3:
                    continue
                boxes.append([x, y, x + bw, y + bh])

            # Fusionar cajas cercanas en X (mismo poste)
            boxes = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)
            merged = []
            for b in boxes:
                if not merged:
                    merged.append(b)
                    continue
                px = (merged[-1][0] + merged[-1][2]) / 2
                cx = (b[0] + b[2]) / 2
                if abs(cx - px) < 8:
                    # Unir cajas
                    merged[-1] = [
                        min(merged[-1][0], b[0]),
                        min(merged[-1][1], b[1]),
                        max(merged[-1][2], b[2]),
                        max(merged[-1][3], b[3])
                    ]
                else:
                    merged.append(b)

            # Dividir cajas muy anchas (probable unión entre postes)
            split_boxes = []
            for b in merged:
                x1, y1, x2, y2 = map(int, b)
                bw = x2 - x1
                if bw > 40:
                    roi_edges = vertical[y1:y2, x1:x2]
                    if roi_edges.size == 0:
                        split_boxes.append(b)
                        continue
                    col_sum = np.sum(roi_edges > 0, axis=0)
                    threshold = max(2, int(0.05 * (y2 - y1)))
                    strong_cols = np.where(col_sum >= threshold)[0]
                    if strong_cols.size == 0:
                        split_boxes.append(b)
                        continue
                    # Agrupar columnas fuertes en segmentos
                    segments = []
                    start = strong_cols[0]
                    prev = strong_cols[0]
                    for c in strong_cols[1:]:
                        if c - prev > 3:
                            segments.append((start, prev))
                            start = c
                        prev = c
                    segments.append((start, prev))

                    for s in segments:
                        sx1 = x1 + s[0]
                        sx2 = x1 + s[1]
                        if sx2 - sx1 < 6:
                            continue
                        split_boxes.append([sx1, y1, sx2, y2])
                else:
                    split_boxes.append(b)

            annotated = image.copy()
            for b in split_boxes:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

                roi_edges = vertical[y1:y2, x1:x2]
                if roi_edges.size > 0:
                    lines = cv2.HoughLinesP(
                        roi_edges,
                        rho=1,
                        theta=np.pi / 180,
                        threshold=30,
                        minLineLength=max(20, int((y2 - y1) * 0.4)),
                        maxLineGap=10
                    )
                else:
                    lines = None

                if lines is not None and len(lines) > 0:
                    # Elegir la línea más larga (vector dirección)
                    best = None
                    best_len = 0
                    for line in lines:
                        lx1, ly1, lx2, ly2 = line[0]
                        length = math.sqrt((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2)
                        if length > best_len:
                            best_len = length
                            best = (lx1, ly1, lx2, ly2)
                    if best:
                        lx1, ly1, lx2, ly2 = best
                        cv2.line(annotated, (x1 + lx1, y1 + ly1), (x1 + lx2, y1 + ly2), (0, 0, 255), 3)
                else:
                    # Fallback: línea vertical en el centro del bbox
                    cx = int((x1 + x2) / 2)
                    cv2.line(annotated, (cx, y1), (cx, y2), (0, 0, 255), 3)

            _, buffer = cv2.imencode('.jpg', annotated)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            result['image'] = f"data:image/jpeg;base64,{img_base64}"
            result['boxes'] = [{'bbox': [float(x1), float(y1), float(x2), float(y2)]} for x1, y1, x2, y2 in split_boxes]
            result['count'] = len(split_boxes)
            result['success'] = True
        except Exception as e:
            result['error'] = str(e)
        finally:
            try:
                os.remove(file_path)
            except:
                pass

        return jsonify(result)
        
@app.route('/video')
def video_analysis():
    """Manual video frame analysis page - no model loading needed"""
    return render_template('index.html',
                         breed_options=BREED_OPTIONS,
                         category_options=CATEGORY_OPTIONS,
                         age_options=AGE_OPTIONS)

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """Analyze a single frame from the video player canvas"""
    file = request.files.get('frame')
    if not file:
        return jsonify({'success': False, 'error': 'No frame provided'})

    breed = request.form.get('breed', 'desconocido')
    category = request.form.get('category', 'desconocido')
    age_range = request.form.get('age_range', 'desconocido')

    # Save to temp file
    temp_path = os.path.join(tempfile.gettempdir(), f'frame_{uuid.uuid4().hex}.jpg')
    file.save(temp_path)

    try:
        if not weight_estimator:
            return jsonify({'success': False, 'error': 'Weight estimator not available'})

        # Call estimate_weight with all return options
        result_tuple = weight_estimator.estimate_weight(
            temp_path,
            visualize=True,
            debug=True,
            debug_context="FRAME_ANALYZE",
            return_eye_coords=True,
            return_keypoint_coords=True,
            scale_method='poste',
            breed=breed,
            category=category,
            age_range=age_range,
        )

        # Unpack 5-tuple: (img_rgb, weight, eye_coords, kp_coords, details)
        img_rgb = result_tuple[0]
        weight = result_tuple[1]
        eye_coords = result_tuple[2]
        kp_coords = result_tuple[3]
        details = result_tuple[4] if len(result_tuple) > 4 else {}

        # Encode annotated image as base64
        annotated_b64 = None
        if img_rgb is not None:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', img_bgr)
            annotated_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

        # Extract dist1_px and dist2_px from keypoint_coords or details
        dist1_px = None
        dist2_px = None
        if kp_coords and len(kp_coords) > 0:
            last_kp = kp_coords[-1]
            if isinstance(last_kp, dict):
                dist1_px = last_kp.get('dist1_px')
                dist2_px = last_kp.get('dist2_px')
        if dist1_px is None and isinstance(details, dict):
            dist1_px = details.get('dist1_px')
        if dist2_px is None and isinstance(details, dict):
            dist2_px = details.get('dist2_px')

        # Calculate cow_height_cm: needs 2 posts (for scale) + cow detected (for height)
        cm_per_px = details.get('cm_per_px') if isinstance(details, dict) else None
        animal_bbox_height_px = details.get('animal_bbox_height_px') if isinstance(details, dict) else None
        postes_heights = details.get('postes_heights_px', []) if isinstance(details, dict) else []
        cow_height_cm = None

        if len(postes_heights) >= 2 and animal_bbox_height_px:
            avg_post_height_px = sum(postes_heights) / len(postes_heights)
            cm_per_px = 122.0 / avg_post_height_px
            cow_height_cm = animal_bbox_height_px * cm_per_px
            print(f"[FRAME_ANALYZE] Using avg post height: {postes_heights} -> avg={avg_post_height_px:.1f}px -> cm_per_px={cm_per_px:.5f} -> cow_height={cow_height_cm:.1f}cm")

        postes_detected = details.get('postes_detected', 0) if isinstance(details, dict) else 0
        message = details.get('message', '') if isinstance(details, dict) else ''

        # Determine keypoints_found from kp_coords (more reliable than details dict,
        # which may not include this field in _no_scale_details path)
        keypoints_found = False
        if kp_coords and len(kp_coords) > 0:
            last_kp = kp_coords[-1]
            if isinstance(last_kp, dict):
                keypoints_found = bool(last_kp.get('keypoints_accepted', False))
        if not keypoints_found and isinstance(details, dict):
            keypoints_found = bool(details.get('keypoints_found', False))
        # Final fallback: if we have both dist1_px and dist2_px, keypoints were found
        if not keypoints_found and dist1_px is not None and dist2_px is not None:
            keypoints_found = True

        return jsonify({
            'success': True,
            'annotated_image': annotated_b64,
            'weight_kg': round(weight, 2) if weight else None,
            'details': {
                'cm_per_px': cm_per_px,
                'animal_bbox_height_px': animal_bbox_height_px,
                'postes_detected': postes_detected,
                'postes_heights_px': postes_heights,
                'cow_height_cm': round(cow_height_cm, 2) if cow_height_cm else None,
                'dist1_px': dist1_px,
                'dist2_px': dist2_px,
                'message': message,
                'scale_from_postes': details.get('scale_from_postes', False) if isinstance(details, dict) else False,
                'keypoints_found': keypoints_found,
                'has_eyes': details.get('has_eyes', False) if isinstance(details, dict) else False,
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        try:
            os.remove(temp_path)
        except:
            pass

def to_farm():
    session.pop('facility')
    return redirect(url_for('index'))

# start the server with the 'run()' method
if __name__ == '__main__':
    # Usar puerto 5001 para evitar conflicto con AirPlay en macOS
    # 0.0.0.0 evita problemas de resolución con localhost
    app.run(host='0.0.0.0', port=5001, debug=True)
    # app.run(debug=True)

# TODO: atrás para otra granja
# TODO: Página de despedida
