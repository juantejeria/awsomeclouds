from flask import Flask, redirect, url_for, request, render_template, jsonify, session, Response, stream_with_context
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from testing import ModelLoad, ImageScore
from weight_estimation import WeightEstimator
from depth_estimation import DepthEstimator
from breed_coefficients import BREED_OPTIONS, CATEGORY_OPTIONS, AGE_OPTIONS, get_weight_range, get_estimated_height, HEIGHT_BY_CATEGORY_AGE
from video_processor import VideoProcessor
from generar_modelos3d_batch import procesar_frame, filtrar_outliers, guardar_ply, detectar_vaca, segmentar, recortar_torso
from reconstruccion_3d import sfm_desde_frames, sfm_real_desde_frames, modelo_hibrido, guardar_ply_con_malla, generar_imagen_resumen
import operator
import configparser
import tempfile
import uuid
import time as _time
from tensorflow.keras import backend as K
import base64
import cv2 
import numpy as np
import math
import json
import statistics


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Override json.dumps for SSE events to always use NumpyEncoder
_json_dumps_original = json.dumps
def _json_dumps_safe(*args, **kwargs):
    kwargs.setdefault('cls', NumpyEncoder)
    return _json_dumps_original(*args, **kwargs)
json.dumps = _json_dumps_safe

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
        poste1_height_cm=50,
        poste2_height_cm=50,
        use_monocular_depth=True
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

# ── Frame cache for two-phase scan/analyze flow ──
_frame_cache = {}  # UUID → {'path': str, 'created': float}
_FRAME_CACHE_TTL = 300  # 5 minutes

def _cache_frame(temp_path):
    """Store a frame path in cache and return its UUID."""
    frame_id = uuid.uuid4().hex
    _frame_cache[frame_id] = {'path': temp_path, 'created': _time.time()}
    return frame_id

def _get_cached_frame(frame_id):
    """Return cached frame path or None if expired/missing."""
    _cleanup_frame_cache()
    entry = _frame_cache.get(frame_id)
    if entry and os.path.exists(entry['path']):
        return entry['path']
    return None

def _cleanup_frame_cache():
    """Remove entries older than TTL and delete their files."""
    now = _time.time()
    expired = [fid for fid, e in _frame_cache.items() if now - e['created'] > _FRAME_CACHE_TTL]
    for fid in expired:
        try:
            path = _frame_cache[fid]['path']
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        del _frame_cache[fid]

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

@app.route('/video')
def video_analysis():
    """Manual video frame analysis page - no model loading needed"""
    return render_template('index.html',
                         breed_options=BREED_OPTIONS,
                         category_options=CATEGORY_OPTIONS,
                         age_options=AGE_OPTIONS)

@app.route('/scan_frame', methods=['POST'])
def scan_frame():
    """Phase 1: Run detection only, return all cows (with thumbnails) and posts."""
    file = request.files.get('frame')
    if not file:
        return jsonify({'success': False, 'error': 'No frame provided'})

    temp_path = os.path.join(tempfile.gettempdir(), f'frame_{uuid.uuid4().hex}.jpg')
    file.save(temp_path)

    try:
        if not weight_estimator:
            os.remove(temp_path)
            return jsonify({'success': False, 'error': 'Weight estimator not available'})

        # Cache the frame (do NOT delete — it will be used by /analyze_frame)
        frame_id = _cache_frame(temp_path)

        scan_result = weight_estimator.scan_detections(
            temp_path, debug=True, debug_context="SCAN_FRAME"
        )

        return jsonify({
            'success': True,
            'frame_image_id': frame_id,
            'cows': scan_result['cows'],
            'posts': scan_result['posts'],
            'preview_image': 'data:image/jpeg;base64,' + scan_result.get('preview_b64', ''),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Clean up on error
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """Analyze a single frame from the video player canvas"""
    # Support both cached frame (from /scan_frame) and direct upload
    frame_image_id = request.form.get('frame_image_id')
    file = request.files.get('frame')

    if not frame_image_id and not file:
        return jsonify({'success': False, 'error': 'No frame provided'})

    breed = request.form.get('breed', 'desconocido')
    category = request.form.get('category', 'desconocido')
    age_range = request.form.get('age_range', 'desconocido')

    # Parse optional selection params (from two-phase flow)
    cow_index = int(request.form.get('cow_index', 0))
    post_indices_str = request.form.get('post_indices', '')
    post_indices = None
    if post_indices_str:
        try:
            post_indices = [int(x.strip()) for x in post_indices_str.split(',') if x.strip()]
        except ValueError:
            post_indices = None

    # Resolve frame path: cached (from /scan_frame) or direct upload
    _from_cache = False
    if frame_image_id:
        temp_path = _get_cached_frame(frame_image_id)
        if not temp_path:
            return jsonify({'success': False, 'error': 'Frame cache expired, please re-scan'})
        _from_cache = True
    else:
        temp_path = os.path.join(tempfile.gettempdir(), f'frame_{uuid.uuid4().hex}.jpg')
        file.save(temp_path)

    try:
        if not weight_estimator:
            return jsonify({'success': False, 'error': 'Weight estimator not available'})

        # Call estimate_weight with all return options + cow/post selection
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
            cow_index=cow_index,
            post_indices=post_indices,
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

        # Use cm_per_px from estimate_weight (consistent with user's cinta selection)
        cm_per_px = details.get('cm_per_px') if isinstance(details, dict) else None
        animal_bbox_height_px = details.get('animal_bbox_height_px') if isinstance(details, dict) else None
        postes_heights = details.get('postes_heights_px', []) if isinstance(details, dict) else []
        cow_height_cm = None

        if cm_per_px and animal_bbox_height_px:
            cow_height_cm = animal_bbox_height_px * cm_per_px
            print(f"[FRAME_ANALYZE] Using estimate_weight cm_per_px={cm_per_px:.5f} -> cow_height={cow_height_cm:.1f}cm (bbox_h={animal_bbox_height_px:.0f}px, postes_heights={postes_heights})")

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
        # Only delete if NOT from cache (cache has its own TTL cleanup)
        if not _from_cache:
            try:
                os.remove(temp_path)
            except:
                pass

@app.route('/batch_screen', methods=['POST'])
def batch_screen():
    """Batch screen all video frames, returning results as SSE stream."""
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({'error': 'No video provided'}), 400

    cm_per_px = request.form.get('cm_per_px')
    if not cm_per_px:
        return jsonify({'error': 'cm_per_px is required'}), 400
    cm_per_px = float(cm_per_px)

    frame_interval = int(request.form.get('frame_interval', 30))
    min_cow_score = float(request.form.get('min_cow_score', 0.75))
    breed = request.form.get('breed', 'desconocido')
    category = request.form.get('category', 'desconocido')
    age_range = request.form.get('age_range', 'desconocido')

    # Parse post_indices from calibration (only use these posts)
    post_indices_str = request.form.get('post_indices', '')
    post_indices = None
    if post_indices_str:
        try:
            post_indices = [int(x.strip()) for x in post_indices_str.split(',') if x.strip()]
        except ValueError:
            post_indices = None

    weight_min, weight_max = get_weight_range(category)

    # Save uploaded video to temp file
    temp_video_path = os.path.join(tempfile.gettempdir(), f'batch_{uuid.uuid4().hex}.mp4')
    video_file.save(temp_video_path)

    def generate():
        cap = None
        try:
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                yield f"event: error\ndata: {json.dumps({'message': 'No se pudo abrir el video'})}\n\n"
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # CAP_PROP_FRAME_COUNT no es confiable en muchos codecs.
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            duration_msec_end = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if duration_msec_end > 0:
                duration_sec = duration_msec_end / 1000.0
                total_frames_duration = int(round(fps * duration_sec))
                total_frames = max(total_frames_prop, total_frames_duration)
                print(f"  [Screening] FRAME_COUNT={total_frames_prop}, duration_based={total_frames_duration}, using={total_frames}")
            else:
                total_frames = total_frames_prop

            real_interval = max(1, int(round(frame_interval * fps / 30.0)))
            frames_to_process = max(1, total_frames // real_interval)
            print(f"  [Screening] fps={fps:.1f}, frame_interval_param={frame_interval}, real_interval={real_interval}, total_frames={total_frames}, to_process={frames_to_process}")

            yield f"event: started\ndata: {json.dumps({'total_frames': total_frames, 'frames_to_process': frames_to_process, 'fps': fps})}\n\n"

            processed = 0
            all_results = []

            for frame_num in range(0, total_frames, real_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    processed += 1
                    yield f"event: frame_skip\ndata: {json.dumps({'frame_num': frame_num, 'processed': processed, 'total': frames_to_process, 'reason': 'read_error'})}\n\n"
                    continue

                # Write frame to temp JPEG
                temp_frame_path = os.path.join(tempfile.gettempdir(), f'bframe_{uuid.uuid4().hex}.jpg')
                cv2.imwrite(temp_frame_path, frame)

                try:
                    if not weight_estimator:
                        processed += 1
                        yield f"event: frame_skip\ndata: {json.dumps({'frame_num': frame_num, 'processed': processed, 'total': frames_to_process, 'reason': 'no_estimator'})}\n\n"
                        continue

                    # Single estimate_weight call with same params as /analyze_frame
                    # (no separate scan_detections — cow_score comes from details dict)
                    result_tuple = weight_estimator.estimate_weight(
                        temp_frame_path,
                        visualize=True,
                        debug=True,
                        debug_context=f"BATCH_F{frame_num}",
                        return_eye_coords=True,
                        return_keypoint_coords=True,
                        scale_method='poste',
                        breed=breed,
                        category=category,
                        age_range=age_range,
                        cow_index=0,
                        post_indices=post_indices,
                        override_cm_per_px=cm_per_px,
                    )

                    # Unpack 5-tuple (same as analyze_frame)
                    img_rgb = result_tuple[0]
                    weight = result_tuple[1]
                    eye_coords = result_tuple[2]
                    kp_coords = result_tuple[3]
                    details = result_tuple[4] if len(result_tuple) > 4 else {}

                    processed += 1

                    # Check cow confidence from estimate_weight result
                    cow_score = details.get('cow_score') if isinstance(details, dict) else None
                    if min_cow_score > 0 and cow_score is not None and cow_score < min_cow_score:
                        yield f"event: frame_skip\ndata: {json.dumps({'frame_num': frame_num, 'processed': processed, 'total': frames_to_process, 'reason': 'low_cow_score', 'cow_score': round(cow_score, 3)})}\n\n"
                        continue

                    # Extract dist1_px and dist2_px from kp_coords (same as analyze_frame)
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

                    # Determine keypoints_found (same 3-tier check as analyze_frame)
                    keypoints_found = False
                    if kp_coords and len(kp_coords) > 0:
                        last_kp = kp_coords[-1]
                        if isinstance(last_kp, dict):
                            keypoints_found = bool(last_kp.get('keypoints_accepted', False))
                    if not keypoints_found and isinstance(details, dict):
                        keypoints_found = bool(details.get('keypoints_found', False))
                    if not keypoints_found and dist1_px is not None and dist2_px is not None:
                        keypoints_found = True

                    if weight is None or not keypoints_found:
                        reason = 'no_keypoints'
                        if isinstance(details, dict) and details.get('message'):
                            reason = details['message']
                        yield f"event: frame_skip\ndata: {json.dumps({'frame_num': frame_num, 'processed': processed, 'total': frames_to_process, 'reason': reason})}\n\n"
                        continue

                    # Calculate cow_height_cm (same as analyze_frame)
                    animal_bbox_height_px = details.get('animal_bbox_height_px') if isinstance(details, dict) else None
                    postes_heights = details.get('postes_heights_px', []) if isinstance(details, dict) else []
                    cow_height_cm = None

                    if len(postes_heights) >= 2 and animal_bbox_height_px:
                        avg_post_height_px = sum(postes_heights) / len(postes_heights)
                        calc_cm_per_px = 50.0 / avg_post_height_px
                        cow_height_cm = animal_bbox_height_px * calc_cm_per_px
                        print(f"[BATCH_F{frame_num}] cow_height: posts={postes_heights} -> avg={avg_post_height_px:.1f}px -> cm_per_px={calc_cm_per_px:.5f} -> height={cow_height_cm:.1f}cm")

                    # Generate thumbnail (640px wide for larger gallery display)
                    annotated_thumb_b64 = ''
                    if img_rgb is not None:
                        h, w = img_rgb.shape[:2]
                        thumb_w = 640
                        thumb_h = int(h * thumb_w / w)
                        thumb = cv2.resize(img_rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
                        _, buf = cv2.imencode('.jpg', thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        annotated_thumb_b64 = base64.b64encode(buf).decode('utf-8')

                    in_range = weight_min <= weight <= weight_max

                    frame_result = {
                        'frame_num': frame_num,
                        'processed': processed,
                        'total': frames_to_process,
                        'keypoints_found': True,
                        'weight_kg': round(weight, 2),
                        'in_range': in_range,
                        'annotated_thumb': annotated_thumb_b64,
                        'dist1_px': dist1_px,
                        'dist2_px': dist2_px,
                        'cm_per_px': details.get('cm_per_px') if isinstance(details, dict) else None,
                        'animal_bbox_height_px': animal_bbox_height_px,
                        'cow_score': round(cow_score, 3) if cow_score else None,
                        'cow_height_cm': round(cow_height_cm, 2) if cow_height_cm else None,
                        'postes_heights_px': postes_heights,
                    }

                    all_results.append(frame_result)
                    yield f"event: frame_result\ndata: {json.dumps(frame_result)}\n\n"

                except Exception as e:
                    processed += 1
                    yield f"event: frame_skip\ndata: {json.dumps({'frame_num': frame_num, 'processed': processed, 'total': frames_to_process, 'reason': str(e)})}\n\n"
                finally:
                    try:
                        os.remove(temp_frame_path)
                    except Exception:
                        pass

            # Summary
            valid_results = [r for r in all_results if r['in_range']]
            valid_weights = [r['weight_kg'] for r in valid_results]
            all_weights = [r['weight_kg'] for r in all_results]

            summary = {
                'total_processed': processed,
                'detected_count': len(all_results),
                'valid_count': len(valid_results),
                'outlier_count': len(all_results) - len(valid_results),
                'weight_range': [weight_min, weight_max],
            }

            if valid_weights:
                summary['avg_weight'] = round(statistics.mean(valid_weights), 2)
                summary['median_weight'] = round(statistics.median(valid_weights), 2)
                summary['std_dev'] = round(statistics.stdev(valid_weights), 2) if len(valid_weights) > 1 else 0
                summary['min_weight'] = round(min(valid_weights), 2)
                summary['max_weight'] = round(max(valid_weights), 2)
            elif all_weights:
                summary['avg_weight'] = round(statistics.mean(all_weights), 2)
                summary['median_weight'] = round(statistics.median(all_weights), 2)
                summary['std_dev'] = round(statistics.stdev(all_weights), 2) if len(all_weights) > 1 else 0
                summary['min_weight'] = round(min(all_weights), 2)
                summary['max_weight'] = round(max(all_weights), 2)

            yield f"event: complete\ndata: {json.dumps({'summary': summary})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            if cap:
                cap.release()
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/api/video_modelo3d', methods=['POST'])
def video_modelo3d():
    """SfM: extract frames → single 3D reconstruction → single volume/weight. SSE stream."""
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({'error': 'No video provided'}), 400

    cow_height_cm = request.form.get('cow_height_cm')
    if not cow_height_cm:
        return jsonify({'error': 'cow_height_cm is required'}), 400
    cow_height_cm = float(cow_height_cm)

    frame_interval = int(request.form.get('frame_interval', 30))
    vaca_name = request.form.get('vaca_name', 'vaca_video')
    vaca_name = secure_filename(vaca_name) or 'vaca_video'
    mode = request.form.get('mode', 'hibrido')  # 'hibrido' or 'sfm'

    temp_video_path = os.path.join(tempfile.gettempdir(), f'modelo3d_{uuid.uuid4().hex}.mp4')
    video_file.save(temp_video_path)

    if not weight_estimator:
        os.remove(temp_video_path)
        return jsonify({'error': 'Weight estimator not loaded'}), 500
    cow_model = weight_estimator.cow_model
    coco_model = weight_estimator.coco_model

    def generate():
        cap = None
        try:
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                yield f"event: error\ndata: {json.dumps({'message': 'No se pudo abrir el video'})}\n\n"
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # CAP_PROP_FRAME_COUNT no es confiable en muchos codecs.
            # Verificar con duración real si es posible.
            # Seek al final para contar frames reales
            duration_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)  # seek al final
            duration_msec_end = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # volver al inicio

            if duration_msec_end > 0:
                duration_sec = duration_msec_end / 1000.0
                total_frames_duration = int(round(fps * duration_sec))
                # Usar el mayor entre FRAME_COUNT y el calculado por duración
                total_frames = max(total_frames_prop, total_frames_duration)
                print(f"  [Modelo3D] FRAME_COUNT={total_frames_prop}, duration_based={total_frames_duration}, using={total_frames}")
            else:
                total_frames = total_frames_prop
                duration_sec = total_frames / fps if fps > 0 else 0

            # frame_interval viene como "cada N frames asumiendo 30fps"
            # Convertir a intervalo real basado en el FPS del video
            # Ej: frame_interval=30 con video a 15fps → real_interval=15 (1 frame/seg)
            real_interval = max(1, int(round(frame_interval * fps / 30.0)))
            frames_to_process = max(1, total_frames // real_interval)
            print(f"  [Modelo3D] fps={fps:.1f}, frame_interval_param={frame_interval}, real_interval={real_interval}, total_frames={total_frames}, to_process={frames_to_process}, duration={duration_sec:.1f}s")

            yield f"event: started\ndata: {json.dumps({'total_frames': total_frames, 'frames_to_process': frames_to_process, 'fps': round(fps, 1), 'duration_sec': round(duration_sec, 1)})}\n\n"

            # ── Phase 1: Extract frames with YOLO + GrabCut + torso crop ──
            extracted_frames = []
            extracted_masks = []
            extracted_masks_full = []
            extracted_bboxes = []
            extracted_num = 0

            for frame_num in range(0, total_frames, real_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    extracted_num += 1
                    continue

                try:
                    bbox = detectar_vaca(frame, cow_model, coco_model)
                    if bbox is None:
                        extracted_num += 1
                        yield f"event: extracting\ndata: {json.dumps({'frame_num': frame_num, 'extracted': extracted_num, 'total': frames_to_process, 'accepted': len(extracted_frames), 'skipped': True})}\n\n"
                        continue

                    mask_full, contorno_full = segmentar(frame, bbox)
                    if mask_full is None:
                        extracted_num += 1
                        yield f"event: extracting\ndata: {json.dumps({'frame_num': frame_num, 'extracted': extracted_num, 'total': frames_to_process, 'accepted': len(extracted_frames), 'skipped': True})}\n\n"
                        continue

                    mask_torso, _ = recortar_torso(mask_full, bbox)
                    if mask_torso is None:
                        mask_torso = mask_full

                    extracted_frames.append(frame)
                    extracted_masks.append(mask_torso)
                    extracted_masks_full.append(mask_full)
                    extracted_bboxes.append(bbox)
                    extracted_num += 1

                    # Thumbnail
                    h_img, w_img = frame.shape[:2]
                    thumb_w = 160
                    thumb_h = int(h_img * thumb_w / w_img)
                    thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                    _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    thumb_b64 = base64.b64encode(buf).decode('utf-8')

                    yield f"event: extracting\ndata: {json.dumps({'frame_num': frame_num, 'extracted': extracted_num, 'total': frames_to_process, 'accepted': len(extracted_frames), 'skipped': False, 'thumb_b64': thumb_b64})}\n\n"

                except Exception:
                    extracted_num += 1
                    continue

            if len(extracted_frames) < 2:
                yield f"event: error\ndata: {json.dumps({'message': f'Solo se extrajeron {len(extracted_frames)} frames validos (minimo 2).'})}\n\n"
                return

            # ── Phase 2: 3D Reconstruction (mode-dependent) ──
            recon_events = []

            def on_recon_progress(step, total, message):
                recon_events.append({'step': step, 'total_steps': total, 'message': message})

            if mode == 'sfm':
                sfm_result = sfm_real_desde_frames(extracted_frames, extracted_masks, cow_height_cm, bboxes=extracted_bboxes, masks_full=extracted_masks_full, on_progress=on_recon_progress)
            elif mode == 'sfm_legacy':
                sfm_result = sfm_desde_frames(extracted_frames, extracted_masks, cow_height_cm, bboxes=extracted_bboxes, masks_full=extracted_masks_full, on_progress=on_recon_progress)
            else:
                sfm_result = modelo_hibrido(extracted_frames, extracted_masks, cow_height_cm, bboxes=extracted_bboxes, masks_full=extracted_masks_full, on_progress=on_recon_progress)

            # Yield all accumulated progress events
            for evt in recon_events:
                yield f"event: sfm_progress\ndata: {json.dumps(evt)}\n\n"

            if sfm_result is None:
                method_name = 'SfM' if mode == 'sfm' else 'Híbrido'
                yield f"event: error\ndata: {json.dumps({'message': f'El modelo {method_name} no generó puntos 3D suficientes. Las imágenes pueden no tener suficiente variación de ángulo.'})}\n\n"
                return

            # ── Phase 3: Save PLY and emit result ──
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(base_dir, 'output_modelos3d_batch', vaca_name)
            os.makedirs(output_dir, exist_ok=True)

            ply_3d_name = f"{vaca_name}_3d.ply"
            ply_3d_path = os.path.join(output_dir, ply_3d_name)

            points_3d = sfm_result['points_3d']
            colors = sfm_result['colors']
            triangles = sfm_result['triangles']

            if len(triangles) > 0:
                guardar_ply_con_malla(ply_3d_path, points_3d, triangles, colors)
            else:
                # Fallback: point cloud only (import from reconstruccion_3d)
                from reconstruccion_3d import guardar_ply as guardar_ply_nube
                guardar_ply_nube(ply_3d_path, points_3d, colors)

            # Helper to ensure all values are JSON-safe (no numpy types)
            def _js(v):
                if isinstance(v, (np.integer,)):
                    return int(v)
                if isinstance(v, (np.floating,)):
                    return float(v)
                return v

            # Generar imagen resumen (modelo_escalado)
            try:
                img_resumen_path = os.path.join(output_dir, f"{vaca_name}_modelo_escalado.png")
                generar_imagen_resumen(sfm_result, img_resumen_path, vaca_name=vaca_name)
            except Exception as e:
                print(f"  [Modelo3D] Error generando imagen resumen: {e}")

            # Save resumen.json
            resumen = {
                'vaca': vaca_name,
                'method': mode,
                'cow_height_cm': float(cow_height_cm),
                'volumen_cm3': _js(sfm_result['volumen_cm3']),
                'volumen_litros': _js(sfm_result['volumen_litros']),
                'peso_kg': _js(sfm_result['peso_kg']),
                'peso_barril_kg': _js(sfm_result.get('peso_barril_kg', sfm_result['peso_kg'])),
                'alto_cm': _js(sfm_result['alto_cm']),
                'largo_cm': _js(sfm_result['largo_cm']),
                'ancho_cm': _js(sfm_result['ancho_cm']),
                'superficie_cm2': _js(sfm_result['superficie_cm2']),
                'num_points': _js(sfm_result['num_points']),
                'num_pairs': _js(sfm_result['num_pairs']),
                'num_triangles': _js(sfm_result['num_triangles']),
                'frames_extracted': len(extracted_frames),
                'scale_factor': _js(sfm_result['scale_factor']),
            }
            with open(os.path.join(output_dir, f"{vaca_name}_resumen.json"), 'w') as f:
                json.dump(resumen, f, indent=2, ensure_ascii=False)

            summary = {
                'volumen_litros': _js(sfm_result['volumen_litros']),
                'peso_kg': _js(sfm_result['peso_kg']),
                'peso_barril_kg': _js(sfm_result.get('peso_barril_kg', sfm_result['peso_kg'])),
                'alto_cm': _js(sfm_result['alto_cm']),
                'largo_cm': _js(sfm_result['largo_cm']),
                'ancho_cm': _js(sfm_result['ancho_cm']),
                'num_points': _js(sfm_result['num_points']),
                'num_pairs': _js(sfm_result['num_pairs']),
                'num_triangles': _js(sfm_result['num_triangles']),
                'frames_used': len(extracted_frames),
                'mode': mode,
                'ply_id': vaca_name,
                'ply_3d': ply_3d_name,
            }

            yield f"event: complete\ndata: {json.dumps(summary)}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            if cap:
                cap.release()
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


def to_farm():
    session.pop('facility')
    return redirect(url_for('index'))

# ── 3D Model Viewer endpoints ──

MODELO_GRANDES_DIR = 'output_modelos3d_grandes'
MODELO_DESFILE_DIR = 'output_modelos3d_desfile26marz'
MODELO_26MARZ_DIR = 'output_modelos3d_26marz'
MODELO_RECORTE26MARZ_DIR = 'output_modelos3d_Recorte26marz_altdiag'
ALTO_ESTIMADO_DEFAULT = 120.0  # cm fallback assumed during model generation


def _discover_modelo_dirs():
    """Auto-discover model subdirectories inside all model output dirs."""
    base = os.path.dirname(os.path.abspath(__file__))
    dirs = {}
    for model_dir in [MODELO_RECORTE26MARZ_DIR]:
        batch_path = os.path.join(base, model_dir)
        if os.path.isdir(batch_path):
            for entry in sorted(os.listdir(batch_path)):
                if entry.startswith('_'):
                    continue
                full = os.path.join(batch_path, entry)
                if os.path.isdir(full):
                    dirs[entry] = os.path.join(model_dir, entry)
    return dirs


def _load_resumen(dir_path):
    """Load the *_resumen.json from a model directory."""
    for f in sorted(os.listdir(dir_path)):
        if f.endswith('_resumen.json') or f == 'resumen.json':
            with open(os.path.join(dir_path, f)) as jf:
                return json.load(jf)
    return {}


@app.route('/api/modelos_disponibles')
def modelos_disponibles():
    modelos = []
    base = os.path.dirname(os.path.abspath(__file__))
    modelo_dirs = _discover_modelo_dirs()
    for vaca, carpeta in modelo_dirs.items():
        dir_path = os.path.join(base, carpeta)
        if not os.path.isdir(dir_path):
            continue
        ply_3d = None
        ply_lat = None
        for f in os.listdir(dir_path):
            if f.endswith('.ply') and '3d' in f.lower():
                ply_3d = f
            elif f.endswith('.ply') and 'lateral' in f.lower():
                ply_lat = f
        if not ply_3d and not ply_lat:
            continue
        meta = _load_resumen(dir_path)
        # Support both old format (peso_kg, volumen_litros) and new (peso_real_kg, vol_total_litros)
        peso = meta.get('peso_kg') or meta.get('peso_real_kg')
        vol = meta.get('volumen_litros') or meta.get('vol_total_litros')
        vol_barril = meta.get('vol_barril_litros') or meta.get('volumen_barril_litros')
        modelos.append({
            'id': vaca,
            'nombre': vaca.replace('_', ' ').title(),
            'ply_3d': ply_3d,
            'ply_lateral': ply_lat,
            'escala_cm_px': meta.get('escala_cm_px'),
            'alto_estimado_cm': meta.get('altura_estimada_cm', ALTO_ESTIMADO_DEFAULT),
            'foto_usada': meta.get('foto_usada'),
            'fotos_validas': meta.get('fotos_validas'),
            'peso_kg': peso,
            'volumen_litros': vol,
            'vol_barril_litros': vol_barril,
            'largo_cm': meta.get('largo_cm'),
            'alto_cm': meta.get('alto_cm'),
            'superficie_cm2': meta.get('superficie_cm2'),
        })
    return jsonify(modelos)


@app.route('/api/modelo3d/<vaca>/recalcular', methods=['POST'])
def recalcular_volumen(vaca):
    """Recalculate metrics using actual height instead of the assumed default.

    The model was generated assuming alto_cm = resumen's alto_cm.
    The correction factor is: new_height / original_alto_cm.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    modelo_dirs = _discover_modelo_dirs()
    carpeta = modelo_dirs.get(vaca)
    if not carpeta:
        return jsonify({'error': 'Vaca no encontrada'}), 404

    data = request.get_json(force=True)
    altura_cm = data.get('altura_cm')
    if not altura_cm or not isinstance(altura_cm, (int, float)) or altura_cm <= 0:
        return jsonify({'error': 'altura_cm invalida'}), 400

    dir_path = os.path.join(base, carpeta)
    meta = _load_resumen(dir_path)
    if not meta:
        return jsonify({'error': 'Resumen no encontrado'}), 404

    # The original alto_cm is the height the model was generated with
    alto_original = meta.get('alto_cm', ALTO_ESTIMADO_DEFAULT)
    if alto_original <= 0:
        alto_original = ALTO_ESTIMADO_DEFAULT

    # Factor de corrección: altura real / altura con la que se generó
    factor = altura_cm / alto_original
    factor2 = factor ** 2
    factor3 = factor ** 3

    volumen_litros = round(meta.get('volumen_litros', meta.get('volumen_cm3', 0) / 1000) * factor3, 1)
    peso_kg = round(volumen_litros * 1.03, 1)

    return jsonify({
        'id': vaca,
        'altura_cm': round(altura_cm, 1),
        'largo_cm': round(meta.get('largo_cm', 0) * factor, 1),
        'alto_cm': round(alto_original * factor, 1),
        'area_lateral_cm2': round(meta.get('area_lateral_cm2', 0) * factor2, 1),
        'volumen_cm3': round(meta.get('volumen_cm3', 0) * factor3, 1),
        'volumen_litros': volumen_litros,
        'superficie_cm2': round(meta.get('superficie_cm2', 0) * factor2, 1),
        'peso_kg': peso_kg,
    })


@app.route('/api/altura_estimada')
def altura_estimada():
    """Return estimated bbox height for category + age combination."""
    category = request.args.get('category', 'desconocido')
    age_range = request.args.get('age_range', 'desconocido')
    altura = get_estimated_height(category, age_range)
    return jsonify({'altura_cm': altura, 'category': category, 'age_range': age_range})


@app.route('/api/alturas_tabla')
def alturas_tabla():
    """Return the full height table for the UI."""
    return jsonify(HEIGHT_BY_CATEGORY_AGE)


@app.route('/api/modelo3d/<vaca>/<archivo>')
def get_modelo_3d(vaca, archivo):
    base = os.path.dirname(os.path.abspath(__file__))
    modelo_dirs = _discover_modelo_dirs()
    carpeta = modelo_dirs.get(vaca)
    if not carpeta:
        return jsonify({'error': 'Vaca no encontrada'}), 404
    safe_archivo = secure_filename(archivo)
    filepath = os.path.join(base, carpeta, safe_archivo)
    if not os.path.isfile(filepath) or not safe_archivo.endswith('.ply'):
        return jsonify({'error': 'Archivo PLY no encontrado'}), 404
    from flask import send_file
    return send_file(filepath, mimetype='application/octet-stream')


# start the server with the 'run()' method
if __name__ == '__main__':
    # Usar puerto 5001 para evitar conflicto con AirPlay en macOS
    # 0.0.0.0 evita problemas de resolución con localhost
    app.run(host='0.0.0.0', port=5001, debug=True)
    # app.run(debug=True)

# TODO: atrás para otra granja
# TODO: Página de despedida
