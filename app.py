from flask import Flask, redirect, url_for, request, render_template, jsonify, session, Response, stream_with_context, send_file
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
        poste1_height_cm=112,
        poste2_height_cm=112,
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

# Modelo barril_seg para recortar el bbox al cuerpo (sin cabeza) y obtener
# la altura a la cruz/lomo de forma robusta (no depende de keypoints).
# Modelo silueta_seg para obtener la posición exacta de las pezuñas (bottom).
try:
    from ultralytics import YOLO as _YOLO_seg
    _app_dir = os.path.dirname(os.path.abspath(__file__))
    _barril_path = os.path.join(_app_dir, 'barril_seg.pt')
    _silueta_path = os.path.join(_app_dir, 'silueta_seg.pt')
    barril_seg_model = _YOLO_seg(_barril_path) if os.path.exists(_barril_path) else None
    silueta_seg_model = _YOLO_seg(_silueta_path) if os.path.exists(_silueta_path) else None
    if barril_seg_model:
        print(f"[app] barril_seg.pt cargado")
    if silueta_seg_model:
        print(f"[app] silueta_seg.pt cargado")
except Exception as e:
    print(f"Advertencia: No se pudieron cargar seg models: {e}")
    barril_seg_model = None
    silueta_seg_model = None

# ── Frame cache for two-phase scan/analyze flow ──
_frame_cache = {}  # UUID → {'path': str, 'created': float}
_FRAME_CACHE_TTL = 300  # 5 minutes

# ── Locked reference rectangle per video (calibración fija) ──
# video_id → {'post1': {cx, top_tape, floor, tape_px}, 'post2': {...}}
_locked_references = {}

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


def _reparar_mascara_oclusion(binmask, frac_alto=0.45, ventana_envelope=5):
    """Rellena columnas DENTRO del rango X de la máscara interpolando top_y/bot_y
    linealmente desde los vecinos válidos. Recupera porciones ocluidas (p.ej.
    poste de escala cruzando el barril) sin detectar el oclusor — la propia
    máscara dicta qué reparar.

    Pre-pasada: vacía columnas con altura ANÓMALAMENTE PEQUEÑA respecto a la
    mediana (frac_alto * h_mediana). Esto detecta huecos parciales (columnas
    con unos pocos píxeles dispersos por restos bajo el poste) que la lógica
    clásica de "columna totalmente vacía" no captura.

    Modifica `binmask` in-place. Devuelve un set con los índices de las
    columnas que quedaron reparadas (vaciadas anómalas + interpoladas).
    """
    if binmask is None or binmask.size == 0:
        return set()
    bh, bw = binmask.shape
    cols_sum = binmask.sum(axis=0)
    cols_valid = np.where(cols_sum > 0)[0]
    if cols_valid.size < 2:
        return set()

    # Pre-pasada: vaciar columnas con altura anómala
    heights = np.zeros(bw, dtype=np.int32)
    for _c in cols_valid:
        _rows = np.where(binmask[:, _c] > 0)[0]
        heights[_c] = int(_rows[-1] - _rows[0] + 1)
    h_med = int(np.median(heights[cols_valid]))
    umbral_h = max(2, int(frac_alto * h_med))
    cols_reparadas = set()
    for _c in cols_valid:
        if heights[_c] < umbral_h:
            binmask[:, _c] = 0
            cols_reparadas.add(int(_c))

    # Recalcular cols válidas después del vaciado
    cols_sum = binmask.sum(axis=0)
    cols_valid = np.where(cols_sum > 0)[0]
    if cols_valid.size < 2:
        return cols_reparadas

    col_first = int(cols_valid[0])
    col_last = int(cols_valid[-1])
    top_arr = np.full(bw, -1, dtype=np.int32)
    bot_arr = np.full(bw, -1, dtype=np.int32)
    for _c in cols_valid:
        _rows = np.where(binmask[:, _c] > 0)[0]
        top_arr[_c] = int(_rows[0])
        bot_arr[_c] = int(_rows[-1])
    c = col_first + 1
    while c < col_last:
        if top_arr[c] < 0:
            gap_start = c
            gap_end = c
            while gap_end + 1 < col_last and top_arr[gap_end + 1] < 0:
                gap_end += 1
            # ENVELOPE en VENTANA: en vez de mirar solo la vecina inmediata,
            # tomamos el min(top) y max(bot) de las `ventana_envelope` cols
            # más cercanas a cada lado del hueco. Así si varias vecinas
            # también están afectadas por la oclusión (poste recortó
            # columnas seguidas), llegamos a una columna sana donde el
            # barril mide su altura real.
            tops_L = []
            bots_L = []
            for kk in range(1, ventana_envelope + 1):
                idx = gap_start - kk
                if idx < c0:
                    break
                if top_arr[idx] >= 0:
                    tops_L.append(int(top_arr[idx]))
                    bots_L.append(int(bot_arr[idx]))
            tops_R = []
            bots_R = []
            for kk in range(1, ventana_envelope + 1):
                idx = gap_end + kk
                if idx > col_last:
                    break
                if top_arr[idx] >= 0:
                    tops_R.append(int(top_arr[idx]))
                    bots_R.append(int(bot_arr[idx]))
            if not tops_L or not tops_R:
                # fallback: solo inmediata si la ventana cae fuera del rango
                tops_L = [int(top_arr[gap_start - 1])] if not tops_L else tops_L
                bots_L = [int(bot_arr[gap_start - 1])] if not bots_L else bots_L
                tops_R = [int(top_arr[gap_end + 1])] if not tops_R else tops_R
                bots_R = [int(bot_arr[gap_end + 1])] if not bots_R else bots_R
            top_k = min(min(tops_L), min(tops_R))
            bot_k = max(max(bots_L), max(bots_R))
            for col_k in range(gap_start, gap_end + 1):
                if bot_k >= top_k:
                    binmask[top_k:bot_k + 1, col_k] = 1
                    top_arr[col_k] = top_k
                    bot_arr[col_k] = bot_k
                    cols_reparadas.add(int(col_k))
            c = gap_end + 1
        else:
            c += 1
    return cols_reparadas


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

                # Rango de verde amplio (pasto fresco + seco) solo para detectar piso
                FLOOR_GREEN_LOWER = np.array([20, 25, 40])
                FLOOR_GREEN_UPPER = np.array([95, 255, 255])

                def _detect_floor_y(x_center, y_start, bbox_y2):
                    """Escanea hacia abajo desde y_start buscando la primera fila dominada
                    por pasto. Busca en un strip horizontal centrado en x_center.
                    Retorna (floor_y, confianza) o (bbox_y2, 0.0) si no encuentra.
                    """
                    h_img, w_img = image.shape[:2]
                    # Strip más ancho que el poste para capturar pasto alrededor
                    strip_half = 60
                    x_from = max(0, x_center - strip_half)
                    x_to = min(w_img, x_center + strip_half)
                    y_max = min(h_img, y_start + max(300, (bbox_y2 - y_start) * 3))

                    if y_max <= y_start or x_to <= x_from:
                        return bbox_y2, 0.0

                    roi = image[y_start:y_max, x_from:x_to]
                    if roi.size == 0:
                        return bbox_y2, 0.0

                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    green = cv2.inRange(hsv, FLOOR_GREEN_LOWER, FLOOR_GREEN_UPPER)
                    kernel = np.ones((3, 3), np.uint8)
                    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel, iterations=1)

                    row_ratio = np.mean(green > 0, axis=1)
                    # Primera fila con al menos 40% de verde
                    threshold = 0.40
                    hits = np.where(row_ratio >= threshold)[0]
                    if len(hits) == 0:
                        return bbox_y2, 0.0

                    # Exigir consistencia: 6 filas siguientes también verdes (evita ruido)
                    for row in hits:
                        window = row_ratio[row:row + 6]
                        if len(window) >= 3 and np.mean(window >= threshold) >= 0.6:
                            confidence = float(row_ratio[row])
                            return y_start + int(row), confidence

                    return bbox_y2, 0.0

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

                    # Detectar piso (pasto) debajo del poste
                    floor_y, floor_conf = _detect_floor_y(cx, line_y2, y2)

                    # Dibujar línea horizontal cyan en el piso detectado
                    floor_color = (255, 255, 0)  # cyan (BGR)
                    half_w = max(30, (x2 - x1))
                    cv2.line(annotated, (cx - half_w, floor_y), (cx + half_w, floor_y),
                             (0, 0, 0), 4)
                    cv2.line(annotated, (cx - half_w, floor_y), (cx + half_w, floor_y),
                             floor_color, 2)
                    cv2.putText(annotated, f'piso ({floor_conf*100:.0f}%)',
                                (cx + half_w + 5, floor_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, floor_color, 2)

                    return {
                        'cx': cx,
                        'top_tape': line_y1,
                        'bottom_tape': line_y2,
                        'floor': floor_y,
                        'floor_confidence': round(floor_conf, 3),
                        'tape_px': height_px,
                        'bbox': [x1, y1, x2, y2],
                    }

                p1_info = _draw_poste_with_yellow_line(poste1_bbox, 'POSTE 1') if poste1_bbox is not None else None
                p2_info = _draw_poste_with_yellow_line(poste2_bbox, 'POSTE 2') if poste2_bbox is not None else None

                # Rectángulo virtual: top cinta 1/2 → piso 1/2, escala extendida con 50cm
                if p1_info is not None and p2_info is not None and p1_info['tape_px'] > 0 and p2_info['tape_px'] > 0:
                    TAPE_CM = 112.0
                    rect_color = (0, 255, 255)  # amarillo (BGR)

                    scale1 = TAPE_CM / p1_info['tape_px']  # cm/px en poste 1
                    scale2 = TAPE_CM / p2_info['tape_px']  # cm/px en poste 2

                    h1_full_px = max(0, p1_info['floor'] - p1_info['top_tape'])
                    h2_full_px = max(0, p2_info['floor'] - p2_info['top_tape'])
                    h1_full_cm = h1_full_px * scale1
                    h2_full_cm = h2_full_px * scale2

                    tl = (p1_info['cx'], p1_info['top_tape'])
                    tr = (p2_info['cx'], p2_info['top_tape'])
                    bl = (p1_info['cx'], p1_info['floor'])
                    br = (p2_info['cx'], p2_info['floor'])

                    # Dibujar los 4 lados del rectángulo virtual
                    for a, b in [(tl, tr), (bl, br), (tl, bl), (tr, br)]:
                        cv2.line(annotated, a, b, (0, 0, 0), 5)        # halo negro
                        cv2.line(annotated, a, b, rect_color, 2)        # línea amarilla

                    # Labels con altura total de cada lado
                    for pt_top, pt_bot, val_cm, side in [
                        (tl, bl, h1_full_cm, 'L'),
                        (tr, br, h2_full_cm, 'R'),
                    ]:
                        txt = f'{val_cm:.0f}cm'
                        y_mid = (pt_top[1] + pt_bot[1]) // 2
                        x_txt = pt_top[0] + 10 if side == 'L' else pt_top[0] - 100
                        cv2.putText(annotated, txt, (x_txt, y_mid),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                        cv2.putText(annotated, txt, (x_txt, y_mid),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_color, 2)

                    result['rectangle'] = {
                        'top_left': list(tl),
                        'top_right': list(tr),
                        'bottom_left': list(bl),
                        'bottom_right': list(br),
                        'tape_cm': TAPE_CM,
                        'scale1_cm_per_px': round(scale1, 5),
                        'scale2_cm_per_px': round(scale2, 5),
                        'altura_lado1_cm': round(h1_full_cm, 1),
                        'altura_lado2_cm': round(h2_full_cm, 1),
                        'tape1_px': p1_info['tape_px'],
                        'tape2_px': p2_info['tape_px'],
                    }

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

@app.route('/lock_reference', methods=['POST'])
def lock_reference():
    """Fija el rectángulo de referencia para un video. Body JSON:
    {
      "video_id": "...",
      "post1": {"cx": int, "top_tape": int, "floor": int, "tape_px": int},
      "post2": {"cx": int, "top_tape": int, "floor": int, "tape_px": int}
    }
    """
    data = request.get_json(force=True) or {}
    video_id = data.get('video_id')
    post1 = data.get('post1')
    post2 = data.get('post2')
    if not video_id or not post1 or not post2:
        return jsonify({'success': False, 'error': 'video_id, post1 y post2 requeridos'}), 400
    for k in ('cx', 'top_tape', 'floor', 'tape_px'):
        if k not in post1 or k not in post2:
            return jsonify({'success': False, 'error': f'falta campo {k} en post1/post2'}), 400
    _locked_references[video_id] = {
        'post1': {k: float(post1[k]) for k in ('cx', 'top_tape', 'floor', 'tape_px')},
        'post2': {k: float(post2[k]) for k in ('cx', 'top_tape', 'floor', 'tape_px')},
        'created': _time.time(),
    }
    return jsonify({'success': True, 'video_id': video_id,
                    'reference': _locked_references[video_id]})


@app.route('/get_reference/<video_id>', methods=['GET'])
def get_reference(video_id):
    ref = _locked_references.get(video_id)
    if not ref:
        return jsonify({'success': False, 'error': 'no hay referencia fijada'}), 404
    return jsonify({'success': True, 'video_id': video_id, 'reference': ref})


@app.route('/clear_reference/<video_id>', methods=['POST', 'DELETE'])
def clear_reference(video_id):
    if video_id in _locked_references:
        del _locked_references[video_id]
        return jsonify({'success': True, 'video_id': video_id, 'cleared': True})
    return jsonify({'success': True, 'video_id': video_id, 'cleared': False})


@app.route('/calibrate_frame', methods=['POST'])
def calibrate_frame():
    """Calibra la referencia de postes SIN necesidad de detectar vaca.
    Dado un frame (o frame_image_id) y post_indices seleccionados, detecta
    los postes, arma el rectángulo cinta→piso y retorna:
      - preview anotado
      - rectangle_ref listo para fijar

    Form params:
      frame_image_id (opcional si viene cacheado)
      frame (opcional si no viene cacheado)
      post_indices: csv de índices (como en /scan_frame)
    """
    frame_image_id = request.form.get('frame_image_id')
    file = request.files.get('frame')
    post_indices_str = request.form.get('post_indices', '')
    post_indices = []
    if post_indices_str:
        try:
            post_indices = [int(x.strip()) for x in post_indices_str.split(',') if x.strip()]
        except ValueError:
            post_indices = []

    if not frame_image_id and not file:
        return jsonify({'success': False, 'error': 'No frame provided'}), 400
    if len(post_indices) != 2:
        return jsonify({'success': False, 'error': 'Se requieren exactamente 2 post_indices'}), 400

    _from_cache = False
    if frame_image_id:
        temp_path = _get_cached_frame(frame_image_id)
        if not temp_path:
            return jsonify({'success': False, 'error': 'Frame cache expired, please re-scan'}), 400
        _from_cache = True
    else:
        temp_path = os.path.join(tempfile.gettempdir(), f'frame_{uuid.uuid4().hex}.jpg')
        file.save(temp_path)

    try:
        if not weight_estimator:
            return jsonify({'success': False, 'error': 'weight_estimator no disponible'}), 500

        # Usar scan_detections para mantener consistencia de índices con /scan_frame
        scan_result = weight_estimator.scan_detections(temp_path, debug=False)
        posts_scan = scan_result.get('posts', [])
        if len(posts_scan) < 2:
            return jsonify({'success': False,
                            'error': f'Solo {len(posts_scan)} postes detectados (se necesitan 2)'}), 400

        # Filtrar por post_indices (los mismos que mandó la UI desde /scan_frame)
        try:
            sel = [posts_scan[i] for i in post_indices]
        except IndexError:
            return jsonify({'success': False,
                            'error': f'post_indices fuera de rango (detectados={len(posts_scan)})'}), 400

        # Necesitamos la IMAGEN REDIMENSIONADA (coords de posts_scan están en ese espacio)
        img_orig, resized_image, img_rgb, scale_factor, pad_x, pad_y, w_orig, h_orig = \
            weight_estimator._load_and_resize(temp_path, lambda m: None)

        from weight_estimation import _draw_tape_and_floor_on

        annotated_rgb = img_rgb.copy()
        infos = []
        for p in sel:
            x1, y1, x2, y2 = map(int, p['bbox'])
            cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), (255, 0, 255), 1)
            info = _draw_tape_and_floor_on(annotated_rgb, p['bbox'])
            if info is not None:
                infos.append(info)

        if len(infos) != 2:
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
            img_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')
            return jsonify({
                'success': False,
                'error': f'No se pudo detectar cinta+piso en los 2 postes (detectados OK: {len(infos)})',
                'preview_image': img_b64,
            }), 200

        p1, p2 = infos[0], infos[1]
        if p1['cx'] > p2['cx']:
            p1, p2 = p2, p1
        rect_color = (0, 255, 255)  # yellow RGB
        for a, b in [((p1['cx'], p1['top_tape']), (p2['cx'], p2['top_tape'])),
                     ((p1['cx'], p1['floor']),    (p2['cx'], p2['floor'])),
                     ((p1['cx'], p1['top_tape']), (p1['cx'], p1['floor'])),
                     ((p2['cx'], p2['top_tape']), (p2['cx'], p2['floor']))]:
            cv2.line(annotated_rgb, a, b, rect_color, 1)
        for p, side in [(p1, 'L'), (p2, 'R')]:
            if p['tape_px'] > 0:
                scale = 112.0 / p['tape_px']
                h_cm = (p['floor'] - p['top_tape']) * scale
                txt = f"{h_cm:.0f}cm"
                ymid = (p['top_tape'] + p['floor']) // 2
                x_txt = p['cx'] + 12 if side == 'L' else p['cx'] - 100
                cv2.putText(annotated_rgb, txt, (x_txt, ymid),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, rect_color, 2)

        # Coords están en espacio letterbox (resized). Convertir a original para el UI.
        def _to_orig_x(x):
            return (x - pad_x) / scale_factor if scale_factor else x

        def _to_orig_y(y):
            return (y - pad_y) / scale_factor if scale_factor else y

        def _to_orig_len(l):
            return l / scale_factor if scale_factor else l

        rectangle_ref = {
            'post1': {'cx': int(p1['cx']), 'top_tape': int(p1['top_tape']),
                      'floor': int(p1['floor']), 'tape_px': int(p1['tape_px'])},
            'post2': {'cx': int(p2['cx']), 'top_tape': int(p2['top_tape']),
                      'floor': int(p2['floor']), 'tape_px': int(p2['tape_px'])},
            # Coords en espacio ORIGINAL del video (para overlay en <video>)
            'original_coords': {
                'post1': {
                    'cx': round(_to_orig_x(p1['cx']), 1),
                    'top_tape': round(_to_orig_y(p1['top_tape']), 1),
                    'floor': round(_to_orig_y(p1['floor']), 1),
                    'tape_px': round(_to_orig_len(p1['tape_px']), 1),
                },
                'post2': {
                    'cx': round(_to_orig_x(p2['cx']), 1),
                    'top_tape': round(_to_orig_y(p2['top_tape']), 1),
                    'floor': round(_to_orig_y(p2['floor']), 1),
                    'tape_px': round(_to_orig_len(p2['tape_px']), 1),
                },
                'video_w': int(w_orig),
                'video_h': int(h_orig),
            },
        }

        _, buf = cv2.imencode('.jpg', cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
        img_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')

        return jsonify({
            'success': True,
            'preview_image': img_b64,
            'rectangle_ref': rectangle_ref,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if not _from_cache and temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.route('/generate_3d_consensus', methods=['POST'])
def generate_3d_consensus():
    """Combina N contornos (barril_contour_norm) de frames distintos en un
    contorno consenso (mediana por posición) y genera un único PLY del barril
    + un único volumen.

    Body JSON:
      {
        "cow_name": "vaca_X",
        "altura_cm": 125.3,
        "contours": [
          {"n_samples": 60, "width_cm": 80.2, "heights_cm": [...]},
          ...
        ]
      }
    """
    import json as _json
    from pathlib import Path as _Path

    try:
        data = request.get_json(force=True) or {}
        cow_name_raw = (data.get('cow_name', '') or 'vaca_live').strip()
        cow_name = ''.join(c for c in cow_name_raw if c.isalnum() or c in '_-')
        altura_cm = float(data.get('altura_cm') or 0)
        contours = data.get('contours') or []

        if not contours:
            return jsonify({'success': False, 'error': 'sin contornos'}), 400

        # Validar y normalizar
        N = None
        widths = []
        heights_mat = []
        tops_mat = []
        bottoms_mat = []
        for c in contours:
            if not c or not c.get('heights_cm'):
                continue
            h = c.get('heights_cm')
            n = c.get('n_samples') or len(h)
            w = float(c.get('width_cm') or 0)
            if w <= 0 or n <= 2 or len(h) != n:
                continue
            if N is None:
                N = n
            if n != N:
                continue  # saltamos frames con distinto sample count
            widths.append(w)
            heights_mat.append(h)
            t = c.get('tops_cm')
            b = c.get('bottoms_cm')
            if t and b and len(t) == N and len(b) == N:
                tops_mat.append(t)
                bottoms_mat.append(b)

        if not widths:
            return jsonify({'success': False, 'error': 'no hay contornos válidos'}), 400

        widths_arr = np.array(widths, dtype=float)
        heights_arr = np.array(heights_mat, dtype=float)  # (n_frames, N)

        # CONSENSO: mediana por posición
        width_median = float(np.median(widths_arr))
        heights_median = np.median(heights_arr, axis=0)  # (N,)

        # Si tenemos tops/bottoms por sample (frames nuevos), usamos la mediana
        # de cada uno para reconstruir la silueta REAL — la malla va a tener
        # forma de vaca (lomo arriba, barriga abajo), no un tubo simétrico.
        # Fallback a elipse simétrica si vienen frames viejos sin esos campos.
        use_shape = len(tops_mat) == len(widths) and len(tops_mat) > 0
        if use_shape:
            tops_arr = np.array(tops_mat, dtype=float)
            bots_arr = np.array(bottoms_mat, dtype=float)
            tops_median = np.median(tops_arr, axis=0)
            bots_median = np.median(bots_arr, axis=0)
        else:
            tops_median = None
            bots_median = None

        # Generar rebanadas del consenso
        K_DEPTH = 0.25
        rebanadas = []
        for i in range(N):
            x_cm = width_median * i / (N - 1)
            if use_shape:
                top_i = float(tops_median[i])
                bot_i = float(bots_median[i])
                h_cm = top_i - bot_i
                if h_cm <= 0:
                    continue
                y_c = (top_i + bot_i) / 2.0
            else:
                h_cm = float(heights_median[i])
                if h_cm <= 0:
                    continue
                y_c = h_cm / 2.0
            rebanadas.append((x_cm, y_c, h_cm))

        if len(rebanadas) < 3:
            return jsonify({'success': False, 'error': 'rebanadas insuficientes'}), 400

        # Volumen del consenso (integración trapezoidal)
        vol_cm3 = 0.0
        for i in range(len(rebanadas) - 1):
            x0, _, h0 = rebanadas[i]
            x1, _, h1 = rebanadas[i + 1]
            dx = x1 - x0
            a0, b0 = h0 / 2.0, h0 * K_DEPTH
            a1, b1 = h1 / 2.0, h1 * K_DEPTH
            area_avg = (np.pi * a0 * b0 + np.pi * a1 * b1) / 2.0
            vol_cm3 += area_avg * dx
        barril_consenso_L = round(vol_cm3 / 1000.0, 1)

        # Generar malla elipsoidal con generar_ply_volumen
        sys_path_added = False
        try:
            import sys as _sys
            _proj = os.path.dirname(os.path.abspath(__file__))
            if _proj not in _sys.path:
                _sys.path.insert(0, _proj)
                sys_path_added = True
            from generar_ply_volumen import malla_elipsoidal, escribir_ply
        finally:
            if sys_path_added:
                _sys.path.remove(_proj)

        vertices, tris = malla_elipsoidal(rebanadas, n_vert=32)

        # Guardar en output_modelos3d_live/<cow_name>/
        # NO tocamos _3d.ply ni _lateral.ply: esos vienen del frame
        # representativo vía /generate_3d_from_frame (silueta real + colores
        # de la imagen, estilo V1barrilbien). Acá solo escribimos _volumen.ply
        # (malla elipsoidal del consenso multi-frame) y el resumen con el
        # volumen consenso — que es el dato mostrado al usuario.
        proj_dir = _Path(os.path.dirname(os.path.abspath(__file__)))
        out_dir = proj_dir / MODELO_LIVE_DIR / cow_name
        out_dir.mkdir(parents=True, exist_ok=True)
        ply_vol = out_dir / f'{cow_name}_volumen.ply'
        escribir_ply(ply_vol, vertices, tris,
                     comentario=f'{cow_name} volumen consenso de {len(widths)} frames | altura={altura_cm:.1f}cm | barril={barril_consenso_L}L')

        resumen = {
            'individuo': cow_name,
            'altura_real_cm': altura_cm,
            'vol_barril_litros': barril_consenso_L,
            'metodo': 'consenso_multi_frame',
            'frames_usados': len(widths),
            'width_consenso_cm': round(width_median, 1),
            'generado_desde_pasada': True,
        }
        with open(out_dir / f'{cow_name}_resumen.json', 'w') as rf:
            _json.dump(resumen, rf, indent=2)

        return jsonify({
            'success': True,
            'model_id': cow_name,
            'barril_consenso_L': barril_consenso_L,
            'frames_usados': len(widths),
            'width_consenso_cm': round(width_median, 1),
            'ply_volumen': f'{cow_name}_volumen.ply',
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/generate_3d_from_frame', methods=['POST'])
def generate_3d_from_frame():
    """Genera PLYs (_3d, _lateral, _volumen) de la vaca a partir del frame
    representativo + silueta_seg + locked_reference. Guarda en
    output_modelos3d_live/<cow_name>/ para que el viewer 3D lo muestre.
    """
    import json as _json
    from pathlib import Path as _Path

    file = request.files.get('frame')
    if not file:
        return jsonify({'success': False, 'error': 'no frame'}), 400

    cow_name = (request.form.get('cow_name', '') or 'vaca_live').strip()
    cow_name = ''.join(c for c in cow_name if c.isalnum() or c in '_-')
    altura_cm = float(request.form.get('altura_cm', 0) or 0)
    barril_L_str = request.form.get('barril_L', '0') or '0'
    try:
        barril_L = float(barril_L_str)
    except Exception:
        barril_L = 0.0

    video_id = request.form.get('video_id', '').strip() or None
    inline = request.form.get('locked_reference_json', '').strip()
    locked_reference = None
    if inline:
        try:
            locked_reference = _json.loads(inline)
        except Exception:
            pass
    if not locked_reference and video_id:
        locked_reference = _locked_references.get(video_id)
    if not locked_reference:
        return jsonify({'success': False, 'error': 'no locked_reference'}), 400

    if barril_seg_model is None:
        return jsonify({'success': False, 'error': 'barril_seg no disponible'}), 500

    temp_path = os.path.join(tempfile.gettempdir(), f'gen3d_{uuid.uuid4().hex}.jpg')
    file.save(temp_path)

    try:
        image = cv2.imread(temp_path)
        if image is None:
            return jsonify({'success': False, 'error': 'cannot read image'}), 400
        h_orig, w_orig = image.shape[:2]

        # Detectar vaca → bbox
        r_cow = weight_estimator.coco_model(image, classes=[19], conf=0.2, verbose=False)
        if not r_cow or len(r_cow[0].boxes) == 0:
            return jsonify({'success': False, 'error': 'no se detectó vaca'}), 400
        boxes = r_cow[0].boxes.xyxy.cpu().numpy()
        scores = r_cow[0].boxes.conf.cpu().numpy()
        bi = int(np.argmax(scores))
        bx1, by1, bx2, by2 = [int(v) for v in boxes[bi]]
        pad = max(20, int(0.08 * max(bx2 - bx1, by2 - by1)))
        cx1 = max(0, bx1 - pad)
        cy1 = max(0, by1 - pad)
        cx2 = min(w_orig, bx2 + pad)
        cy2 = min(h_orig, by2 + pad)
        cow_crop = image[cy1:cy2, cx1:cx2]

        # Mask del BARRIL (modelo 3D = solo torso, sin patas/cabeza/cuello).
        # Unir TODAS las máscaras del barril por encima del ruido (cuando el
        # poste parte el torso, barril_seg devuelve 2 blobs separados — sin
        # unirlos el modelo 3D sale a la mitad).
        r_bar = barril_seg_model(cow_crop, conf=0.25, verbose=False)
        if not r_bar or r_bar[0].masks is None or len(r_bar[0].masks.data) == 0:
            return jsonify({'success': False, 'error': 'barril no detectado'}), 400
        masks = r_bar[0].masks.data.cpu().numpy()
        areas = np.array([float(np.sum(m)) for m in masks])
        max_area = float(areas.max()) if areas.size else 0.0
        if max_area <= 0:
            sil_mask = masks[int(np.argmax(areas))]
        else:
            keep = areas >= 0.05 * max_area
            sil_mask = np.max(masks[keep], axis=0)
        if sil_mask.shape != (cow_crop.shape[0], cow_crop.shape[1]):
            sil_mask = cv2.resize(sil_mask, (cow_crop.shape[1], cow_crop.shape[0]))
        binmask_full = np.zeros((h_orig, w_orig), dtype=np.uint8)
        binmask_full[cy1:cy2, cx1:cx2] = (sil_mask > 0.5).astype(np.uint8)

        # Reparar oclusiones verticales (p.ej. postes de escala que cortan
        # el barril). Misma lógica que /detect_cow_fast — interpola top/bot
        # desde los vecinos válidos. Sin esto el PLY sale con la muesca.
        cols_reparadas = _reparar_mascara_oclusion(binmask_full)
        if cols_reparadas:
            print(f"[generate_3d_from_frame] barril reparado: {len(cols_reparadas)} columnas")

        # Calcular escala cm/px en la posición de la vaca (cow_cx, bbox_y2)
        oc = locked_reference.get('original_coords') or {}
        _p1 = oc.get('post1') if oc else locked_reference.get('post1', {})
        _p2 = oc.get('post2') if oc else locked_reference.get('post2', {})
        if not _p1 or not _p2:
            return jsonify({'success': False, 'error': 'ref incompleta'}), 400
        _cx1 = float(_p1.get('cx', 0))
        _cx2 = float(_p2.get('cx', 0))
        _tape1 = float(_p1.get('tape_px', 0))
        _tape2 = float(_p2.get('tape_px', 0))
        if _cx1 > _cx2:
            _cx1, _cx2 = _cx2, _cx1
            _tape1, _tape2 = _tape2, _tape1
        cow_cx_val = (bx1 + bx2) / 2.0
        p_x = (cow_cx_val - _cx1) / max(1e-6, (_cx2 - _cx1))
        p_x_cl = max(0.0, min(1.0, p_x))
        cm_per_px = (1 - p_x_cl) * (112.0 / _tape1) + p_x_cl * (112.0 / _tape2)

        # Extraer contorno + puntos interiores + triangulación Delaunay
        # (mismo approach que generar_modelos3d_grandes.py — silueta real, no rebanadas genéricas)
        from scipy.spatial import Delaunay

        contours, _ = cv2.findContours(binmask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return jsonify({'success': False, 'error': 'sin contorno'}), 400
        contour = max(contours, key=cv2.contourArea)
        perim = cv2.arcLength(contour, True)
        contour_simple = cv2.approxPolyDP(contour, 0.002 * perim, True)
        pts_b_px = contour_simple.reshape(-1, 2).astype(float)

        # Grid de puntos interiores para triangulación densa
        ys, xs = np.where(binmask_full > 0)
        if xs.size == 0:
            return jsonify({'success': False, 'error': 'mask vacía'}), 400
        grid_step = max(8, int(0.02 * max(xs.max() - xs.min(), ys.max() - ys.min())))
        pts_i_px = []
        for gy in range(int(ys.min()), int(ys.max()) + 1, grid_step):
            for gx in range(int(xs.min()), int(xs.max()) + 1, grid_step):
                if binmask_full[gy, gx] > 0:
                    pts_i_px.append([float(gx), float(gy)])
        pts_i_px = np.array(pts_i_px) if pts_i_px else np.empty((0, 2))

        # Combinar boundary + interior, triangular con Delaunay
        all_px = np.vstack([pts_b_px, pts_i_px]) if len(pts_i_px) else pts_b_px
        all_px = np.unique(all_px, axis=0)
        if len(all_px) < 3:
            return jsonify({'success': False, 'error': 'pocos puntos'}), 400
        tri = Delaunay(all_px)
        # Filtrar triángulos cuyo centroide caiga dentro del mask
        tris_validos = []
        for s in tri.simplices:
            cx, cy = all_px[s].mean(axis=0).astype(int)
            if 0 <= cy < binmask_full.shape[0] and 0 <= cx < binmask_full.shape[1] \
                    and binmask_full[cy, cx] > 0:
                tris_validos.append(s)
        if not tris_validos:
            return jsonify({'success': False, 'error': 'sin triángulos válidos'}), 400
        tris_arr = np.array(tris_validos)

        # Colores desde la imagen original. Los puntos cuya columna X cayó en
        # zona reparada por el poste se pintan de NARANJA para distinguir
        # visualmente la sección corregida en el viewer 3D.
        COLOR_REPARADO = (255, 140, 0)
        colores = []
        for pt in all_px:
            ix = max(0, min(int(pt[0]), image.shape[1] - 1))
            iy = max(0, min(int(pt[1]), image.shape[0] - 1))
            if int(pt[0]) in cols_reparadas:
                colores.append(list(COLOR_REPARADO))
            else:
                b_ch, g_ch, r_ch = image[iy, ix]
                colores.append([int(r_ch), int(g_ch), int(b_ch)])
        colores = np.array(colores, dtype=np.uint8)

        # Convertir puntos px → cm (Y flip: arriba = y+)
        pts_cm = np.zeros_like(all_px, dtype=float)
        pts_cm[:, 0] = (all_px[:, 0] - all_px[:, 0].min()) * cm_per_px
        pts_cm[:, 1] = -(all_px[:, 1] - all_px[:, 1].min()) * cm_per_px
        pts_cm[:, 1] -= pts_cm[:, 1].min()  # base a y=0

        # Importar helpers del script batch
        sys_path_added = False
        try:
            import sys as _sys
            _proj = os.path.dirname(os.path.abspath(__file__))
            if _proj not in _sys.path:
                _sys.path.insert(0, _proj)
                sys_path_added = True
            from generar_modelos3d_grandes import guardar_ply
        finally:
            if sys_path_added:
                _sys.path.remove(_proj)

        proj_dir = _Path(os.path.dirname(os.path.abspath(__file__)))
        out_dir = proj_dir / MODELO_LIVE_DIR / cow_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Lateral: 2D silhueta (z=0) con colores
        ply_lat = out_dir / f'{cow_name}_lateral.ply'
        escala_info = f'Escala: {cm_per_px:.4f} cm/px | Alto: {altura_cm:.1f} cm'
        guardar_ply(str(ply_lat), pts_cm, tris_arr, colores, simetrico=False,
                    escala_info=escala_info)

        # 3D: silueta mirror en Z con profundidad elíptica → shell faithfull
        ply_3d = out_dir / f'{cow_name}_3d.ply'
        guardar_ply(str(ply_3d), pts_cm, tris_arr, colores, simetrico=True,
                    escala_info=escala_info)

        # Volumen PLY (rebanadas elípticas) desde el mismo contorno
        sys_path_added = False
        try:
            if _proj not in _sys.path:
                _sys.path.insert(0, _proj)
                sys_path_added = True
            from generar_ply_volumen import (
                rebanadas_desde_contorno, malla_elipsoidal, escribir_ply,
            )
        finally:
            if sys_path_added:
                _sys.path.remove(_proj)

        pts_b_cm = np.zeros_like(pts_b_px, dtype=float)
        pts_b_cm[:, 0] = (pts_b_px[:, 0] - pts_b_px[:, 0].min()) * cm_per_px
        pts_b_cm[:, 1] = -(pts_b_px[:, 1] - pts_b_px[:, 1].min()) * cm_per_px
        pts_b_cm[:, 1] -= pts_b_cm[:, 1].min()
        rebanadas = rebanadas_desde_contorno(pts_b_cm, n_slices=80)
        ply_vol = out_dir / f'{cow_name}_volumen.ply'
        if len(rebanadas) >= 3:
            vertices_v, tris_v = malla_elipsoidal(rebanadas, n_vert=32)
            escribir_ply(ply_vol, vertices_v, tris_v,
                         comentario=f'{cow_name} volumen rebanadas')

        # Resumen JSON para que modelos_disponibles lo liste con datos
        resumen = {
            'individuo': cow_name,
            'altura_real_cm': altura_cm,
            'vol_total_litros': None,
            'vol_barril_litros': barril_L if barril_L > 0 else None,
            'escala_cm_px': cm_per_px,
            'metodo': 'live_from_pass',
            'generado_desde_pasada': True,
        }
        with open(out_dir / f'{cow_name}_resumen.json', 'w') as rf:
            _json.dump(resumen, rf, indent=2)

        return jsonify({
            'success': True,
            'model_id': cow_name,
            'ply_3d': f'{cow_name}_3d.ply',
            'ply_volumen': f'{cow_name}_volumen.ply',
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


@app.route('/generate_result_card', methods=['POST'])
def generate_result_card():
    """Genera un PNG con: nombre de vaca, altura, volumen barril + imágenes de
    silueta y barril del frame representativo. Para descargar como resultado.
    """
    file = request.files.get('frame')
    if not file:
        return jsonify({'success': False, 'error': 'no frame'}), 400

    cow_name = (request.form.get('cow_name', '') or 'vaca').strip()
    altura_cm = request.form.get('altura_cm', '0')
    barril_L = request.form.get('barril_L', '0')
    frame_num = request.form.get('frame_num', '0')
    n_frames = request.form.get('n_frames', '0')

    temp_path = os.path.join(tempfile.gettempdir(), f'result_{uuid.uuid4().hex}.jpg')
    file.save(temp_path)

    try:
        image = cv2.imread(temp_path)
        if image is None:
            return jsonify({'success': False, 'error': 'cannot read image'}), 400
        h_orig, w_orig = image.shape[:2]

        # Detectar vaca (para crop común)
        if not weight_estimator or weight_estimator.coco_model is None:
            return jsonify({'success': False, 'error': 'cow model no disponible'}), 500
        r_cow = weight_estimator.coco_model(image, classes=[19], conf=0.2, verbose=False)
        if not r_cow or len(r_cow[0].boxes) == 0:
            return jsonify({'success': False, 'error': 'no se detectó vaca en el frame'}), 400
        boxes = r_cow[0].boxes.xyxy.cpu().numpy()
        scores = r_cow[0].boxes.conf.cpu().numpy()
        bi = int(np.argmax(scores))
        bx1, by1, bx2, by2 = [int(v) for v in boxes[bi]]
        pad = max(20, int(0.08 * max(bx2 - bx1, by2 - by1)))
        cx1 = max(0, bx1 - pad)
        cy1 = max(0, by1 - pad)
        cx2 = min(w_orig, bx2 + pad)
        cy2 = min(h_orig, by2 + pad)
        cow_crop = image[cy1:cy2, cx1:cx2]
        if cow_crop.size == 0:
            return jsonify({'success': False, 'error': 'crop vacío'}), 400

        def _render_mask_overlay(crop, model, color_bgr):
            if model is None:
                return None
            r = model(crop, conf=0.25, verbose=False)
            if not r or r[0].masks is None or r[0].masks.data is None or len(r[0].masks.data) == 0:
                return None
            masks = r[0].masks.data.cpu().numpy()
            areas = [float(np.sum(m)) for m in masks]
            best = masks[int(np.argmax(areas))]
            if best.shape != (crop.shape[0], crop.shape[1]):
                best = cv2.resize(best, (crop.shape[1], crop.shape[0]))
            binmask = (best > 0.5).astype(np.uint8)
            overlay = crop.copy()
            overlay[binmask > 0] = color_bgr
            blended = cv2.addWeighted(crop, 0.45, overlay, 0.55, 0)
            # Borde del contorno
            contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, color_bgr, 2)
            return blended

        silueta_render = _render_mask_overlay(cow_crop, silueta_seg_model, (255, 180, 0))  # cyan/azul
        barril_render = _render_mask_overlay(cow_crop, barril_seg_model, (0, 140, 230))    # naranja

        # Componer canvas final
        THUMB_H = 360
        def _fit_thumb(img):
            if img is None:
                ph = np.full((THUMB_H, int(THUMB_H * 1.3), 3), 240, dtype=np.uint8)
                cv2.putText(ph, 'sin mask', (20, THUMB_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                return ph
            s = THUMB_H / img.shape[0]
            w = int(img.shape[1] * s)
            return cv2.resize(img, (w, THUMB_H))

        sil_thumb = _fit_thumb(silueta_render)
        bar_thumb = _fit_thumb(barril_render)
        gap_w = 20
        imgs_w = sil_thumb.shape[1] + gap_w + bar_thumb.shape[1]
        canvas_w = max(800, imgs_w + 80)
        header_h = 130
        footer_h = 60
        canvas_h = header_h + THUMB_H + footer_h + 60

        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        # Header banner
        cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), (50, 70, 90), -1)
        cv2.putText(canvas, cow_name.upper(), (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(canvas, f'Altura promedio: {altura_cm} cm',
                    (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 235, 170), 2)
        try:
            bL = float(barril_L)
            if bL > 0:
                cv2.putText(canvas, f'Volumen barril: {barril_L} L',
                            (canvas_w // 2 + 20, 95), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 235, 170), 2)
        except Exception:
            pass

        # Imágenes: silueta a la izquierda, barril a la derecha
        img_y = header_h + 20
        x = (canvas_w - imgs_w) // 2
        canvas[img_y:img_y + THUMB_H, x:x + sil_thumb.shape[1]] = sil_thumb
        x2 = x + sil_thumb.shape[1] + gap_w
        canvas[img_y:img_y + THUMB_H, x2:x2 + bar_thumb.shape[1]] = bar_thumb
        # Labels debajo de cada imagen
        cv2.putText(canvas, 'SILUETA (cuerpo completo)',
                    (x + 10, img_y + THUMB_H + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 70, 90), 2)
        cv2.putText(canvas, 'BARRIL (torso/lomo)',
                    (x2 + 10, img_y + THUMB_H + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 70, 90), 2)

        # Footer con metadatos
        cv2.putText(canvas,
                    f'Frame {frame_num} · {n_frames} mediciones promediadas',
                    (30, canvas_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

        _, buf = cv2.imencode('.png', canvas)
        img_b64 = 'data:image/png;base64,' + base64.b64encode(buf).decode('utf-8')
        return jsonify({'success': True, 'image': img_b64})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


@app.route('/list_saved_folders', methods=['GET'])
def list_saved_folders():
    """Lista las carpetas en checkpoints/22abril/ con metadata (n_frames,
    central_frame, has_locked_reference, mtime)."""
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(proj_dir, 'checkpoints', '22abril')
    if not os.path.isdir(base):
        return jsonify({'success': True, 'folders': []})
    out = []
    for entry in sorted(os.listdir(base), reverse=True):
        full = os.path.join(base, entry)
        if not os.path.isdir(full):
            continue
        jpgs = [f for f in os.listdir(full) if f.lower().endswith('.jpg')]
        ctx = None
        ctx_path = os.path.join(full, 'context.json')
        if os.path.exists(ctx_path):
            try:
                import json as _json
                with open(ctx_path) as cf:
                    ctx = _json.load(cf)
            except Exception:
                ctx = None
        try:
            mtime = os.path.getmtime(full)
        except Exception:
            mtime = 0
        out.append({
            'name': entry,
            'n_frames': len(jpgs),
            'mtime': mtime,
            'central_frame': ctx.get('central_frame') if ctx else None,
            'has_locked_reference': bool(ctx and ctx.get('locked_reference')),
            'mode': ctx.get('mode') if ctx else None,
        })
    return jsonify({'success': True, 'folders': out})


@app.route('/list_saved_frames', methods=['POST'])
def list_saved_frames():
    """Lista los frames guardados en checkpoints/22abril/<folder>/.
    Devuelve [{file_name, frame_num, offset}] ordenado por offset (-10..+10).
    """
    folder = (request.form.get('folder', '') or '').strip()
    if not folder:
        return jsonify({'success': False, 'error': 'folder requerido'}), 400
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(proj_dir, 'checkpoints', '22abril', secure_filename(folder))
    if not os.path.isdir(full_path):
        return jsonify({'success': False, 'error': f'no existe {full_path}'}), 404
    items = []
    for fn in sorted(os.listdir(full_path)):
        if not fn.lower().endswith('.jpg'):
            continue
        # fname formato: frame_{m##|p##|000}_f<frame_num>.jpg
        parts = fn.replace('.jpg', '').split('_')
        if len(parts) < 3:
            continue
        sign_part, fpart = parts[1], parts[2]
        try:
            offset_abs = int(sign_part[1:]) if sign_part[0] in ('m', 'p') else int(sign_part)
            offset = -offset_abs if sign_part[0] == 'm' else offset_abs
            frame_num = int(fpart[1:]) if fpart.startswith('f') else int(fpart)
        except ValueError:
            continue
        items.append({'file_name': fn, 'frame_num': frame_num, 'offset': offset})
    items.sort(key=lambda x: x['offset'])
    # Cargar context.json si existe (locked_reference, central_frame, fps)
    ctx = None
    ctx_path = os.path.join(full_path, 'context.json')
    if os.path.exists(ctx_path):
        try:
            import json as _json
            with open(ctx_path) as cf:
                ctx = _json.load(cf)
        except Exception:
            ctx = None
    return jsonify({'success': True, 'folder': folder, 'frames': items, 'context': ctx})


@app.route('/saved_frame/<folder>/<filename>')
def saved_frame(folder, filename):
    """Sirve un frame guardado (para que el front pueda hacer fetch del blob)."""
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    safe_folder = secure_filename(folder)
    safe_file = secure_filename(filename)
    full_path = os.path.join(proj_dir, 'checkpoints', '22abril', safe_folder, safe_file)
    if not os.path.exists(full_path):
        return jsonify({'success': False, 'error': 'not found'}), 404
    return send_file(full_path, mimetype='image/jpeg')


@app.route('/save_frames_around', methods=['POST'])
def save_frames_around():
    """Guarda 21 frames (central ± 10) en checkpoints/22abril/<timestamp>/.

    DOS MODOS:
    1) MODO BACKEND (preferido): el cliente manda `video` (multipart) +
       `central_frame` + `fps`. El backend abre el video con cv2.VideoCapture
       y extrae los 21 frames con calidad 95 (lossless desde el codec).
       Esto evita la pérdida de calidad del flujo canvas+toBlob del browser
       que confunde al detector de vacas en frames con animales oscuros.
    2) MODO LEGACY: el cliente manda `frames[]` ya capturados desde canvas.
       Solo para compatibilidad; degrada calidad.

    Si el cliente manda `locked_reference_json`, se guarda en context.json
    para que después se pueda procesar sin re-marcar los postes.
    """
    from datetime import datetime as _dt
    import json as _json
    import tempfile
    try:
        central_frame_str = request.form.get('central_frame', '0')
        fps_str = request.form.get('fps', '30')
        try:
            central_frame = int(central_frame_str)
        except ValueError:
            central_frame = 0
        try:
            fps = float(fps_str)
        except ValueError:
            fps = 30.0

        proj_dir = os.path.dirname(os.path.abspath(__file__))
        base = os.path.join(proj_dir, 'checkpoints', '22abril')
        os.makedirs(base, exist_ok=True)
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f'central{central_frame}_{ts}'
        out_dir = os.path.join(base, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        saved = 0
        video_file = request.files.get('video')
        WINDOW = int(request.form.get('window', '10'))

        if video_file is not None:
            # MODO BACKEND: extraer frames del video con cv2 (calidad 95)
            ext = os.path.splitext(video_file.filename or '')[1] or '.mp4'
            tmp_path = tempfile.mktemp(suffix=ext)
            video_file.save(tmp_path)
            try:
                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    return jsonify({'success': False, 'error': 'no se pudo abrir el video en backend'}), 400
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                first = max(0, central_frame - WINDOW)
                last = min(total - 1, central_frame + WINDOW)
                for fnum in range(first, last + 1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    offset = fnum - central_frame
                    sign = 'm' if offset < 0 else ('p' if offset > 0 else '0')
                    abs_off = abs(offset)
                    fname = f'frame_{sign}{abs_off:02d}_f{fnum}.jpg' if offset != 0 else f'frame_000_f{fnum}.jpg'
                    cv2.imwrite(os.path.join(out_dir, fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
                cap.release()
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        else:
            # MODO LEGACY: blobs capturados desde canvas
            files = request.files.getlist('frames[]')
            if not files:
                return jsonify({'success': False, 'error': 'no se recibieron frames ni video'}), 400
            for f in files:
                fname = secure_filename(f.filename or f'frame_{saved}.jpg')
                f.save(os.path.join(out_dir, fname))
                saved += 1

        # context.json
        ctx = {
            'central_frame': central_frame,
            'fps': fps,
            'video_id': request.form.get('video_id', '') or None,
            'saved_at': _dt.now().isoformat(),
            'mode': 'backend' if video_file is not None else 'canvas',
        }
        lref_str = (request.form.get('locked_reference_json', '') or '').strip()
        if lref_str:
            try:
                ctx['locked_reference'] = _json.loads(lref_str)
            except Exception as e:
                ctx['locked_reference_error'] = str(e)
        with open(os.path.join(out_dir, 'context.json'), 'w') as cf:
            _json.dump(ctx, cf, indent=2)

        return jsonify({
            'success': True,
            'folder': out_dir,
            'folder_name': folder_name,
            'n_frames': saved,
            'central_frame': central_frame,
            'has_locked_reference': bool(lref_str),
            'mode': ctx['mode'],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/detect_cow_fast', methods=['POST'])
def detect_cow_fast():
    """Detección rápida de vaca + cálculo de altura usando la referencia fijada.
    Sin keypoints, sin postes, sin peso. ~150-250 ms por llamada.

    Form params: frame, video_id, locked_reference_json (post1, post2, original_coords opcional)
    Retorna: {success, detected, animal_bbox_original, cow_height_cm, cm_per_px, p, video_w, video_h}
    """
    file = request.files.get('frame')
    if not file:
        return jsonify({'success': False, 'error': 'no frame'}), 400

    video_id = request.form.get('video_id', '').strip() or None
    inline = request.form.get('locked_reference_json', '').strip()

    # Priorizar el inline (es más completo — incluye original_coords del calibrate).
    # El cached solo guarda post1/post2 en coords resized, sin original_coords.
    locked_reference = None
    if inline:
        try:
            import json as _json
            locked_reference = _json.loads(inline)
        except Exception as _e:
            print(f"[detect_cow_fast] parse inline fail: {_e}")
    if not locked_reference and video_id:
        locked_reference = _locked_references.get(video_id)

    if not locked_reference:
        return jsonify({'success': False, 'error': 'no locked reference'}), 400

    if not weight_estimator or weight_estimator.coco_model is None:
        return jsonify({'success': False, 'error': 'cow model no disponible'}), 500

    temp_path = os.path.join(tempfile.gettempdir(), f'fast_{uuid.uuid4().hex}.jpg')
    file.save(temp_path)

    try:
        image = cv2.imread(temp_path)
        if image is None:
            return jsonify({'success': False, 'error': 'cannot read image'}), 400
        h_orig, w_orig = image.shape[:2]

        # Usamos la MISMA cascada de detección que estimate_weight (manual)
        # para que los bboxes coincidan entre flow manual y automático.
        img_resized, resized_image, img_rgb, scale_factor, pad_x, pad_y, w_orig_chk, h_orig_chk = \
            weight_estimator._load_and_resize(temp_path, lambda m: None)
        # Nota: _detect_all_cows usa mapping interno y retorna bboxes en coords RESIZED (letterbox).
        _boxes, _kps, _scores, _classes, _det = weight_estimator._detect_all_cows(
            img_resized, resized_image, scale_factor, pad_x, pad_y, w_orig_chk, h_orig_chk,
            None, lambda m: None
        )

        if not _det or _boxes is None or len(_boxes) == 0:
            return jsonify({'success': True, 'detected': False,
                            'video_w': w_orig, 'video_h': h_orig})

        # Elegir el bbox de mayor score
        if _scores is not None and len(_scores) > 0:
            best_idx = int(np.argmax(_scores))
            cow_score = float(_scores[best_idx])
        else:
            best_idx = 0
            cow_score = 0.0
        rx1, ry1, rx2, ry2 = [float(v) for v in _boxes[best_idx]]
        # Bbox YOLO original (sin recortes de silueta_seg/barril_seg). Lo
        # usamos para el chequeo de solapamiento con el poste cercano: el
        # bbox COW original siempre incluye cabeza/patas, así no se nos
        # escapa un poste que cae dentro del cuerpo aunque barril_seg haya
        # apretado los laterales.
        rx1_yolo, rx2_yolo = rx1, rx2

        # Ajustar BOTTOM (ry2) usando silueta_seg mask (bottom = pezuñas).
        silueta_bottom_used = False
        ry2_original = ry2  # para sanity check
        if silueta_seg_model is not None:
            try:
                rx1_i, ry1_i, rx2_i, ry2_i = int(rx1), int(ry1), int(rx2), int(ry2)
                bbox_h_yolo = max(1, ry2_i - ry1_i)
                spad = max(10, int(0.05 * bbox_h_yolo))
                sx1 = max(0, rx1_i - spad)
                sy1 = max(0, ry1_i - spad)
                sx2 = min(resized_image.shape[1], rx2_i + spad)
                sy2 = min(resized_image.shape[0], ry2_i + spad)
                s_crop = resized_image[sy1:sy2, sx1:sx2]
                if s_crop.size > 0:
                    r_sil = silueta_seg_model(s_crop, conf=0.25, verbose=False)
                    if (r_sil and r_sil[0].masks is not None
                            and r_sil[0].masks.data is not None
                            and len(r_sil[0].masks.data) > 0):
                        s_masks = r_sil[0].masks.data.cpu().numpy()
                        s_areas = [float(np.sum(m)) for m in s_masks]
                        best_sil = s_masks[int(np.argmax(s_areas))]
                        smh, smw = best_sil.shape
                        if (smh, smw) != (s_crop.shape[0], s_crop.shape[1]):
                            best_sil = cv2.resize(best_sil, (s_crop.shape[1], s_crop.shape[0]))
                        bin_sil = (best_sil > 0.5).astype(np.uint8)
                        rows_sil = np.where(bin_sil.sum(axis=1) > 0)[0]
                        if rows_sil.size > 0:
                            bottom_y_in_crop = int(rows_sil[-1])
                            new_ry2 = float(sy1 + bottom_y_in_crop)
                            # Sanity: el bottom de silueta no puede estar a más de
                            # 30% del alto del bbox YOLO. Si se va, descartamos.
                            diff = new_ry2 - ry2_original
                            max_diff = 0.30 * bbox_h_yolo
                            if abs(diff) <= max_diff:
                                ry2 = new_ry2
                                silueta_bottom_used = True
                            else:
                                print(f"[detect_cow_fast] silueta bottom rechazado: "
                                      f"diff={diff:.0f}px > max={max_diff:.0f}px")
            except Exception as _e:
                print(f"[detect_cow_fast] silueta_seg fallo: {_e}")

        # Recortar TOP del bbox al top del barril (lomo/cruz) usando barril_seg.
        # El bottom (ry2) ya fue ajustado con KP5 (o sigue siendo el de YOLO).
        # Si barril_seg no está cargado o no encuentra mask, dejamos ry1 original.
        barril_top_used = False
        barril_binmask = None  # para cálculo de volumen posterior
        barril_crop_origin = None  # (cx1_i, cy1_i) de donde está la mask
        if barril_seg_model is not None:
            try:
                # Trabajamos con coords RESIZED (la imagen es resized_image)
                rx1_i, ry1_i, rx2_i, ry2_i = int(rx1), int(ry1), int(rx2), int(ry2)
                cpad = max(10, int(0.05 * (ry2_i - ry1_i)))
                cx1_i = max(0, rx1_i - cpad)
                cy1_i = max(0, ry1_i - cpad)
                cx2_i = min(resized_image.shape[1], rx2_i + cpad)
                cy2_i = min(resized_image.shape[0], ry2_i + cpad)
                crop_img = resized_image[cy1_i:cy2_i, cx1_i:cx2_i]
                if crop_img.size > 0:
                    r_barril = barril_seg_model(crop_img, conf=0.25, verbose=False)
                    if (r_barril and r_barril[0].masks is not None
                            and r_barril[0].masks.data is not None
                            and len(r_barril[0].masks.data) > 0):
                        masks_data = r_barril[0].masks.data.cpu().numpy()
                        # Unir TODAS las máscaras del barril por encima del ruido.
                        # Cuando un poste parte el torso, barril_seg devuelve 2
                        # blobs separados; argmax se quedaba con uno solo y el
                        # modelo salía a la mitad. Combinamos todo el cow_crop.
                        areas_arr = np.array([float(np.sum(m)) for m in masks_data])
                        max_area = float(areas_arr.max()) if areas_arr.size else 0.0
                        if max_area <= 0:
                            best_mask = masks_data[int(np.argmax(areas_arr))]
                        else:
                            keep = areas_arr >= 0.05 * max_area
                            best_mask = np.max(masks_data[keep], axis=0)
                        # Las máscaras de YOLO pueden venir en diferente resolución
                        mh, mw = best_mask.shape
                        if (mh, mw) != (crop_img.shape[0], crop_img.shape[1]):
                            best_mask = cv2.resize(best_mask, (crop_img.shape[1], crop_img.shape[0]))
                        binmask = (best_mask > 0.5).astype(np.uint8)
                        rows_with_mask = np.where(binmask.sum(axis=1) > 0)[0]
                        if rows_with_mask.size > 0:
                            top_y_in_crop = int(rows_with_mask[0])
                            ry1 = float(cy1_i + top_y_in_crop)  # top del barril
                            barril_top_used = True
                            # Opcional: también recortar laterales al barril
                            cols_with_mask = np.where(binmask.sum(axis=0) > 0)[0]
                            if cols_with_mask.size > 0:
                                rx1 = float(cx1_i + int(cols_with_mask[0]))
                                rx2 = float(cx1_i + int(cols_with_mask[-1]))
                            # Guardar la mask para calcular volumen del barril después
                            barril_binmask = binmask
                            barril_crop_origin = (cx1_i, cy1_i)
            except Exception as _e:
                print(f"[detect_cow_fast] barril_seg fallo: {_e}")

        if not barril_top_used:
            print(f"[detect_cow_fast] barril_seg no disponible o sin mask → bbox sin modificar (puede incluir cabeza)")

        # Los bboxes están en coords RESIZED (letterbox). Convertir a ORIGINAL.
        if scale_factor and scale_factor > 0:
            x1 = (rx1 - pad_x) / scale_factor
            y1 = (ry1 - pad_y) / scale_factor
            x2 = (rx2 - pad_x) / scale_factor
            y2 = (ry2 - pad_y) / scale_factor
            x1_yolo = (rx1_yolo - pad_x) / scale_factor
            x2_yolo = (rx2_yolo - pad_x) / scale_factor
        else:
            x1, y1, x2, y2 = rx1, ry1, rx2, ry2
            x1_yolo, x2_yolo = rx1_yolo, rx2_yolo
        cow_cx = (x1 + x2) / 2.0
        bbox_h = y2 - y1

        # Sacar postes de la ref. Preferir original_coords si viene
        oc = locked_reference.get('original_coords') or {}
        _p1 = oc.get('post1') if oc else locked_reference.get('post1', {})
        _p2 = oc.get('post2') if oc else locked_reference.get('post2', {})
        if not _p1 or not _p2:
            return jsonify({'success': False, 'error': 'locked_reference incompleto'}), 400

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

        if cx2 == cx1 or tape_px_1 <= 0 or tape_px_2 <= 0:
            return jsonify({'success': False, 'error': 'locked_reference invalido'}), 400

        # Si el poste más cercano (= mayor tape_px, más pixeles por cm) cae
        # dentro del bbox YOLO original de la vaca, el setup interfiere con
        # el modelado del barril. Usamos el bbox YOLO (no el recortado por
        # barril_seg) porque cuando la mask se parte por el poste, los
        # laterales se aprietan y podríamos perder casos donde el poste
        # quedó visualmente "dentro" del cuerpo. Mantenemos el ajuste de
        # top (barril_top_used) para que la altura siga precisa, pero
        # descartamos la mask para que no alimente contorno/volumen ni el
        # consenso multi-frame.
        closer_post_cx = cx1 if tape_px_1 >= tape_px_2 else cx2
        barril_post_overlap = (x1_yolo <= closer_post_cx <= x2_yolo)
        if barril_post_overlap and barril_binmask is not None:
            print(f"[detect_cow_fast] bbox YOLO solapa poste cercano "
                  f"(cx={closer_post_cx:.0f}, bbox_yolo=[{x1_yolo:.0f},{x2_yolo:.0f}]) "
                  f"→ descarto mask del barril")
            barril_binmask = None

        # Posicion horizontal (lateral): referencia, no usada para escala
        p_x = (cow_cx - cx1) / (cx2 - cx1)

        # INTERSECCIÓN SEGMENTO-SEGMENTO en el rango X COMÚN:
        #   Segmento A (floor): (cx1, floor1) → (cx2, floor2)
        #   Segmento B (bbox bottom): (x1, y2) → (x2, y2)
        # Los segmentos se cruzan en un punto si:
        #   1) Sus rangos X se superponen
        #   2) En ese rango común, la línea del piso pasa por y=y2 (cambia de lado)
        # Ya NO hacemos early-return cuando no hay cruce — solo lo marcamos
        # con within_rectangle=False. La altura pierde precisión sin cruce,
        # pero el barril_contour_norm sigue siendo útil (tops/bottoms cm
        # contra y2=pies, con escala interpolada por p_x).
        within_rectangle = True
        cruce_reason = None
        x_cross = None
        X_lo = max(x1, cx1)
        X_hi = min(x2, cx2)

        if X_lo > X_hi:
            within_rectangle = False
            cruce_reason = (f'rangos X no se superponen: bbox=[{x1:.0f},{x2:.0f}] '
                            f'floor=[{cx1:.0f},{cx2:.0f}]')
        else:
            if abs(cx2 - cx1) < 0.5:
                fy_lo = (floor1 + floor2) / 2
                fy_hi = fy_lo
            else:
                fy_lo = floor1 + (X_lo - cx1) / (cx2 - cx1) * (floor2 - floor1)
                fy_hi = floor1 + (X_hi - cx1) / (cx2 - cx1) * (floor2 - floor1)
            d_lo = fy_lo - y2
            d_hi = fy_hi - y2
            if d_lo * d_hi > 1e-6:
                within_rectangle = False
                cruce_reason = (f'floor no cruza bbox_bottom(y={y2:.0f}) en rango X común '
                                f'[{X_lo:.0f},{X_hi:.0f}]')
            else:
                if abs(d_lo - d_hi) > 1e-6:
                    alpha = d_lo / (d_lo - d_hi)
                    x_cross = X_lo + alpha * (X_hi - X_lo)
                else:
                    x_cross = (X_lo + X_hi) / 2

        # t (posición a lo largo del piso para interpolar escala entre postes).
        # Si hubo cruce válido, usa esa X. Si no, fallback al centro del bbox
        # de la vaca (p_x), clampeado al rango [0,1].
        if x_cross is not None and abs(cx2 - cx1) > 0.5:
            t = (x_cross - cx1) / (cx2 - cx1)
            t = max(0.0, min(1.0, t))
        else:
            t = max(0.0, min(1.0, p_x))

        # Escala en el PUNTO DE CRUCE: t es la posición a lo largo del piso, que también
        # interpola la escala entre los postes (post 1 a t=0, post 2 a t=1)
        scale_1 = 112.0 / tape_px_1
        scale_2 = 112.0 / tape_px_2
        cm_per_px = (1 - t) * scale_1 + t * scale_2

        if cm_per_px <= 0:
            return jsonify({
                'success': True, 'detected': True, 'within_rectangle': False,
                'reason': f'cm_per_px={cm_per_px:.4f} inválido',
                'animal_bbox_original': [x1, y1, x2, y2],
                'video_w': w_orig, 'video_h': h_orig,
            })

        cow_height_cm = bbox_h * cm_per_px
        if cow_height_cm <= 0:
            return jsonify({
                'success': True, 'detected': True, 'within_rectangle': False,
                'reason': f'altura={cow_height_cm:.1f} inválida',
                'animal_bbox_original': [x1, y1, x2, y2],
                'video_w': w_orig, 'video_h': h_orig,
            })

        y_cross = y2  # por definición del segmento horizontal de la vaca

        # VOLUMEN DEL BARRIL + CONTORNO NORMALIZADO
        # El contorno se samplea a 60 posiciones equiespaciadas en X, cada una
        # con la altura del barril en cm. Permite combinar N frames posteriormente
        # (barril consenso). Todas las mediciones están en cm reales.
        barril_volumen_litros = None
        barril_contour_norm = None
        barril_cols_rellenadas = 0
        if barril_binmask is not None and cm_per_px > 0 and scale_factor > 0:
            try:
                cm_per_resized_px = cm_per_px / scale_factor
                K_DEPTH = 0.25
                bh_mask, bw_mask = barril_binmask.shape
                barril_cols_rellenadas = len(_reparar_mascara_oclusion(barril_binmask))

                vol_cm3 = 0.0
                # Volumen por rebanadas (todas las columnas)
                for col_x in range(bw_mask):
                    col = barril_binmask[:, col_x]
                    rows = np.where(col > 0)[0]
                    if rows.size == 0:
                        continue
                    h_px_resized = rows[-1] - rows[0] + 1
                    h_cm = h_px_resized * cm_per_resized_px
                    a = h_cm / 2.0
                    b = h_cm * K_DEPTH
                    area = np.pi * a * b
                    vol_cm3 += area * cm_per_resized_px
                barril_volumen_litros = round(vol_cm3 / 1000.0, 1)

                # Contorno normalizado: N muestras equiespaciadas entre los bordes X.
                # Guardamos heights_cm + tops_cm + bottoms_cm (distancia en cm desde
                # el piso=y2 hasta el lomo y la barriga) para poder reconstruir la
                # silueta real (asimétrica) en el consenso multi-frame.
                cols_any = np.where(barril_binmask.sum(axis=0) > 0)[0]
                if cols_any.size > 1 and barril_crop_origin is not None:
                    x_min = int(cols_any[0])
                    x_max = int(cols_any[-1])
                    width_px = x_max - x_min + 1
                    width_cm = width_px * cm_per_resized_px
                    N_SAMPLES = 60
                    heights_cm = []
                    tops_cm = []
                    bottoms_cm = []
                    _crop_cy1 = barril_crop_origin[1]
                    for i in range(N_SAMPLES):
                        x_px = int(x_min + (width_px - 1) * i / (N_SAMPLES - 1))
                        col = barril_binmask[:, x_px]
                        col_rows = np.where(col > 0)[0]
                        if col_rows.size == 0:
                            heights_cm.append(0.0)
                            tops_cm.append(0.0)
                            bottoms_cm.append(0.0)
                        else:
                            top_row = int(col_rows[0])   # lomo
                            bot_row = int(col_rows[-1])  # barriga
                            top_y_resized = _crop_cy1 + top_row
                            bot_y_resized = _crop_cy1 + bot_row
                            top_y_orig = (top_y_resized - pad_y) / scale_factor
                            bot_y_orig = (bot_y_resized - pad_y) / scale_factor
                            top_above_floor = (y2 - top_y_orig) * cm_per_px
                            bot_above_floor = (y2 - bot_y_orig) * cm_per_px
                            heights_cm.append(round(top_above_floor - bot_above_floor, 2))
                            tops_cm.append(round(top_above_floor, 2))
                            bottoms_cm.append(round(bot_above_floor, 2))
                    barril_contour_norm = {
                        'n_samples': N_SAMPLES,
                        'width_cm': round(width_cm, 2),
                        'heights_cm': heights_cm,
                        'tops_cm': tops_cm,
                        'bottoms_cm': bottoms_cm,
                    }
            except Exception as _e:
                print(f"[detect_cow_fast] barril volumen/contour fail: {_e}")

        # ALINEACIÓN GEOMÉTRICA: chequear si bbox_y2 está cerca de la línea del
        # piso en la columna de la vaca. Si sí → bbox bottom está bien en los
        # pies (medición confiable). Si el bbox se extiende muy por debajo del
        # piso → altura inflada.
        p_x_clamped = max(0.0, min(1.0, p_x))
        floor_at_cow_cx = floor1 + p_x_clamped * (floor2 - floor1)
        y_diff_floor = y2 - floor_at_cow_cx  # + = bbox bajo piso, - = bbox arriba
        bbox_h_px = max(1.0, y2 - y1)
        BBOX_ALIGN_TOLERANCE = 0.03  # 3% del alto del bbox
        bbox_aligned = abs(y_diff_floor) <= BBOX_ALIGN_TOLERANCE * bbox_h_px

        # La altura se reporta SOLO en frames con poste solapado: ahí el
        # bbox de la vaca contiene el poste cercano, así que cm_per_px en
        # ese punto es el del poste (escala precisa, 112cm contra el mayor
        # tape_px). En los frames sin solapamiento no devolvemos altura;
        # esos frames van al consenso del barril.
        return jsonify({
            'success': True,
            'detected': True,
            'within_rectangle': within_rectangle,
            'cruce_reason': cruce_reason,
            'animal_bbox_original': [x1, y1, x2, y2],
            'cow_score': cow_score,
            'cow_height_cm': round(cow_height_cm, 1) if barril_post_overlap else None,
            'cm_per_px': round(cm_per_px, 5),
            'p': round(p_x, 3),
            't_floor': round(t, 3),
            'x_cross': round(x_cross, 1) if x_cross is not None else None,
            'y_cross': round(y_cross, 1),
            'silueta_bottom_used': silueta_bottom_used,
            'barril_top_used': barril_top_used,
            'barril_post_overlap': barril_post_overlap,
            'barril_volumen_litros': barril_volumen_litros,
            'barril_contour_norm': barril_contour_norm,  # para consenso multi-frame
            'barril_cols_rellenadas': barril_cols_rellenadas,
            'bbox_aligned_with_floor': bbox_aligned,
            'y_diff_floor': round(y_diff_floor, 1),
            'video_w': w_orig,
            'video_h': h_orig,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


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

    # video_id + ref inline. Priorizamos el inline (trae original_coords).
    video_id = request.form.get('video_id', '').strip() or None
    locked_ref_inline = request.form.get('locked_reference_json', '').strip()
    locked_reference = None
    if locked_ref_inline:
        try:
            import json as _json
            parsed = _json.loads(locked_ref_inline)
            if parsed.get('post1') and parsed.get('post2'):
                locked_reference = parsed
                if video_id:
                    _locked_references[video_id] = parsed  # cacheo completo
        except Exception as _e:
            print(f"[analyze] failed parsing locked_reference_json: {_e}")
    if not locked_reference and video_id:
        locked_reference = _locked_references.get(video_id)

    _override_cm_per_px = None  # se setea más abajo si hay locked_reference

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
            locked_reference=locked_reference,
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
                # Rectángulo calculado este frame (para que la UI lo mande a /lock_reference)
                'rectangle_ref': details.get('rectangle_ref') if isinstance(details, dict) else None,
                # Si se usó una referencia fijada
                'locked_ref_used': locked_reference is not None,
                'video_id': video_id,
                # Bbox de la vaca para overlay en vivo
                'animal_bbox_original': details.get('animal_bbox_original') if isinstance(details, dict) else None,
                'video_w': details.get('video_w') if isinstance(details, dict) else None,
                'video_h': details.get('video_h') if isinstance(details, dict) else None,
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
                        calc_cm_per_px = 112.0 / avg_post_height_px
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
                'volumen_litros': _js(sfm_result.get('volumen_barril_litros', sfm_result['volumen_litros'])),
                'peso_kg': _js(sfm_result.get('peso_barril_kg', sfm_result['peso_kg'])),
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
MODELO_LIVE_DIR = 'output_modelos3d_live'
ALTO_ESTIMADO_DEFAULT = 120.0  # cm fallback assumed during model generation


def _discover_modelo_dirs():
    """Auto-discover model subdirectories inside all model output dirs."""
    base = os.path.dirname(os.path.abspath(__file__))
    dirs = {}
    for model_dir in [MODELO_RECORTE26MARZ_DIR, MODELO_LIVE_DIR]:
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
        ply_vol = None
        for f in sorted(os.listdir(dir_path)):
            fl = f.lower()
            if not f.endswith('.ply'):
                continue
            # Ignorar backups generados por el script de reparación de malla
            if '_orig.ply' in fl or fl.endswith('.bak') or '.bak.' in fl:
                continue
            if 'volumen' in fl:
                ply_vol = f
            elif 'lateral' in fl:
                ply_lat = f
            elif '3d' in fl:
                ply_3d = f
        if not ply_3d and not ply_lat:
            continue
        meta = _load_resumen(dir_path)
        # El volumen/peso mostrados son SIEMPRE del barril (fallback a total si no existe)
        vol_barril = meta.get('volumen_barril_litros') or meta.get('vol_barril_litros')
        peso_barril = meta.get('peso_barril_kg')
        vol = vol_barril if vol_barril is not None else (meta.get('volumen_litros') or meta.get('vol_total_litros'))
        peso_real = meta.get('peso_real_kg')
        if peso_real is not None:
            peso = peso_real
        elif peso_barril is not None:
            peso = peso_barril
        elif vol_barril is not None:
            peso = round(float(vol_barril) * 1.03, 1)  # derivar del barril cuando no hay peso_barril guardado
        else:
            peso = meta.get('peso_kg')
        modelos.append({
            'id': vaca,
            'nombre': vaca.replace('_', ' ').title(),
            'ply_3d': ply_3d,
            'ply_lateral': ply_lat,
            'ply_volumen': ply_vol,
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

    vol_base = meta.get('volumen_barril_litros')
    if vol_base is None:
        vol_base = meta.get('vol_barril_litros')
    if vol_base is None:
        vol_base = meta.get('volumen_litros', meta.get('volumen_cm3', 0) / 1000)
    volumen_litros = round(float(vol_base) * factor3, 1)
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
    # Puerto vía env var PORT (default 5001 — evita conflicto con AirPlay en macOS).
    # 0.0.0.0 evita problemas de resolución con localhost.
    # Para correr varias instancias en paralelo: PORT=5002 python app.py
    _port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=_port, debug=True)

# TODO: atrás para otra granja
# TODO: Página de despedida
