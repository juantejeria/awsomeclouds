# Extraído de app.py — rutas del consenso 3D en la UI

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
        largo_cm = float(data.get('largo_cm') or 0)
        barril_dir = (data.get('barril_dir', '') or '').strip().lower()
        if barril_dir not in ('left', 'right'):
            barril_dir = 'unknown'
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

        # El volumen consenso es el volumen ENCERRADO de la malla _3d.ply
        # (silueta espejada con profundidad elíptica) que dejó
        # /generate_3d_from_frame. Única fuente de volumen: ya no se generan
        # rebanadas/cilindros ni _volumen.ply. La mediana multi-frame se
        # conserva solo como metadato (ancho consenso, frames usados).
        sys_path_added = False
        try:
            import sys as _sys
            _proj = os.path.dirname(os.path.abspath(__file__))
            if _proj not in _sys.path:
                _sys.path.insert(0, _proj)
                sys_path_added = True
            from core.generar_modelos3d_grandes import volumen_ply_cerrado
        finally:
            if sys_path_added:
                _sys.path.remove(_proj)

        proj_dir = _Path(os.path.dirname(os.path.abspath(__file__)))
        out_dir = proj_dir / MODELO_LIVE_DIR / cow_name
        ply_3d = out_dir / f'{cow_name}_3d.ply'
        if not ply_3d.is_file():
            return jsonify({'success': False,
                            'error': 'falta _3d.ply (genera el modelo del frame primero)'}), 400
        barril_consenso_L = volumen_ply_cerrado(str(ply_3d))

        out_dir.mkdir(parents=True, exist_ok=True)
        # Conservar el sentido ya auto-detectado por /generate_3d_from_frame en
        # vez de pisarlo con 'unknown' cuando el request no lo trae.
        if barril_dir == 'unknown':
            try:
                _prev = out_dir / f'{cow_name}_resumen.json'
                if _prev.is_file():
                    with open(_prev) as _pf:
                        _pdir = (_json.load(_pf).get('barril_dir') or 'unknown')
                    if _pdir in ('left', 'right'):
                        barril_dir = _pdir
            except Exception:
                pass
        resumen = {
            'individuo': cow_name,
            'altura_real_cm': altura_cm,
            'vol_barril_litros': barril_consenso_L,
            'largo_cm': round(largo_cm, 1) if largo_cm > 0 else None,
            'barril_dir': barril_dir,
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
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def _seg_mask_crop(model, crop):
    """Corre un modelo de segmentación sobre el crop y une las máscaras grandes."""
    r = model(crop, conf=0.25, verbose=False)
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        return None
    m = r[0].masks.data.cpu().numpy()
    a = np.array([float(x.sum()) for x in m])
    if a.max() <= 0:
        return None
    k = a >= 0.05 * a.max()
    s = np.max(m[k], axis=0)
    if s.shape != crop.shape[:2]:
        s = cv2.resize(s, (crop.shape[1], crop.shape[0]))
    return (s > 0.5).astype(np.uint8)


def _detectar_sentido_barril(cow_crop, barril_mask_crop):
    """Sentido del animal a partir de las máscaras (coords del crop).

    La cabeza+cuello sobresalen del barril en la mitad SUPERIOR del cuerpo: el
    lado (izq/der) con más masa de silueta por fuera del barril es la cabeza.
    Mismo criterio que muestra_lomo_cruz_cola.py (head_left). Devuelve
    'left'/'right' (cabeza a ese lado) o 'unknown' si no se puede decidir.
    """
    if silueta_seg_model is None or barril_mask_crop is None:
        return 'unknown'
    sil = _seg_mask_crop(silueta_seg_model, cow_crop)
    if sil is None:
        return 'unknown'
    cols = np.where(barril_mask_crop.sum(0) > 0)[0]
    rows = np.where(barril_mask_crop.sum(1) > 0)[0]
    if not len(cols) or not len(rows):
        return 'unknown'
    bxmin, bxmax = int(cols[0]), int(cols[-1])
    btop, bbot = int(rows[0]), int(rows[-1])
    bmid = (btop + bbot) // 2
    lm = int(sil[btop:bmid, :bxmin].sum())
    rm = int(sil[btop:bmid, bxmax + 1:].sum())
    if lm == rm:
        return 'unknown'
    return 'left' if lm > rm else 'right'




@app.route('/generate_3d_from_frame', methods=['POST'])
def generate_3d_from_frame():
    """Genera PLYs (_3d, _lateral) de la vaca a partir del frame
    representativo + silueta_seg + locked_reference. Guarda en
    output_modelos3d_live/<cow_name>/ para que el viewer 3D lo muestre.
    El volumen reportado es el volumen encerrado de la malla _3d.ply.
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
    try:
        largo_cm = float(request.form.get('largo_cm', 0) or 0)
    except Exception:
        largo_cm = 0.0
    barril_dir = (request.form.get('barril_dir', '') or '').strip().lower()
    if barril_dir not in ('left', 'right'):
        barril_dir = 'unknown'

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

        # Sentido del animal: si no vino del form, auto-detectar (silueta vs
        # barril). El viewer lo usa para ubicar el diámetro torácico del lado de
        # la cabeza. El PLY no espeja X, así que cabeza a la izq de la imagen =
        # 'left' (frente en xMin del modelo).
        if barril_dir == 'unknown':
            try:
                auto_dir = _detectar_sentido_barril(cow_crop, (sil_mask > 0.5).astype(np.uint8))
                if auto_dir in ('left', 'right'):
                    barril_dir = auto_dir
                    print(f"[generate_3d_from_frame] sentido auto-detectado: {barril_dir}")
                else:
                    print("[generate_3d_from_frame] sentido no determinado (unknown)")
            except Exception as _e:
                print(f"[generate_3d_from_frame] auto-sentido falló: {_e}")

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
        cm_per_px = (1 - p_x_cl) * (VARA_CM / _tape1) + p_x_cl * (VARA_CM / _tape2)

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
            from core.generar_modelos3d_grandes import guardar_ply, volumen_malla_cerrada
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

        # 3D: silueta mirror en Z con profundidad elíptica → shell faithfull.
        # El volumen reportado es el volumen ENCERRADO de esta malla cerrada
        # (única fuente de volumen; ya no se generan rebanadas/cilindros).
        ply_3d = out_dir / f'{cow_name}_3d.ply'
        pts_3d, tris_3d = guardar_ply(str(ply_3d), pts_cm, tris_arr, colores,
                                      simetrico=True, escala_info=escala_info)
        vol_barril_L = volumen_malla_cerrada(pts_3d, tris_3d)

        # Resumen JSON para que modelos_disponibles lo liste con datos
        resumen = {
            'individuo': cow_name,
            'altura_real_cm': altura_cm,
            'vol_total_litros': None,
            'vol_barril_litros': vol_barril_L if vol_barril_L > 0 else None,
            'largo_cm': round(largo_cm, 1) if largo_cm > 0 else None,
            'barril_dir': barril_dir,
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
            'vol_barril_litros': vol_barril_L,
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


