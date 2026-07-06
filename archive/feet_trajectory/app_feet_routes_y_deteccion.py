# Extraído de app.py — rutas y cálculo de trayectoria de pezuñas

@app.route('/saved_feet_map/<folder>')
def saved_feet_map(folder):
    """Sirve el feet_map.png de una carpeta guardada."""
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    safe_folder = secure_filename(folder)
    full_path = os.path.join(proj_dir, 'checkpoints', SAVED_FRAMES_DATASET, safe_folder, 'feet_map.png')
    if not os.path.exists(full_path):
        return jsonify({'success': False, 'error': 'not found'}), 404
    return send_file(full_path, mimetype='image/png')




# =====

@app.route('/save_feet_map', methods=['POST'])
def save_feet_map():
    """Guarda el mapa de trayectoria de pezuñas (PNG + JSON) en la carpeta de
    frames guardados. Cuerpo JSON: {folder, png_data_url, payload}.
    """
    import base64
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({'success': False, 'error': 'json inválido'}), 400
    folder = (body.get('folder') or '').strip()
    png_url = body.get('png_data_url') or ''
    payload = body.get('payload') or {}
    if not folder or not png_url:
        return jsonify({'success': False, 'error': 'folder y png_data_url requeridos'}), 400
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(proj_dir, 'checkpoints', SAVED_FRAMES_DATASET, secure_filename(folder))
    if not os.path.isdir(full_path):
        return jsonify({'success': False, 'error': f'no existe {full_path}'}), 404
    try:
        # Decodificar dataURL "data:image/png;base64,..."
        if ',' in png_url:
            _, b64 = png_url.split(',', 1)
        else:
            b64 = png_url
        png_bytes = base64.b64decode(b64)
        with open(os.path.join(full_path, 'feet_map.png'), 'wb') as f:
            f.write(png_bytes)
        import json as _json
        with open(os.path.join(full_path, 'feet_map.json'), 'w') as jf:
            _json.dump(payload, jf, indent=2)
        return jsonify({'success': True, 'folder': folder,
                        'png_path': os.path.join(folder, 'feet_map.png'),
                        'json_path': os.path.join(folder, 'feet_map.json')})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500




# =====

                            # Picos del borde inferior = pezuñas (2-4 según cruce de patas).
                            try:
                                from scipy.signal import find_peaks
                                h_mask, w_mask = bin_sil.shape
                                idx_grid = np.arange(h_mask).reshape(-1, 1)
                                masked_y = np.where(bin_sil > 0, idx_grid, -1)
                                bottom_y_per_col = masked_y.max(axis=0)  # -1 si col vacía
                                cols_ok = np.where(bottom_y_per_col >= 0)[0]
                                if cols_ok.size >= 10:
                                    cmin, cmax = int(cols_ok[0]), int(cols_ok[-1])
                                    sig = bottom_y_per_col[cmin:cmax + 1].astype(np.float32)
                                    # Forward-fill huecos
                                    last_v = sig[0] if sig[0] >= 0 else 0
                                    for i in range(sig.size):
                                        if sig[i] < 0:
                                            sig[i] = last_v
                                        else:
                                            last_v = sig[i]
                                    sil_h = max(1.0, float(sig.max() - sig.min()))
                                    min_dist = max(8, int(0.06 * sig.size))
                                    min_prom = max(3.0, 0.04 * sil_h)
                                    peaks, _props = find_peaks(sig, distance=min_dist, prominence=min_prom)
                                    print(f"[detect_cow_fast] feet: sig_len={sig.size} sil_h={sil_h:.0f} "
                                          f"min_dist={min_dist} min_prom={min_prom:.1f} → peaks={len(peaks)}")
                                    for pk in peaks:
                                        x_in_crop = int(cmin + pk)
                                        y_in_crop = int(sig[pk])
                                        x_resized = sx1 + x_in_crop
                                        y_resized = sy1 + y_in_crop
                                        if scale_factor and scale_factor > 0:
                                            xo = (x_resized - pad_x) / scale_factor
                                            yo = (y_resized - pad_y) / scale_factor
                                        else:
                                            xo, yo = float(x_resized), float(y_resized)
                                        feet_points.append({
                                            'x': round(float(xo), 1),
                                            'y': round(float(yo), 1),
                                        })
                            except Exception as _ef:
                                print(f"[detect_cow_fast] feet detection fail: {_ef}")
