# Extraído verbatim de app.py — ruta /batch_screen (REMOVIDO 2026-07-06)
# Dependencias que solo usaba esta ruta (también removidas de app.py):
#   - import statistics
#   - get_weight_range (del import de core.breed_coefficients)

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
                        calc_cm_per_px = VARA_CM / avg_post_height_px
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
