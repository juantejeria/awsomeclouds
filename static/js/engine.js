// Vara de referencia (cm) — definida en config.ini [calibracion], inyectada por base.html
var VARA_CM = (typeof window !== 'undefined' && window.VARA_CM) ? window.VARA_CM : 110;
var K_DEPTH_CFG = (typeof window !== 'undefined' && window.K_DEPTH) ? window.K_DEPTH : 0.25;
// ====================================================================
// engine.js - Manual Frame Selection for Cattle Weight Estimation
// ====================================================================
// Flow:
//   1. Upload video -> HTML5 video player with frame-by-frame navigation
//   2. Navigate to any frame -> click "Analyze" -> backend analyzes that frame
//   3. Select ONE frame as calibration (2 posts + cow -> cow HEIGHT)
//   4. Select MULTIPLE frames as keypoint frames (cow + keypoints -> BL, Girth)
//   5. Click "Calculate Weight" -> JS uses fixed cow height + keypoint distances
// ====================================================================

// ── Breed coefficients (from breed_coefficients.py) ──

var BREED_K = {
    angus: 1.00, hereford: 0.98, shorthorn: 0.96, charolais: 1.04,
    limousin: 0.98, simmental: 1.01, brahman: 0.93, nelore: 0.91,
    gyr: 0.88, brangus: 0.96, bradford: 0.95, braford: 0.95,
    holando: 0.93, holstein: 0.93, generico: 1.00, desconocido: 1.00
};

var CATEGORY_K = {
    ternero: 0.84, ternera: 0.84, recria: 0.90, vaquillona: 0.95,
    novillito: 0.97, novillo: 1.00, vaca: 0.95, toro: 1.08,
    desconocido: 1.00
};

var AGE_K = {
    '0-6': 0.85, '6-12': 0.92, '12-18': 0.96, '18-24': 0.98,
    '24-36': 1.00, '36+': 1.00, 'desconocido': 1.00
};

// ── State Management ──

var AppState = {
    videoFile: null,
    videoUrl: null,
    fps: 30,  // Default, estimated from video metadata
    breed: 'desconocido',
    category: 'desconocido',
    age_range: 'desconocido',
    calibrationFrame: null,    // { frameNum, data (from /analyze_frame response) }
    keypointFrames: [],        // [{ frameNum, data }, ...]
    currentAnalysis: null,     // last /analyze_frame response
    currentFrameNum: 0,
    analyzing: false,
    // Two-phase scan/analyze
    scanResult: null,          // last /scan_frame response
    selectedCowIndex: 0,       // which cow is selected
    selectedPostIndices: null,  // which post indices are selected (null = all)
    frameImageId: null,         // cached frame UUID from scan
    // Referencia fija del rectángulo (escala) para todo el video
    videoId: null,
    lockedReference: null,      // { post1: {cx,top_tape,floor,tape_px}, post2: {...} }
    // Análisis en vivo
    liveAnalyzing: false,
    liveLastResult: null,       // {animal_bbox_original, weight_kg, cow_height_cm, video_w, video_h}
    liveSampleInFlight: false,
    // Detector de pasadas
    passingDetecting: false,
    passingAbort: false,
    passingResults: [],         // [{frameNum, weight_kg, cow_height_cm, annotated_image, p}]
    passingStats: { analyzed: 0, detected: 0, in_rect: 0, out_rect: 0, no_cow: 0 }
};

function generateVideoId() {
    // UUID v4 simplificado
    return 'vid-' + Date.now().toString(36) + '-' + Math.random().toString(36).substr(2, 9);
}

function updateReferenceBadge() {
    var $badge = $('#referenceBadge');
    if (!$badge.length) return;
    if (AppState.lockedReference) {
        $badge.show().removeClass('badge-warning').addClass('badge-success')
            .html('<i class="fas fa-thumbtack"></i> Referencia fijada');
    } else {
        $badge.hide();
    }
    drawReferenceOverlay();
}

function getVideoContentRect(video) {
    // Calcula el rectángulo REAL del contenido dentro del elemento <video>,
    // considerando el letterbox interno si el aspect ratio del elemento no
    // coincide con el nativo del video.
    var elemW = video.clientWidth;
    var elemH = video.clientHeight;
    var videoW = video.videoWidth;
    var videoH = video.videoHeight;
    if (!elemW || !elemH || !videoW || !videoH) {
        return { left: 0, top: 0, width: elemW, height: elemH };
    }
    var elemAspect = elemW / elemH;
    var videoAspect = videoW / videoH;
    if (videoAspect > elemAspect) {
        // Video más ancho que el hueco → bandas negras arriba/abajo
        var contentH = elemW / videoAspect;
        return { left: 0, top: (elemH - contentH) / 2, width: elemW, height: contentH };
    } else {
        // Video más alto → bandas negras a los lados
        var contentW = elemH * videoAspect;
        return { left: (elemW - contentW) / 2, top: 0, width: contentW, height: elemH };
    }
}

function drawReferenceOverlay() {
    var video = document.getElementById('videoPlayer');
    var canvas = document.getElementById('videoOverlay');
    if (!video || !canvas) return;
    var ctx = canvas.getContext('2d');

    // Canvas ocupa todo el <video>, pero solo dibujamos dentro del contentRect
    var elemW = video.clientWidth;
    var elemH = video.clientHeight;
    if (elemW === 0 || elemH === 0) return;

    canvas.width = elemW;
    canvas.height = elemH;
    canvas.style.width = elemW + 'px';
    canvas.style.height = elemH + 'px';

    // Alinear canvas con el video (puede estar centrado con margin:0 auto)
    var videoRect = video.getBoundingClientRect();
    var parentRect = video.parentElement.getBoundingClientRect();
    canvas.style.top = (videoRect.top - parentRect.top) + 'px';
    canvas.style.left = (videoRect.left - parentRect.left) + 'px';

    ctx.clearRect(0, 0, elemW, elemH);

    if (!AppState.lockedReference || !AppState.lockedReference.original_coords) return;
    var oc = AppState.lockedReference.original_coords;
    if (!oc.video_w || !oc.video_h) return;

    // Rect del contenido real del video (excluye letterbox interno)
    var cr = getVideoContentRect(video);

    // Factor de escala: pixeles nativos del video → pixeles de display del contenido
    var sx = cr.width / oc.video_w;
    var sy = cr.height / oc.video_h;

    var p1 = oc.post1;
    var p2 = oc.post2;
    var pL = p1.cx < p2.cx ? p1 : p2;
    var pR = p1.cx < p2.cx ? p2 : p1;

    // Aplicar escala + offset del contentRect (letterbox)
    var L = {
        cx: cr.left + pL.cx * sx,
        top: cr.top + pL.top_tape * sy,
        floor: cr.top + pL.floor * sy,
        tape_px: pL.tape_px * sy,
    };
    var R = {
        cx: cr.left + pR.cx * sx,
        top: cr.top + pR.top_tape * sy,
        floor: cr.top + pR.floor * sy,
        tape_px: pR.tape_px * sy,
    };

    // Rectángulo amarillo
    ctx.strokeStyle = 'rgba(255, 235, 59, 0.95)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(L.cx, L.top); ctx.lineTo(R.cx, R.top);
    ctx.moveTo(L.cx, L.floor); ctx.lineTo(R.cx, R.floor);
    ctx.moveTo(L.cx, L.top); ctx.lineTo(L.cx, L.floor);
    ctx.moveTo(R.cx, R.top); ctx.lineTo(R.cx, R.floor);
    ctx.stroke();

    // Cintas rojas (VARA_CM) en cada poste
    ctx.strokeStyle = 'rgba(244, 67, 54, 0.95)';
    ctx.lineWidth = 3;
    [L, R].forEach(function(p) {
        ctx.beginPath();
        ctx.moveTo(p.cx, p.top);
        ctx.lineTo(p.cx, p.top + p.tape_px);
        ctx.stroke();
    });

    // Etiquetas cm (altura lateral)
    ctx.fillStyle = 'rgba(255, 235, 59, 1)';
    ctx.strokeStyle = 'rgba(0,0,0,0.8)';
    ctx.lineWidth = 3;
    ctx.font = 'bold 14px sans-serif';
    [L, R].forEach(function(p, idx) {
        var cm = Math.round((p.floor - p.top) / (p.tape_px / VARA_CM));
        var txt = cm + 'cm';
        var xTxt = idx === 0 ? p.cx + 8 : p.cx - 55;
        var yTxt = (p.top + p.floor) / 2;
        ctx.strokeText(txt, xTxt, yTxt);
        ctx.fillText(txt, xTxt, yTxt);
    });

    // Bbox de la vaca (live-analyze)
    if (AppState.liveLastResult && AppState.liveLastResult.animal_bbox_original) {
        var bb = AppState.liveLastResult.animal_bbox_original;
        var vW = AppState.liveLastResult.video_w || oc.video_w;
        var vH = AppState.liveLastResult.video_h || oc.video_h;
        var cx2 = cr.width / vW;
        var cy2 = cr.height / vH;
        var x1 = cr.left + bb[0] * cx2;
        var y1 = cr.top + bb[1] * cy2;
        var x2 = cr.left + bb[2] * cx2;
        var y2 = cr.top + bb[3] * cy2;

        // Bbox verde grueso
        ctx.strokeStyle = 'rgba(76, 175, 80, 0.95)';
        ctx.lineWidth = 4;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Punto de escala = centro-bottom del bbox (pie de la vaca en el piso)
        var footX = (x1 + x2) / 2;
        var footY = y2;
        ctx.fillStyle = 'rgba(255, 193, 7, 1)';
        ctx.strokeStyle = 'rgba(0,0,0,0.9)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(footX, footY, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Calcular "Xcm" al nivel del pie: altura del bbox en cm interpolando escala del rectángulo
        // entre los postes según la posición X del pie (cow_cx en coords originales)
        var cow_cx_orig = (bb[0] + bb[2]) / 2;
        var posLR = (cow_cx_orig - pL.cx / sx) / (pR.cx / sx - pL.cx / sx);
        // scale por poste en espacio original
        var s1_orig = VARA_CM / (pL.tape_px / sy);
        var s2_orig = VARA_CM / (pR.tape_px / sy);
        var scale_at_cow = (1 - posLR) * s1_orig + posLR * s2_orig;  // cm/px original
        var bbox_h_orig = bb[3] - bb[1];
        var cow_cm = bbox_h_orig * scale_at_cow;

        // Label "Xcm" justo al lado del punto-pie
        var lblMain = cow_cm.toFixed(0) + ' cm';
        ctx.font = 'bold 20px sans-serif';
        var tw = ctx.measureText(lblMain).width;
        var labelX = footX + 14;
        var labelY = footY - 10;
        ctx.fillStyle = 'rgba(0,0,0,0.85)';
        ctx.fillRect(labelX - 4, labelY - 22, tw + 10, 28);
        ctx.fillStyle = 'rgba(255, 193, 7, 1)';
        ctx.fillText(lblMain, labelX, labelY);

        // Label secundario arriba del bbox: peso si existe
        if (AppState.liveLastResult.weight_kg) {
            var w_lbl = AppState.liveLastResult.weight_kg.toFixed(0) + ' kg';
            ctx.font = 'bold 14px sans-serif';
            var w_tw = ctx.measureText(w_lbl).width;
            ctx.fillStyle = 'rgba(0,0,0,0.75)';
            ctx.fillRect(x1, y1 - 22, w_tw + 12, 20);
            ctx.fillStyle = 'rgba(76, 175, 80, 1)';
            ctx.fillText(w_lbl, x1 + 6, y1 - 7);
        }
    }
}

function lockCurrentReference(rectRef) {
    if (!rectRef || !rectRef.post1 || !rectRef.post2) {
        alert('No hay rectángulo para fijar. Ejecutá un análisis válido primero.');
        return;
    }
    if (!AppState.videoId) {
        AppState.videoId = generateVideoId();
    }
    $.ajax({
        type: 'POST',
        url: '/lock_reference',
        contentType: 'application/json',
        data: JSON.stringify({
            video_id: AppState.videoId,
            post1: rectRef.post1,
            post2: rectRef.post2,
        }),
        success: function(resp) {
            if (resp.success) {
                // Mergear original_coords (viene del /calibrate_frame response original, no del backend /lock_reference)
                AppState.lockedReference = Object.assign({}, resp.reference, {
                    original_coords: rectRef.original_coords
                });
                updateReferenceBadge();
                // Feedback visual en el panel
                $('#btnLockReference').replaceWith(
                    '<span class="badge bg-success"><i class="fas fa-check"></i> Fijada</span>'
                );
            } else {
                alert('Error fijando referencia: ' + (resp.error || 'desconocido'));
            }
        },
        error: function(xhr) {
            alert('Error: ' + (xhr.responseJSON && xhr.responseJSON.error || xhr.statusText));
        },
    });
}

// ── Detector de pasadas (recorre el video entero) ──

function toggleDetectPassings() {
    if (AppState.passingDetecting) {
        AppState.passingAbort = true;
    } else {
        startDetectPassings();
    }
}

function startDetectPassings() {
    if (!AppState.lockedReference) {
        alert('Primero fijá una referencia (calibrá 2 postes y click "Fijar referencia").');
        return;
    }
    var video = document.getElementById('videoPlayer');
    if (!video || !video.duration) {
        alert('No hay video cargado.');
        return;
    }
    AppState.passingDetecting = true;
    AppState.passingAbort = false;
    AppState.passingResults = [];
    AppState.passingStats = { analyzed: 0, detected: 0, in_rect: 0, out_rect: 0, no_cow: 0 };

    $('#btnDetectPassingsLabel').text('Cancelar');
    $('#passingStatusPanel').show().text('Procesando…');
    $('#screeningCard').fadeIn(200);
    $('#screeningGallery').empty().css('display', 'flex');
    $('#screeningProgress').show();
    $('#screeningProgressBar').css('width', '0%');
    $('#screeningProgressText').text('Detectando pasadas…');
    $('#screeningSummary').hide().empty();

    // Pausar video y desactivar live
    if (!video.paused) video.pause();
    if (AppState.liveAnalyzing) stopLiveAnalyze();

    var totalFrames = getTotalFrames();
    // Default: 10 samples por segundo de video (interval = fps/10, mín 1)
    var fps = AppState.fps || 30;
    var defaultInterval = Math.max(1, Math.round(fps / 10));
    var samplesPerSec = Math.round(fps / defaultInterval);
    var inputStr = prompt('¿Cada cuántos frames analizar?\n' +
        'Default = ' + defaultInterval + ' frames (' + samplesPerSec + ' muestras por segundo, video a ' + fps + ' fps).\n' +
        'Ej. ' + Math.max(1, Math.round(fps/20)) + ' → 20/s (más denso)   ' +
        Math.round(fps) + ' → 1/s (más rápido)', defaultInterval);
    if (inputStr === null) {
        // Cancelado
        stopDetectPassings();
        return;
    }
    var interval = parseInt(inputStr) || defaultInterval;
    if (interval < 1) interval = 1;

    var startFrame = getCurrentFrameNum();
    if (startFrame < 0 || startFrame >= totalFrames) startFrame = 0;

    // Corre hasta el fin del video; el usuario detiene con el mismo botón (toggleDetectPassings).
    var endFrame = totalFrames;

    $('#screeningProgressText').text('Procesando desde frame ' + startFrame + ' (Cancelar para detener)…');
    processPassingLoop(startFrame, endFrame, interval);
}

function _sleep(ms) { return new Promise(function(r) { setTimeout(r, ms); }); }

function _seekVideo(video, t) {
    return new Promise(function(resolve) {
        var done = false;
        var on = function() {
            if (done) return;
            done = true;
            video.removeEventListener('seeked', on);
            resolve();
        };
        video.addEventListener('seeked', on);
        video.currentTime = t;
        setTimeout(function() { on(); }, 2000);
    });
}

async function processPassingLoop(startFrame, endFrame, interval) {
    var video = document.getElementById('videoPlayer');
    var totalFramesVideo = getTotalFrames();
    var MAX_CONCURRENT = 2;
    var active = 0;
    var next = startFrame;
    var pending = [];

    // Sin filtro por posición: mostramos TODAS las detecciones, el usuario decide
    var P_MIN = -999;
    var P_MAX = 999;

    async function fireRequest(frameNum, blob) {
        active++;
        AppState.passingStats.analyzed++;
        try {
            var fd = new FormData();
            fd.append('frame', blob, 'passing_' + frameNum + '.jpg');
            if (AppState.videoId) fd.append('video_id', AppState.videoId);
            if (AppState.lockedReference) {
                fd.append('locked_reference_json', JSON.stringify({
                    post1: AppState.lockedReference.post1,
                    post2: AppState.lockedReference.post2,
                    original_coords: AppState.lockedReference.original_coords,
                }));
            }
            var resp = await fetch('/detect_cow_fast', { method: 'POST', body: fd });
            var data = await resp.json();
            if (data && data.success && data.detected) {
                AppState.passingStats.detected++;
                var isValid = !!data.within_rectangle;
                if (isValid) {
                    AppState.passingStats.in_rect++;
                } else {
                    AppState.passingStats.out_rect++;
                    if (data.cruce_reason) {
                        console.info('[PASSING] frame', frameNum, 'sin cruce:', data.cruce_reason);
                    }
                }
                // Aceptamos TODO frame con vaca detectada. La selección de
                // frames de barril se hace después por proximidad al cruce
                // (counter ±10 alrededor del frame 0), no por presencia de
                // contorno limpio — la malla se repara automáticamente.
                var thumbUrl = await renderPassingThumbnail(blob, data, isValid);
                AppState.passingResults.push({
                    frameNum: frameNum,
                    valid: isValid,
                    within_rectangle: isValid,
                    cow_height_cm: data.cow_height_cm,
                    cm_per_px: data.cm_per_px,
                    animal_bbox_original: data.animal_bbox_original,
                    video_w: data.video_w,
                    video_h: data.video_h,
                    p: data.p,
                    t_floor: data.t_floor,
                    x_cross: data.x_cross,
                    y_cross: data.y_cross,
                    silueta_bottom_used: !!data.silueta_bottom_used,
                    barril_top_used: !!data.barril_top_used,
                    barril_post_overlap: !!data.barril_post_overlap,
                    barril_volumen_litros: data.barril_volumen_litros,
                    barril_contour_norm: data.barril_contour_norm,  // para consenso
                    barril_cols_rellenadas: data.barril_cols_rellenadas,
                    bbox_aligned_with_floor: !!data.bbox_aligned_with_floor,
                    y_diff_floor: data.y_diff_floor,
                    annotated_image: thumbUrl,
                });
                appendPassingThumbnail(AppState.passingResults.length - 1);

                // El loop corre hasta endFrame o hasta que el usuario presione
                // "Cancelar" (toggleDetectPassings → passingAbort = true).
            } else {
                AppState.passingStats.no_cow++;
            }
        } catch (err) {
            console.error('[PASSING] err', err);
        } finally {
            active--;
        }
    }

    async function renderPassingThumbnail(blob, data, isValid) {
        if (isValid === undefined) isValid = true;
        var THUMB_W = 520;
        var img = await createImageBitmap(blob);
        var scale = THUMB_W / img.width;
        var W = Math.round(img.width * scale);
        var H = Math.round(img.height * scale);
        var canvas = document.createElement('canvas');
        canvas.width = W;
        canvas.height = H;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, W, H);

        // Rectángulo de referencia
        var oc = AppState.lockedReference && AppState.lockedReference.original_coords;
        if (oc) {
            var pL = oc.post1.cx < oc.post2.cx ? oc.post1 : oc.post2;
            var pR = oc.post1.cx < oc.post2.cx ? oc.post2 : oc.post1;
            // Top + laterales en amarillo
            ctx.strokeStyle = 'rgba(255,235,59,0.95)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(pL.cx*scale, pL.top_tape*scale); ctx.lineTo(pR.cx*scale, pR.top_tape*scale);
            ctx.moveTo(pL.cx*scale, pL.top_tape*scale); ctx.lineTo(pL.cx*scale, pL.floor*scale);
            ctx.moveTo(pR.cx*scale, pR.top_tape*scale); ctx.lineTo(pR.cx*scale, pR.floor*scale);
            ctx.stroke();

            // LINEA DEL PISO DEL RECTANGULO = AZUL
            ctx.strokeStyle = 'rgba(33, 150, 243, 1)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(pL.cx*scale, pL.floor*scale);
            ctx.lineTo(pR.cx*scale, pR.floor*scale);
            ctx.stroke();

            // Cintas rojas — soporta postes inclinados via top_tape_x
            ctx.strokeStyle = 'rgba(244,67,54,0.95)';
            ctx.lineWidth = 1;
            [pL, pR].forEach(function(p) {
                var tx = (p.top_tape_x !== undefined ? p.top_tape_x : p.cx) * scale;
                var ty = p.top_tape * scale;
                var bx = p.cx * scale;
                var by = p.floor * scale;
                ctx.beginPath();
                ctx.moveTo(tx, ty);
                ctx.lineTo(bx, by);
                ctx.stroke();
            });

            // Rotated rect en celeste si hay tilt > 0.5°
            [pL, pR].forEach(function(p) {
                if (!Array.isArray(p.rot_corners) || p.rot_corners.length !== 4) return;
                if (Math.abs(p.angle_deg || 0) < 0.5) return;
                ctx.strokeStyle = 'rgba(100,220,255,0.85)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                p.rot_corners.forEach(function(c, i) {
                    var x = c[0] * scale, y = c[1] * scale;
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                ctx.closePath();
                ctx.stroke();
            });
        }

        // Bbox de la vaca — lados en verde/naranja según validez, pero la LINEA INFERIOR azul
        var bb = data.animal_bbox_original;
        var bx = bb[0]*scale, by = bb[1]*scale;
        var bw = (bb[2]-bb[0])*scale, bh = (bb[3]-bb[1])*scale;

        // 3 lados (top + laterales) en verde/naranja
        ctx.strokeStyle = isValid ? 'rgba(76,175,80,0.95)' : 'rgba(255,152,0,0.95)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(bx, by + bh); ctx.lineTo(bx, by);             // lado izq
        ctx.lineTo(bx + bw, by);                                  // lado top
        ctx.lineTo(bx + bw, by + bh);                             // lado der
        ctx.stroke();

        // LINEA INFERIOR DEL BBOX DE LA VACA = AZUL (solo del ancho del bbox)
        ctx.strokeStyle = 'rgba(33, 150, 243, 1)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(bx, by + bh);
        ctx.lineTo(bx + bw, by + bh);
        ctx.stroke();

        if (isValid && data.x_cross !== undefined && data.y_cross !== undefined) {
            // PUNTO DE CRUCE VÁLIDO
            var crossX = data.x_cross * scale;
            var crossY = data.y_cross * scale;

            // Proyección sobre la línea del suelo: marcador + distancias en cm a cada poste
            if (oc) {
                var floorYavg = ((pL.floor + pR.floor) / 2) * scale;
                var avgTapePx = (pL.tape_px + pR.tape_px) / 2;
                var cmPerPxPosts = avgTapePx > 0 ? (VARA_CM / avgTapePx) : 0;
                var distLcm = (data.x_cross - pL.cx) * cmPerPxPosts;
                var distRcm = (pR.cx - data.x_cross) * cmPerPxPosts;

                // Línea vertical punteada del cruce al suelo
                ctx.strokeStyle = 'rgba(255,193,7,0.85)';
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                ctx.moveTo(crossX, crossY);
                ctx.lineTo(crossX, floorYavg);
                ctx.stroke();
                ctx.setLineDash([]);

                // Marcador en el suelo
                ctx.fillStyle = 'rgba(255,193,7,1)';
                ctx.strokeStyle = 'rgba(0,0,0,0.9)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(crossX, floorYavg, 2.5, 0, Math.PI*2);
                ctx.fill();
                ctx.stroke();

                // Etiqueta debajo del suelo: "←Lcm | Rcm→"
                var distLbl = '←' + Math.round(distLcm) + ' | ' + Math.round(distRcm) + '→ cm';
                ctx.font = 'bold 12px sans-serif';
                var dtw = ctx.measureText(distLbl).width;
                var dLabelX = crossX - dtw / 2;
                var dLabelY = floorYavg + 16;
                if (dLabelX < 2) dLabelX = 2;
                if (dLabelX + dtw + 6 > W) dLabelX = W - dtw - 6;
                if (dLabelY + 4 > H) dLabelY = floorYavg - 6;
                ctx.fillStyle = 'rgba(0,0,0,0.85)';
                ctx.fillRect(dLabelX - 3, dLabelY - 11, dtw + 6, 14);
                ctx.fillStyle = 'rgba(255,193,7,1)';
                ctx.fillText(distLbl, dLabelX, dLabelY);
            }

            ctx.fillStyle = 'rgba(255,193,7,1)';
            ctx.strokeStyle = 'rgba(0,0,0,0.9)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(crossX, crossY, 3, 0, Math.PI*2);
            ctx.fill();
            ctx.stroke();
        } else {
            // INVALID: mostrar banner superior con razón
            var reason = data.reason || 'cruce fuera del rectángulo';
            ctx.font = 'bold 12px sans-serif';
            var msg = '⚠ ' + reason;
            if (msg.length > 60) msg = msg.substring(0, 57) + '...';
            var mw = ctx.measureText(msg).width;
            ctx.fillStyle = 'rgba(255,152,0,0.92)';
            ctx.fillRect(0, 0, Math.min(W, mw + 16), 22);
            ctx.fillStyle = 'rgba(0,0,0,1)';
            ctx.fillText(msg, 8, 15);
        }

        return canvas.toDataURL('image/jpeg', 0.82);
    }

    var rangeSize = Math.max(1, endFrame - startFrame);
    while (next < endFrame && !AppState.passingAbort) {
        // Esperar si hay demasiadas requests en vuelo
        while (active >= MAX_CONCURRENT && !AppState.passingAbort) {
            await _sleep(80);
        }
        if (AppState.passingAbort) break;

        var frameNum = next;
        next += interval;
        var targetTime = frameNum / AppState.fps;
        if (targetTime > video.duration) targetTime = video.duration;

        // Seek + captura sequencial (el video sólo puede estar en un tiempo a la vez)
        await _seekVideo(video, targetTime);
        var blob = await captureVideoFrameBlob(video);

        // Progress UI (% dentro del rango solicitado)
        var pct = Math.floor(100 * (frameNum - startFrame) / rangeSize);
        $('#screeningProgressBar').css('width', pct + '%');
        $('#screeningProgressText').text('Frame ' + frameNum + ' / ' + endFrame +
            ' (' + AppState.passingResults.length + ' pasadas)');
        $('#passingStatusPanel').text('Frame ' + frameNum + ' · ' + AppState.passingResults.length + ' pasadas');

        // Fire async, no await
        fireRequest(frameNum, blob);
    }

    // Esperar que terminen las requests en vuelo
    while (active > 0) await _sleep(100);

    stopDetectPassings();
    finalizePassingResults();
}

function stopDetectPassings() {
    AppState.passingDetecting = false;
    AppState.passingAbort = false;
    $('#btnDetectPassingsLabel').text('Detectar pasadas');
    $('#passingStatusPanel').hide();
    $('#screeningProgress').hide();
}

function processPassingFrame() {
    // LEGACY, reemplazada por processPassingLoop (async/await + paralelismo).
    return;
    // unreachable
    var video = null;
    var targetTime = 0;
    var onSeeked = function() {};
    video.addEventListener('seeked', onSeeked);
    video.currentTime = targetTime;
    // Fallback si no dispara 'seeked'
    setTimeout(function() {
        if (!seeked) {
            video.removeEventListener('seeked', onSeeked);
            onSeeked();
        }
    }, 2000);
}

function appendPassingThumbnail(idx) {
    var r = AppState.passingResults[idx];
    // Galería de ALTURA: solo frames con POSTE SOLAPADO (el bbox de la
    // vaca contiene al poste cercano → el cm/px de ese punto es el del
    // poste = 110cm contra el mayor tape_px → escala precisa). El
    // backend devuelve cow_height_cm únicamente en esos frames.
    if (!r.cow_height_cm) return;

    var includedInAvg = (r.counted_in_avg !== undefined) ? r.counted_in_avg : !!r.silueta_bottom_used;
    var borderColor = includedInAvg ? '#4caf50' : '#ff9800';
    var tStr = (r.t_floor !== undefined && r.t_floor !== null) ? (' · t=' + r.t_floor.toFixed(2)) : '';

    var statusTag = '';
    var yDiffStr = (r.y_diff_floor !== undefined && r.y_diff_floor !== null)
        ? ' (Δpiso=' + r.y_diff_floor.toFixed(0) + 'px)' : '';
    if (!includedInAvg) {
        var reason;
        if (r.within_rectangle === false) {
            reason = 'sin cruce válido en el rectángulo';
        } else if (!r.silueta_bottom_used && !r.bbox_aligned_with_floor) {
            reason = 'sin silueta, bbox desalineado del piso' + yDiffStr;
        } else {
            reason = 'no contado';
        }
        statusTag += '<span style="color:#e65100; font-weight:600;"> · descartado del promedio (' + reason + ')</span>';
    } else if (!r.silueta_bottom_used) {
        statusTag += '<span style="color:#1976d2; font-weight:600;"> · rescatado (bbox alineado al piso' + yDiffStr + ')</span>';
    }
    var rellenoStr = '';
    var html = '<div class="col-md-6 col-12 mb-2" id="passing-card-' + idx + '">' +
        '<div style="padding:4px; cursor:pointer; border:2px solid ' + borderColor + '; border-radius:6px; background:#fff; overflow:hidden;" onclick="showPassingDetail(' + idx + ')">' +
        (r.annotated_image ? '<img src="' + r.annotated_image + '" style="display:block; width:100%; height:auto; border-radius:4px;">' : '') +
        '<div style="font-size:0.85em; padding:4px 6px; line-height:1.3;">' +
        '<strong>Frame ' + r.frameNum + '</strong>' + tStr + ' · ' +
        'Altura: <strong>' + r.cow_height_cm.toFixed(1) + ' cm</strong>' +
        rellenoStr +
        statusTag +
        '</div></div></div>';
    $('#screeningGallery').append(html);
}

window.showPassingDetail = function(idx) {
    var r = AppState.passingResults[idx];
    var video = document.getElementById('videoPlayer');
    if (video && r.frameNum != null) {
        video.currentTime = r.frameNum / AppState.fps;
        $('html, body').animate({ scrollTop: $('#videoCard').offset().top - 20 }, 300);
    }
};

async function downloadResultCard() {
    var cowName = ($('#resultCowName').val() || '').trim();
    if (!cowName) {
        alert('Poné un nombre para la vaca.');
        return;
    }
    if (!AppState.passingResults.length) {
        alert('No hay resultados para exportar.');
        return;
    }

    // pasadas = frames de altura (poste solapado, contados al promedio).
    // barriles = frames del barril (sin poste solapando).
    // Galerías disjuntas: ningún frame aparece en ambas listas.
    var pasadas = AppState.passingResults.filter(function(r) {
        return r.counted_in_avg && r.annotated_image;
    });
    var barriles = AppState.passingResults.filter(function(r) {
        return !r.cow_height_cm && !r.barril_outlier && r.barril_contour_norm
            && r.barril_contour_norm.tops_cm && r.barril_contour_norm.bottoms_cm
            && r.annotated_image;
    });
    if (!pasadas.length) {
        alert('No hay pasadas válidas para exportar.');
        return;
    }

    var $btn = $('#btnDownloadResult').prop('disabled', true)
        .html('<i class="fas fa-spinner fa-spin"></i> Generando…');

    try {
        var alturaStr = (AppState.lastAvgH || 0).toFixed(1) + ' cm';
        var barrilStr = (AppState.lastConsensusB != null)
            ? AppState.lastConsensusB.toFixed(1) + ' L'
            : 'N/A';

        function loadImg(src) {
            return new Promise(function(resolve) {
                if (!src) { resolve(null); return; }
                var img = new Image();
                img.onload = function() { resolve(img); };
                img.onerror = function() { resolve(null); };
                img.src = src;
            });
        }

        var pasadaImgs = await Promise.all(pasadas.map(function(r) {
            return loadImg(r.annotated_image);
        }));
        var barrilUrls = await Promise.all(barriles.map(_buildBarril3DThumbUrl));
        var barrilImgs = await Promise.all(barrilUrls.map(loadImg));

        // Aspect del primer thumbnail válido (todos vienen del mismo video → mismo aspect)
        var refImg = null;
        for (var i = 0; i < pasadaImgs.length; i++) {
            if (pasadaImgs[i]) { refImg = pasadaImgs[i]; break; }
        }
        if (!refImg) {
            for (var k = 0; k < barrilImgs.length; k++) {
                if (barrilImgs[k]) { refImg = barrilImgs[k]; break; }
            }
        }
        var aspect = refImg ? (refImg.height / refImg.width) : 0.56;

        // Layout
        var COLS = 3;
        var THUMB_W = 320;
        var THUMB_H = Math.round(THUMB_W * aspect);
        var GAP = 14;
        var MARGIN = 32;
        var HEADER_H = 150;
        var SECTION_H = 48;
        var FOOTER_H = 52;

        var pasadaRows = Math.ceil(pasadas.length / COLS);
        var barrilRows = barriles.length > 0 ? Math.ceil(barriles.length / COLS) : 0;

        var canvasW = MARGIN + COLS * THUMB_W + (COLS - 1) * GAP + MARGIN;
        var pasadaBlockH = SECTION_H + pasadaRows * THUMB_H + (pasadaRows - 1) * GAP;
        var barrilBlockH = barriles.length > 0
            ? (GAP + SECTION_H + barrilRows * THUMB_H + (barrilRows - 1) * GAP)
            : 0;
        var canvasH = HEADER_H + GAP + pasadaBlockH + barrilBlockH + FOOTER_H;

        var canvas = document.createElement('canvas');
        canvas.width = canvasW;
        canvas.height = canvasH;
        var ctx = canvas.getContext('2d');

        // Fondo
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvasW, canvasH);

        // Header
        ctx.fillStyle = '#334155';
        ctx.fillRect(0, 0, canvasW, HEADER_H);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 34px sans-serif';
        ctx.fillText(cowName.toUpperCase(), MARGIN, 60);
        ctx.font = '20px sans-serif';
        ctx.fillStyle = '#ffebb7';
        ctx.fillText('Altura: ' + alturaStr, MARGIN, 105);
        ctx.fillText('Barril: ' + barrilStr, MARGIN + Math.round(canvasW / 2 - MARGIN), 105);

        function drawGrid(imgs, items, yStart) {
            for (var i = 0; i < items.length; i++) {
                var col = i % COLS;
                var row = Math.floor(i / COLS);
                var x = MARGIN + col * (THUMB_W + GAP);
                var y = yStart + row * (THUMB_H + GAP);
                if (imgs[i]) {
                    ctx.drawImage(imgs[i], x, y, THUMB_W, THUMB_H);
                } else {
                    ctx.fillStyle = '#eeeeee';
                    ctx.fillRect(x, y, THUMB_W, THUMB_H);
                }
                // Banda inferior con solo el frame number (sin L por barril)
                ctx.fillStyle = 'rgba(0,0,0,0.72)';
                ctx.fillRect(x, y + THUMB_H - 24, THUMB_W, 24);
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 13px sans-serif';
                var lbl = 'Frame ' + items[i].frameNum;
                if (items[i].cow_height_cm) {
                    lbl += ' · ' + items[i].cow_height_cm.toFixed(0) + ' cm';
                }
                ctx.fillText(lbl, x + 8, y + THUMB_H - 7);
            }
        }

        // Sección 1: Pasadas detectadas
        var y = HEADER_H + GAP;
        ctx.fillStyle = '#334155';
        ctx.font = 'bold 18px sans-serif';
        ctx.fillText('PASADAS DETECTADAS (' + pasadas.length + ')', MARGIN, y + 26);
        y += SECTION_H;
        drawGrid(pasadaImgs, pasadas, y);
        y += pasadaRows * THUMB_H + (pasadaRows - 1) * GAP;

        // Sección 2: Frames del barril (si hay)
        if (barriles.length > 0) {
            y += GAP;
            ctx.fillStyle = '#334155';
            ctx.font = 'bold 18px sans-serif';
            ctx.fillText('FRAMES DEL BARRIL USADOS EN 3D (' + barriles.length + ')', MARGIN, y + 26);
            y += SECTION_H;
            drawGrid(barrilImgs, barriles, y);
            y += barrilRows * THUMB_H + (barrilRows - 1) * GAP;
        }

        // Footer
        ctx.fillStyle = '#888';
        ctx.font = '13px sans-serif';
        ctx.fillText((AppState.lastN || 0) + ' pasadas válidas',
                     MARGIN, canvasH - 20);

        var url = canvas.toDataURL('image/png');
        var a = document.createElement('a');
        a.href = url;
        a.download = 'resultado_' + cowName + '.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    } catch (err) {
        alert('Error: ' + err);
    } finally {
        $btn.prop('disabled', false).html('<i class="fas fa-download"></i> Descargar resultado');
    }
}

// Calcula el consenso multi-frame del barril: mediana por muestra de
// alturas (60 puntos) + mediana del ancho, y volumen elíptico con K_DEPTH
// 0.25 (mismo cálculo que app.py /detect_cow_fast). Devuelve null si hay
// menos de 2 contornos.
function _percentile(sorted, p) {
    // sorted: array ya ordenado ascendente; p en [0,1]
    if (!sorted.length) return 0;
    if (sorted.length === 1) return sorted[0];
    var idx = p * (sorted.length - 1);
    var lo = Math.floor(idx);
    var hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    var frac = idx - lo;
    return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

function _computeBarrilConsensus(contours, mode) {
    // mode: 'median' | 'p75' | 'maxW_p75' | 'envelope'
    //   median    = mediana en heights y width (comportamiento original)   (A)
    //   p75       = percentil 75 en heights y width                        (B)
    //   maxW_p75  = MAX en width, percentil 75 en heights                  (C)
    //   envelope  = por columna, MIN(tops) + MAX(bottoms) entre frames     (E)
    //               → reconstruye la silueta completa ignorando frames con
    //                 oclusión por poste de referencia. Equivalente a unir
    //                 las máscaras en lugar de promediarlas.
    if (!contours || contours.length < 2) return null;
    mode = mode || 'median';
    var N = contours[0].n_samples;

    function _agg(vals, mode) {
        if (!vals.length) return 0;
        vals.sort(function(a, b) { return a - b; });
        if (mode === 'p75' || mode === 'maxW_p75') {
            return _percentile(vals, 0.75);
        }
        // median
        return vals.length % 2
            ? vals[(vals.length - 1) / 2]
            : (vals[vals.length / 2 - 1] + vals[vals.length / 2]) / 2;
    }

    var heightsAgg = [];
    if (mode === 'envelope') {
        // Para cada columna i: alto_envelope = MAX(top - bot) considerando los
        // tops y bottoms por frame. Necesitamos tanto tops_cm como bottoms_cm.
        // El alto reconstruido es el rango total entre lomo y panza más
        // extendidos vistos en cualquier frame.
        for (var i = 0; i < N; i++) {
            var topsC = contours.map(function(c) { return c.tops_cm ? c.tops_cm[i] : null; })
                                .filter(function(v) { return v != null && v > 0; });
            var botsC = contours.map(function(c) { return c.bottoms_cm ? c.bottoms_cm[i] : null; })
                                .filter(function(v) { return v != null && v >= 0; });
            if (!topsC.length || !botsC.length) {
                // Fallback a mediana de heights si faltan tops/bottoms
                var hs = contours.map(function(c) { return c.heights_cm[i]; })
                                 .filter(function(v) { return v > 0; });
                heightsAgg.push(_agg(hs, 'median'));
                continue;
            }
            var topMax = Math.max.apply(null, topsC);   // lomo más alto entre frames
            var botMin = Math.min.apply(null, botsC);    // panza más baja entre frames
            heightsAgg.push(Math.max(0, topMax - botMin));
        }
    } else {
        for (var i = 0; i < N; i++) {
            var vals = contours.map(function(c) { return c.heights_cm[i]; })
                               .filter(function(v) { return v > 0; });
            heightsAgg.push(_agg(vals, mode));
        }
    }
    var widths = contours.map(function(c) { return c.width_cm; })
                         .filter(function(w) { return w > 0; });
    if (!widths.length) return null;
    var widthAgg;
    if (mode === 'maxW_p75' || mode === 'envelope') {
        widthAgg = Math.max.apply(null, widths);
    } else {
        widthAgg = _agg(widths.slice(), mode);
    }

    var K_DEPTH = K_DEPTH_CFG;
    var dx = widthAgg / (N - 1);
    var volCm3 = 0;
    for (var j = 0; j < N; j++) {
        var h = heightsAgg[j];
        if (h <= 0) continue;
        var a = h / 2.0;
        var b = h * K_DEPTH;
        volCm3 += Math.PI * a * b * dx;
    }
    return {
        volumeL: volCm3 / 1000.0,
        contour: {
            n_samples: N,
            width_cm: widthAgg,
            heights_cm: heightsAgg,
            n_frames: contours.length,
        },
        mode: mode,
    };
}

function finalizePassingResults() {
    // Como los frames se procesan en paralelo, llegan fuera de orden →
    // ordenar por frameNum antes de asignar counters.
    AppState.passingResults.sort(function(a, b) { return a.frameNum - b.frameNum; });

    // Counter: cada frame válido tiene un passing_idx relativo al cruce.
    // El primer frame con cow_height_cm != null es el "frame 0" (cruce con
    // los postes). Frames anteriores: -1, -2, ...; posteriores: +1, +2, ...
    // El barril usa los frames con |passing_idx| <= BARRIL_WINDOW (incluye 0).
    var BARRIL_WINDOW = 10;
    var crossIdx = -1;
    for (var ci = 0; ci < AppState.passingResults.length; ci++) {
        if (AppState.passingResults[ci].cow_height_cm != null) {
            crossIdx = ci;
            break;
        }
    }
    AppState.passingResults.forEach(function(r, i) {
        r.passing_idx = (crossIdx >= 0) ? (i - crossIdx) : null;
        r.barril_eligible = (crossIdx >= 0) &&
            Math.abs(r.passing_idx) <= BARRIL_WINDOW;
    });

    var n = AppState.passingResults.length;
    var s = AppState.passingStats;
    $('#passingStatusPanel').hide();

    var statsHtml = '<div class="alert alert-info py-2 px-3 mb-2" style="font-size:0.88em;">' +
        '<strong>Estadísticas:</strong> ' +
        s.analyzed + ' frames analizados · ' +
        s.detected + ' con vaca detectada · ' +
        s.in_rect + ' dentro del rectángulo · ' +
        s.out_rect + ' fuera del rectángulo · ' +
        s.no_cow + ' sin vaca' +
        '</div>';

    if (n === 0) {
        $('#screeningSummary').show().html(
            statsHtml +
            '<div class="alert alert-warning">No se detectó ninguna pasada dentro del rectángulo.</div>'
        );
        return;
    }

    // Frame entra al promedio si:
    //  · cow_height_cm (= poste solapado, escala precisa del poste cercano), Y
    //  · within_rectangle = True (los pies cruzan el segmento de piso
    //    calibrado entre los postes — sin esto la cm/px del cow es
    //    extrapolada y la altura no es confiable), Y
    //  · (silueta_bottom_used = True (silueta_seg encontró los pies), O
    //     bbox_aligned_with_floor = True (YOLO bottom coincide con piso))
    var rescued = 0;
    AppState.passingResults.forEach(function(r) {
        r.counted_in_avg = false;
        if (!r.cow_height_cm) return;
        if (r.within_rectangle === false) return;  // sin cruce real → no cuenta
        if (r.silueta_bottom_used) {
            r.counted_in_avg = true;
        } else if (r.bbox_aligned_with_floor) {
            r.counted_in_avg = true;
            rescued++;
        }
    });

    // Promedios:
    //  · altura: frames DENTRO del rectángulo (cow_height_cm) que entraron
    //    al promedio (counted_in_avg).
    //  · volumen del barril: frames FUERA del rectángulo (sin altura) con
    //    barril válido — los mismos que alimentan el consenso.
    var sumH_final = 0, nH_final = 0, nExcluded = 0;
    var sumB_final = 0, nB_final = 0;
    AppState.passingResults.forEach(function(r) {
        if (r.cow_height_cm) {
            if (r.counted_in_avg) {
                sumH_final += r.cow_height_cm;
                nH_final++;
            } else {
                nExcluded++;
            }
        }
        if (r.barril_eligible && r.barril_volumen_litros != null && r.barril_volumen_litros > 0) {
            sumB_final += r.barril_volumen_litros;
            nB_final++;
        }
    });
    var avgH = nH_final > 0 ? sumH_final / nH_final : 0;
    var avgB = nB_final > 0 ? sumB_final / nB_final : 0;

    // ── Consenso multi-frame del barril (con descarte por ancho y oclusión) ──
    // Outliers descartados:
    //   · width <= mediana - 30%  → mascara cortada (poste recortó el blob)
    //   · width >= mediana + 30%  → postura estirada (patas abiertas, cuello bajo)
    //   · cols_rellenadas > 15    → oclusión severa, máscara muy reparada
    // Sobre los inliers se computan A (mediana), B (p75), C (maxW+p75).
    var consensusB = null;
    var consensusContour = null;
    var nOutLow = 0, nOutHigh = 0, nOutOclusion = 0;
    var BARRIL_WIDTH_DROP_PCT = 30.0;  // % por debajo / encima de la mediana → outlier
    var BARRIL_COLS_REPAR_MAX = 15;    // > esto → oclusión severa, descartar
    (function() {
        var entries = [];
        AppState.passingResults.forEach(function(r) {
            r.barril_outlier = false;
            r.barril_outlier_reason = null;
            r.barril_width_dev_pct = null;
            if (!r.barril_eligible) return;
            var c = r.barril_contour_norm;
            if (!c || !c.heights_cm || !c.width_cm) return;
            if (c.heights_cm.length !== c.n_samples) return;
            entries.push({ r: r, c: c });
        });
        if (!entries.length) return;

        // Mediana de widths (más robusta que la media para el threshold)
        var widthsSorted = entries.map(function(e) { return e.c.width_cm; })
                                  .sort(function(a, b) { return a - b; });
        var medianW = widthsSorted.length % 2
            ? widthsSorted[(widthsSorted.length - 1) / 2]
            : (widthsSorted[widthsSorted.length / 2 - 1] + widthsSorted[widthsSorted.length / 2]) / 2;
        var lowTh  = medianW * (1 - BARRIL_WIDTH_DROP_PCT / 100);
        var highTh = medianW * (1 + BARRIL_WIDTH_DROP_PCT / 100);

        entries.forEach(function(e) {
            var w = e.c.width_cm;
            var dev = (w - medianW) / medianW * 100;  // signed (+ = sobre la mediana)
            e.r.barril_width_dev_pct = dev;
            var cols = e.r.barril_cols_rellenadas || 0;
            if (w < lowTh) {
                e.r.barril_outlier = true;
                e.r.barril_outlier_reason = 'width_low';
                nOutLow++;
            } else if (w > highTh) {
                e.r.barril_outlier = true;
                e.r.barril_outlier_reason = 'width_high';
                nOutHigh++;
            } else if (cols > BARRIL_COLS_REPAR_MAX) {
                e.r.barril_outlier = true;
                e.r.barril_outlier_reason = 'oclusion';
                nOutOclusion++;
            }
        });

        var inliers = entries.filter(function(e) { return !e.r.barril_outlier; });
        var sourceContours = (inliers.length >= 2 ? inliers : entries).map(function(e) { return e.c; });
        var result = _computeBarrilConsensus(sourceContours, 'median');
        if (result) {
            consensusB = result.volumeL;
            consensusContour = result.contour;
        }
        AppState.consensusVariants = {
            A_median: result ? result.volumeL : null,
            n_frames: sourceContours.length,
            n_total: entries.length,
            outliers_low: nOutLow,
            outliers_high: nOutHigh,
            outliers_oclusion: nOutOclusion,
        };
    })();
    var nBarrilOutliers = nOutLow + nOutHigh + nOutOclusion;

    // Recomputar el promedio simple del barril sin outliers de ancho.
    // En la misma pasada promediamos el LARGO del box del barril
    // (width_cm = extensión horizontal de la máscara del torso, en cm),
    // sobre los mismos frames inliers que el volumen.
    sumB_final = 0;
    nB_final = 0;
    var sumLargo_final = 0, nLargo_final = 0;
    AppState.passingResults.forEach(function(r) {
        if (!r.barril_eligible) return;
        if (r.barril_outlier) return;
        var c = r.barril_contour_norm;
        if (c && c.width_cm > 0) {
            sumLargo_final += c.width_cm;
            nLargo_final++;
        }
        if (r.barril_volumen_litros == null || r.barril_volumen_litros <= 0) return;
        sumB_final += r.barril_volumen_litros;
        nB_final++;
    });
    avgB = nB_final > 0 ? sumB_final / nB_final : 0;
    var avgLargo = nLargo_final > 0 ? sumLargo_final / nLargo_final : 0;

    // PASO 4: re-render gallery (actualizar bordes y labels según counted_in_avg)
    $('#screeningGallery').empty();
    AppState.passingResults.forEach(function(_r, i) { appendPassingThumbnail(i); });

    // PASO 4b: gallery de "frames del barril usados para el 3D".
    // Subconjunto de pasadas con contour válido que alimentan el consenso.

    var excludedMsg = nExcluded > 0
        ? ' · <span style="color:#e65100;"><strong>' + nExcluded + ' descartados</strong> del promedio</span>'
        : '';
    var rescuedMsg = rescued > 0
        ? ' · <span style="color:#1976d2;"><strong>' + rescued + ' rescatados</strong> por alineación bbox-piso</span>'
        : '';
    // Comparación de variantes de consenso del barril:
    //   A = mediana (default — algoritmo original)
    //   B = percentil 75 en heights y width
    //   C = MAX en width + percentil 75 en heights
    // D = "A con backend nuevo" — ya activo. La D efectiva es comparar el A
    // actual contra el A previo (resumen.json guardado). Si A subió respecto
    // al guardado, la ventana de envelope ampliada está recuperando volumen.
    var cv = AppState.consensusVariants || {};
    var barrilMsg = '';
    if (consensusB != null) {
        var fmt = function(v) { return v != null ? v.toFixed(1) + 'L' : '–'; };
        var outDetail = '';
        if (cv.n_total && cv.n_total > cv.n_frames) {
            var parts = [];
            if (cv.outliers_low) parts.push(cv.outliers_low + ' width bajo');
            if (cv.outliers_high) parts.push(cv.outliers_high + ' width alto');
            if (cv.outliers_oclusion) parts.push(cv.outliers_oclusion + ' oclusión');
            outDetail = parts.length ? ' · descartados: ' + parts.join(', ') : '';
        }
        barrilMsg = ' · <strong>Barril:</strong> ' + fmt(cv.A_median) +
            ' <span style="color:#666;">(' + (cv.n_frames || 0) + '/' + (cv.n_total || 0) +
            ' frames' + outDetail + ')</span>';
    }

    // Guardar promedios para el botón de descarga
    AppState.lastAvgH = avgH;
    AppState.lastAvgB = avgB;
    AppState.lastAvgLargo = avgLargo;
    AppState.lastConsensusB = consensusB;
    AppState.lastConsensusContour = consensusContour;
    AppState.lastN = nH_final;

    var downloadSection = (nH_final > 0) ?
        '<div class="mt-3 p-3" style="background:#f5f5f5; border-radius:8px;">' +
        '<label class="config-label" style="display:block; margin-bottom:6px;">Nombre de la vaca</label>' +
        '<div class="d-flex align-items-center flex-wrap" style="gap:10px;">' +
        '<input type="text" id="resultCowName" class="config-select" style="min-width:200px;" placeholder="Ej: vaca1">' +
        '<button class="btn btn-analyze" id="btnDownloadResult"><i class="fas fa-download"></i> Descargar resultado</button>' +
        '</div></div>' : '';

    var largoMsg = (avgLargo > 0)
        ? ' · Largo barril: <strong>' + avgLargo.toFixed(1) + ' cm</strong>' +
          ' <span style="color:#666;">(' + nLargo_final + ' frames)</span>'
        : '';
    var countMsg = '<strong>' + nH_final + ' frames de altura</strong>';
    if (nB_final > 0) countMsg += ' · <strong>' + nB_final + ' frames de barril</strong>';
    var barrilOutlierMsg = nBarrilOutliers > 0
        ? ' · <span style="color:#e65100;"><strong>' + nBarrilOutliers + ' barril(es) descartado(s)</strong> por ancho ≥30% bajo la media</span>'
        : '';
    $('#screeningSummary').show().html(
        statsHtml +
        '<div class="alert alert-success">' +
        '<i class="fas fa-check-circle"></i> ' + countMsg + ' · ' +
        (nH_final ? 'Altura: <strong>' + avgH.toFixed(1) + ' cm</strong>' : 'Sin mediciones válidas') +
        largoMsg +
        barrilMsg +
        rescuedMsg +
        excludedMsg +
        barrilOutlierMsg +
        '</div>' +
        downloadSection
    );

    $('#btnDownloadResult').off('click').on('click', downloadResultCard);
}

// ── Dirección de movimiento (centro X del bbox, primer vs último frame).
// Usada por el consenso 3D para determinar barril_dir. ──
function _detectMovementDir(results) {
    var samples = [];
    for (var i = 0; i < results.length; i++) {
        var bb = results[i].animal_bbox_original;
        if (Array.isArray(bb) && bb.length === 4) {
            samples.push({ idx: i, cx: (bb[0] + bb[2]) / 2 });
        }
    }
    if (samples.length < 2) return { dx: 0, valid: false, samples: samples.length };
    var dx = samples[samples.length - 1].cx - samples[0].cx;
    return { dx: dx, valid: Math.abs(dx) > 5, samples: samples.length };
}

// ── Análisis en vivo ──

function captureVideoFrameBlob(video) {
    return new Promise(function(resolve) {
        var c = document.createElement('canvas');
        c.width = video.videoWidth;
        c.height = video.videoHeight;
        c.getContext('2d').drawImage(video, 0, 0);
        c.toBlob(function(blob) { resolve(blob); }, 'image/jpeg', 0.88);
    });
}

async function showSavedFrames() {
    $('#savedFramesModal').show();
    await refreshSavedFolders();
}

async function refreshSavedFolders() {
    try {
        var resp = await fetch('/list_saved_folders');
        var data = await resp.json();
        var folders = (data && data.folders) || [];
        var $sel = $('#savedFramesFolderSel');
        $sel.empty();
        if (!folders.length) {
            $sel.append('<option value="">— sin carpetas guardadas —</option>');
            $('#savedFramesMeta').text('');
            $('#savedFramesGrid').empty();
            return;
        }
        folders.forEach(function(f) {
            var lbl = f.name + ' (' + f.n_frames + ' frames';
            if (f.central_frame != null) lbl += ', central=' + f.central_frame;
            if (f.has_locked_reference) lbl += ', con ref';
            if (f.mode) lbl += ', ' + f.mode;
            lbl += ')';
            $sel.append('<option value="' + f.name + '">' + lbl + '</option>');
        });
        $sel.off('change').on('change', function() { renderSavedFolderFrames($(this).val()); });
        renderSavedFolderFrames(folders[0].name);
    } catch (e) {
        console.error('[refreshSavedFolders]', e);
        $('#savedFramesMeta').text('Error: ' + e.message);
    }
}

async function renderSavedFolderFrames(folder) {
    if (!folder) { $('#savedFramesGrid').empty(); return; }
    var $meta = $('#savedFramesMeta');
    var $grid = $('#savedFramesGrid');
    $grid.empty();
    $meta.text('Cargando...');
    var fd = new FormData(); fd.append('folder', folder);
    var resp = await fetch('/list_saved_frames', { method: 'POST', body: fd });
    var data = await resp.json();
    if (!data.success) { $meta.text('Error: ' + data.error); return; }
    var frames = data.frames || [];
    var ctx = data.context || {};
    var metaTxt = frames.length + ' frames · ';
    if (ctx.central_frame != null) metaTxt += 'frame central=' + ctx.central_frame + ' · ';
    if (ctx.fps != null) metaTxt += 'fps=' + ctx.fps + ' · ';
    if (ctx.mode) metaTxt += 'modo=' + ctx.mode + ' · ';
    metaTxt += ctx.locked_reference ? 'con locked_reference ✓' : 'SIN locked_reference';
    $meta.text(metaTxt);

    frames.forEach(function(fr) {
        var url = '/saved_frame/' + folder + '/' + fr.file_name;
        var lbl = fr.offset === 0 ? '0 cruce' : (fr.offset > 0 ? '+' + fr.offset : String(fr.offset));
        var bg = fr.offset === 0 ? '#1976d2' : '#37474f';
        $grid.append(
            '<div style="border:1px solid #ddd; border-radius:6px; overflow:hidden; background:#fff;">' +
            '<img src="' + url + '" style="width:100%; height:auto; display:block;">' +
            '<div style="font-size:0.78em; padding:4px 6px; line-height:1.3;">' +
            '<span style="display:inline-block; padding:1px 6px; background:' + bg + '; color:#fff; border-radius:8px; font-weight:700; margin-right:5px;">' + lbl + '</span>' +
            'frame ' + fr.frame_num +
            '</div></div>'
        );
    });
}

async function saveFramesAround() {
    var video = document.getElementById('videoPlayer');
    if (!video || !video.videoWidth) {
        alert('No hay video cargado.');
        return;
    }
    var fps = AppState.fps || 30;
    var centralFrame = Math.round(video.currentTime * fps);

    // Nombre del individuo = nombre de la carpeta (en checkpoints/12 junio).
    var defName = ($('#resultCowName').val() || '').trim() ||
                  (AppState.videoFile ? AppState.videoFile.name.replace(/\.[^.]+$/, '') : '');
    var indName = (window.prompt('Nombre del individuo (será el nombre de la carpeta):', defName) || '').trim();
    if (!indName) { alert('Necesito un nombre para la carpeta del individuo.'); return; }

    var $btn = $('#btnSave21Frames');
    var $lbl = $('#btnSave21FramesLabel');
    var origLbl = $lbl.text();
    $btn.prop('disabled', true);

    var wasPaused = video.paused;
    if (!wasPaused) video.pause();

    try {
        var fd = new FormData();
        fd.append('cow_name', indName);
        fd.append('central_frame', String(centralFrame));
        fd.append('fps', String(fps));
        fd.append('window', '10');
        if (AppState.videoId) fd.append('video_id', AppState.videoId);
        if (AppState.lockedReference) {
            fd.append('locked_reference_json', JSON.stringify({
                post1: AppState.lockedReference.post1,
                post2: AppState.lockedReference.post2,
                original_coords: AppState.lockedReference.original_coords,
            }));
        }

        if (AppState.videoFile) {
            // MODO BACKEND: subimos el video, el backend extrae los 21 frames
            // con cv2 (calidad 95, sin la pérdida del canvas+toBlob).
            $lbl.text('Subiendo video (' + Math.round(AppState.videoFile.size / 1024 / 1024) + ' MB)...');
            fd.append('video', AppState.videoFile, AppState.videoFile.name);
        } else {
            // MODO LEGACY: capturamos los 21 frames del video element
            // (degrada calidad — solo si no tenemos el File original).
            var totalFrames = getTotalFrames();
            var WINDOW = 10;
            var firstFrame = Math.max(0, centralFrame - WINDOW);
            var lastFrame = Math.min(totalFrames - 1, centralFrame + WINDOW);
            var totalToCapture = lastFrame - firstFrame + 1;
            var captured = 0;
            for (var fnum = firstFrame; fnum <= lastFrame; fnum++) {
                $lbl.text('Capturando ' + (captured + 1) + '/' + totalToCapture);
                await _seekVideo(video, fnum / fps);
                var blob = await captureVideoFrameBlob(video);
                var offset = fnum - centralFrame;
                var sign = offset < 0 ? 'm' : (offset > 0 ? 'p' : '0');
                var absOff = Math.abs(offset);
                var fname = 'frame_' + sign + String(absOff).padStart(2, '0') + '_f' + fnum + '.jpg';
                fd.append('frames[]', blob, fname);
                captured++;
            }
        }
        $lbl.text('Procesando...');
        var resp = await fetch('/save_frames_around', { method: 'POST', body: fd });
        var data = await resp.json();
        if (data && data.success) {
            alert('Guardados ' + data.n_frames + ' frames (' + data.mode + ') en:\n' + data.folder);
        } else {
            alert('Error: ' + (data && data.error || resp.statusText));
        }
    } catch (e) {
        console.error('[saveFramesAround] err', e);
        alert('Error capturando frames: ' + e.message);
    } finally {
        $btn.prop('disabled', false);
        $lbl.text(origLbl);
    }
}

function toggleLiveAnalyze() {
    if (AppState.liveAnalyzing) {
        stopLiveAnalyze();
    } else {
        startLiveAnalyze();
    }
}

function startLiveAnalyze() {
    if (!AppState.lockedReference) {
        alert('Primero fijá una referencia (calibrá 2 postes y click "Fijar referencia").');
        return;
    }
    var video = document.getElementById('videoPlayer');
    if (!video || !video.videoWidth) {
        alert('No hay video cargado.');
        return;
    }
    AppState.liveAnalyzing = true;
    $('#btnLiveAnalyze').addClass('btn-analyze').removeClass('btn-control');
    $('#btnLiveAnalyzeLabel').text('Detener');
    $('#liveStatusPanel').show().text('Muestreando…');
    if (video.paused) video.play();
    liveSampleLoop();
}

function stopLiveAnalyze() {
    AppState.liveAnalyzing = false;
    $('#btnLiveAnalyze').removeClass('btn-analyze').addClass('btn-control');
    $('#btnLiveAnalyzeLabel').text('En vivo');
    $('#liveStatusPanel').hide();
    AppState.liveLastResult = null;
    drawReferenceOverlay();  // limpia bbox
}

function liveSampleLoop() {
    if (!AppState.liveAnalyzing) return;
    var video = document.getElementById('videoPlayer');
    if (!video || video.ended) {
        console.log('[LIVE] video ended / missing');
        stopLiveAnalyze();
        return;
    }
    if (AppState.liveSampleInFlight || video.paused) {
        setTimeout(liveSampleLoop, 300);
        return;
    }
    AppState.liveSampleInFlight = true;
    console.log('[LIVE] sampling frame at t=' + video.currentTime.toFixed(2));

    captureVideoFrameBlob(video).then(function(blob) {
        var fd = new FormData();
        fd.append('frame', blob, 'live.jpg');
        fd.append('cow_index', 0);
        fd.append('breed', AppState.breed || 'desconocido');
        fd.append('category', AppState.category || 'desconocido');
        fd.append('age_range', AppState.age_range || 'desconocido');
        if (AppState.videoId) fd.append('video_id', AppState.videoId);
        if (AppState.lockedReference) {
            fd.append('locked_reference_json', JSON.stringify({
                post1: AppState.lockedReference.post1,
                post2: AppState.lockedReference.post2,
            }));
        }
        return fetch('/analyze_frame', { method: 'POST', body: fd });
    }).then(function(resp) {
        return resp.json();
    }).then(function(data) {
        console.log('[LIVE] response', data);
        if (data && data.success) {
            var d = data.details || {};
            AppState.liveLastResult = {
                animal_bbox_original: d.animal_bbox_original,
                video_w: d.video_w,
                video_h: d.video_h,
                weight_kg: data.weight_kg,
                cow_height_cm: d.cow_height_cm,
                cm_per_px: d.cm_per_px,
            };
            var msg;
            if (d.animal_bbox_original) {
                var h_cm_txt = d.cow_height_cm ? d.cow_height_cm.toFixed(0) + ' cm' : '-';
                var w_txt = data.weight_kg ? data.weight_kg.toFixed(1) + ' kg' : '-';
                msg = '🟢 Vaca detectada · ' + h_cm_txt + ' · ' + w_txt;
            } else {
                msg = '⚠️ success=true pero sin bbox';
            }
            $('#liveStatusPanel').text(msg);
            drawReferenceOverlay();
        } else {
            $('#liveStatusPanel').text('Sin vaca detectada: ' + (data && data.error ? data.error : ''));
            AppState.liveLastResult = null;
            drawReferenceOverlay();
        }
    }).catch(function(err) {
        console.error('[LIVE] analyze err', err);
        $('#liveStatusPanel').text('Error: ' + err);
    }).finally(function() {
        AppState.liveSampleInFlight = false;
        if (AppState.liveAnalyzing) {
            setTimeout(liveSampleLoop, 400);
        }
    });
}

function runCalibrateOnly(postIndices, frameNum) {
    $('#selectionPanel').hide();
    $('#analysisLoader').show();
    var formData = new FormData();
    formData.append('frame_image_id', AppState.frameImageId);
    formData.append('post_indices', postIndices.join(','));

    $.ajax({
        type: 'POST',
        url: '/calibrate_frame',
        data: formData,
        contentType: false,
        processData: false,
        timeout: 60000,
        success: function(data) {
            $('#analysisLoader').hide();
            $('#analysisContent').show();
            $('#analysisFrameLabel').text('Frame ' + frameNum + ' - Calibración');
            if (!data.success) {
                $('#annotatedImage').hide();
                $('#analysisDetails').html(
                    '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle"></i> ' +
                    (data.error || 'Error desconocido') + '</div>' +
                    (data.preview_image ? '<img src="' + data.preview_image + '" class="img-fluid mt-2">' : '')
                );
                return;
            }
            $('#annotatedImage').attr('src', data.preview_image).show();
            var rectRef = data.rectangle_ref;
            var html = '<div class="alert alert-success py-2 px-3">' +
                '<i class="fas fa-ruler"></i> <strong>Rectángulo listo.</strong> ' +
                '<button class="btn btn-sm btn-primary ms-2" id="btnLockReference">' +
                '<i class="fas fa-thumbtack"></i> Fijar referencia para todo el video</button></div>';
            $('#analysisDetails').html(html);
            $('#btnLockReference').off('click').on('click', function() {
                lockCurrentReference(rectRef);
            });
        },
        error: function(xhr, status, error) {
            $('#analysisLoader').hide();
            $('#analysisContent').show();
            var msg = (xhr.responseJSON && xhr.responseJSON.error) || error || 'desconocido';
            var preview = (xhr.responseJSON && xhr.responseJSON.preview_image) ? xhr.responseJSON.preview_image : null;
            $('#analysisDetails').html(
                '<div class="alert alert-danger">Error al calibrar: ' + msg + '</div>' +
                (preview ? '<img src="' + preview + '" class="img-fluid mt-2">' : '')
            );
        },
    });
}

function clearCurrentReference() {
    if (!AppState.videoId) return;
    $.ajax({
        type: 'POST',
        url: '/clear_reference/' + AppState.videoId,
        success: function() {
            AppState.lockedReference = null;
            updateReferenceBadge();
            // Re-renderizar último análisis sin ref
            if (AppState.currentAnalysis) {
                displayAnalysis(AppState.currentAnalysis, AppState.currentAnalysis._frameNum);
            }
        },
        error: function() {
            AppState.lockedReference = null;
            updateReferenceBadge();
        },
    });
}

// ── Utility Functions ──

function getCurrentFrameNum() {
    var video = document.getElementById('videoPlayer');
    if (!video || !video.duration) return 0;
    return Math.round(video.currentTime * AppState.fps);
}

function getTotalFrames() {
    var video = document.getElementById('videoPlayer');
    if (!video || !video.duration) return 0;
    return Math.round(video.duration * AppState.fps);
}

function updateFrameCounter() {
    var current = getCurrentFrameNum();
    var total = getTotalFrames();
    AppState.currentFrameNum = current;
    var video = document.getElementById('videoPlayer');
    var curTime = video ? video.currentTime : 0;
    var totTime = video ? (video.duration || 0) : 0;
    var curMin = Math.floor(curTime / 60);
    var curSec = Math.floor(curTime % 60);
    var totMin = Math.floor(totTime / 60);
    var totSec = Math.floor(totTime % 60);
    var timeStr = curMin + ':' + String(curSec).padStart(2,'0') + ' / ' + totMin + ':' + String(totSec).padStart(2,'0');
    $('#frameCounter').text(timeStr + '  (Frame ' + current + '/' + total + ')');
    $('#frameSlider').attr('max', total).val(current);
}

// ── Video Player ──

function initVideoPlayer(file) {
    AppState.videoFile = file;

    // Revoke previous URL
    if (AppState.videoUrl) {
        URL.revokeObjectURL(AppState.videoUrl);
    }
    AppState.videoUrl = URL.createObjectURL(file);

    var video = document.getElementById('videoPlayer');
    video.src = AppState.videoUrl;
    video.load();

    // Show video card, hide placeholder
    $('#noVideoPlaceholder').hide();
    $('#videoCard').fadeIn(300);
    $('#selectionCard').fadeIn(300);
    $('#videoFileName').show();
    $('#videoFileNameText').text(file.name);

    // Reset state
    AppState.calibrationFrame = null;
    AppState.keypointFrames = [];
    AppState.currentAnalysis = null;
    AppState.analyzing = false;
    // Nuevo video → reset regla de medición + mapa de mínimos
    // Nuevo video → nuevo video_id, sin referencia fijada
    AppState.videoId = generateVideoId();
    AppState.lockedReference = null;
    updateReferenceBadge();
    $('#analysisCard').hide();
    $('#resultsCard').hide();
    $('#screeningCard').hide();
    updateSelectionSummary();

    video.onloadedmetadata = function() {
        // Estimate FPS: try to get from video, default 30
        // HTML5 video doesn't expose FPS directly, use requestVideoFrameCallback if available
        AppState.fps = 30;
        updateFrameCounter();
        drawReferenceOverlay();

        // Try to estimate FPS using requestVideoFrameCallback
        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
            var frameCount = 0;
            var startTime = null;
            var callback = function(now, metadata) {
                frameCount++;
                if (startTime === null) {
                    startTime = metadata.mediaTime;
                }
                if (frameCount >= 10) {
                    var elapsed = metadata.mediaTime - startTime;
                    if (elapsed > 0) {
                        AppState.fps = Math.round(frameCount / elapsed);
                        if (AppState.fps < 10) AppState.fps = 30;
                        if (AppState.fps > 120) AppState.fps = 30;
                        console.log('[ENGINE] Estimated FPS:', AppState.fps);
                    }
                    video.pause();
                    video.currentTime = 0;
                    updateFrameCounter();
                    return;
                }
                video.requestVideoFrameCallback(callback);
            };
            video.requestVideoFrameCallback(callback);
            video.play();
        } else {
            video.pause();
            video.currentTime = 0;
        }
    };

    // Update frame counter on timeupdate
    video.ontimeupdate = function() {
        updateFrameCounter();
    };
}

function seekFrame(delta) {
    var video = document.getElementById('videoPlayer');
    if (!video || !video.duration) return;
    video.pause();
    $('#playPauseIcon').removeClass('fa-pause').addClass('fa-play');
    var newTime = video.currentTime + (delta / AppState.fps);
    newTime = Math.max(0, Math.min(newTime, video.duration));
    video.currentTime = newTime;
    updateFrameCounter();
}

function togglePlayPause() {
    var video = document.getElementById('videoPlayer');
    if (!video) return;
    if (video.paused) {
        video.play();
        $('#playPauseIcon').removeClass('fa-play').addClass('fa-pause');
    } else {
        video.pause();
        $('#playPauseIcon').removeClass('fa-pause').addClass('fa-play');
    }
}

// ── Frame Capture ──

function captureFrame() {
    return new Promise(function(resolve, reject) {
        var video = document.getElementById('videoPlayer');
        var canvas = document.getElementById('captureCanvas');
        if (!video || !canvas) {
            reject('Video or canvas not found');
            return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        var ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(function(blob) {
            if (blob) {
                resolve(blob);
            } else {
                reject('Failed to capture frame');
            }
        }, 'image/jpeg', 0.95);
    });
}

// ── Dismiss Analysis ──

function dismissAnalysis() {
    $('#analysisCard').fadeOut(200);
    $('#analysisLoader').hide();
    $('#analysisContent').hide();
    $('#selectionPanel').hide();
    AppState.analyzing = false;
}

// ── Analyze Frame ──

function analyzeCurrentFrame() {
    if (AppState.analyzing) return;

    var video = document.getElementById('videoPlayer');
    if (!video || !video.duration) {
        alert('No hay video cargado.');
        return;
    }

    // Pause video first
    video.pause();
    $('#playPauseIcon').removeClass('fa-pause').addClass('fa-play');

    var frameNum = getCurrentFrameNum();
    AppState.analyzing = true;

    // Show analysis card with loader, hide previous results
    $('#analysisCard').fadeIn(300);
    $('#analysisLoader').show();
    $('#analysisContent').hide();
    $('#selectionPanel').hide();
    $('#analysisFrameLabel').text('Frame ' + frameNum);

    // Read current breed/category/age from dropdowns
    AppState.breed = $('#breed').val() || 'desconocido';
    AppState.category = $('#category').val() || 'desconocido';
    AppState.age_range = $('#age_range').val() || 'desconocido';

    // Phase 1: Scan frame for all cows and posts
    captureFrame().then(function(blob) {
        var formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');

        $.ajax({
            type: 'POST',
            url: '/scan_frame',
            data: formData,
            contentType: false,
            processData: false,
            timeout: 120000,
            success: function(scanData) {
                if (!scanData.success) {
                    AppState.analyzing = false;
                    $('#analysisLoader').hide();
                    $('#analysisContent').show();
                    $('#analysisDetails').html(
                        '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle"></i> ' +
                        (scanData.error || 'Error en escaneo') + '</div>'
                    );
                    return;
                }

                AppState.scanResult = scanData;
                AppState.frameImageId = scanData.frame_image_id;

                // No bloqueamos si no hay vaca: la calibración de postes NO requiere vaca.
                // Si hay postes, mostramos el panel de selección (con o sin vacas).
                AppState.selectedCowIndex = 0;
                AppState.selectedPostIndices = null;
                if (scanData.cows.length === 0 && scanData.posts.length < 2) {
                    AppState.analyzing = false;
                    $('#analysisLoader').hide();
                    $('#analysisContent').show();
                    $('#analysisDetails').html(
                        '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle"></i> ' +
                        'No hay vacas ni suficientes postes (se necesitan 2) en este frame.</div>'
                    );
                    return;
                }
                showSelectionPanel(scanData, frameNum);
            },
            error: function(xhr, status, error) {
                AppState.analyzing = false;
                $('#analysisLoader').hide();
                $('#analysisContent').show();
                $('#analysisDetails').html(
                    '<div class="alert alert-danger">Error al escanear el frame: ' + error + '</div>'
                );
            }
        });
    }).catch(function(err) {
        AppState.analyzing = false;
        $('#analysisLoader').hide();
        alert('Error al capturar frame: ' + err);
    });
}

// ── Selection Panel (multi-cow / multi-post) ──

function showSelectionPanel(scanData, frameNum) {
    $('#analysisLoader').hide();
    $('#selectionPanel').show();
    $('#analysisContent').hide();

    var html = '';

    // Preview image showing all detections
    if (scanData.preview_image) {
        html += '<div class="text-center mb-3">' +
            '<img src="' + scanData.preview_image + '" class="img-fluid rounded" ' +
            'style="max-width:100%; max-height:400px; border:2px solid var(--border-subtle);">' +
            '</div>';
    }

    html += '<h6 class="mb-3" style="font-weight:700; color:var(--text-dark);">' +
        '<i class="fas fa-mouse-pointer"></i> Seleccionar vaca y postes para analizar' +
        '<span style="font-weight:400; color:var(--text-muted); font-size:0.85em; margin-left:8px;">' +
        '(' + scanData.cows.length + ' vaca' + (scanData.cows.length !== 1 ? 's' : '') +
        ', ' + scanData.posts.length + ' poste' + (scanData.posts.length !== 1 ? 's' : '') + ')' +
        '</span></h6>';

    // Cow section — always show
    html += '<div class="mb-3"><label class="config-label">Vaca a medir</label>' +
        '<div class="cow-selector-grid">';
    scanData.cows.forEach(function(cow) {
        var checked = cow.index === 0 ? ' checked' : '';
        var kpBadge = cow.has_keypoints ? '<span class="badge-kp">KP</span>' : '';
        var thumbSrc = cow.thumbnail_b64
            ? 'data:image/jpeg;base64,' + cow.thumbnail_b64
            : '';
        html += '<label class="cow-option">' +
            '<input type="radio" name="cowSelect" value="' + cow.index + '"' + checked + '>' +
            (thumbSrc ? '<img src="' + thumbSrc + '" class="cow-thumb">' : '<div class="cow-thumb-placeholder"><i class="fas fa-cow"></i></div>') +
            '<div class="cow-info">' +
            '<span class="cow-label">Vaca ' + (cow.index + 1) + '</span>' +
            '<span class="cow-score">' + (cow.score * 100).toFixed(0) + '% ' + kpBadge + '</span>' +
            '</div></label>';
    });
    html += '</div></div>';

    var hasLockedRef = !!AppState.lockedReference;

    if (hasLockedRef) {
        // Con referencia fija: NO mostramos selección de postes
        html += '<div class="alert alert-info py-2 px-3 mb-3" style="font-size:0.9em;">' +
            '<i class="fas fa-thumbtack"></i> Usando <strong>referencia fijada</strong> del video ' +
            '(no hace falta seleccionar postes). ' +
            '<button class="btn btn-sm btn-outline-warning ms-2" id="btnClearReferenceInPanel">' +
            '<i class="fas fa-times"></i> Re-calibrar</button></div>';
    } else {
        // Sin ref fija: sección de postes para escala/calibración
        if (scanData.posts.length > 0) {
            html += '<div class="mb-3"><label class="config-label">Postes para escala</label>' +
                '<div class="post-selector-list">';
            scanData.posts.forEach(function(post) {
                var bandClass = post.in_band ? 'post-in-band' : '';
                var heightStr = post.measured_height_px ? post.measured_height_px.toFixed(0) + 'px' : '?';
                var postThumbSrc = post.thumbnail_b64
                    ? 'data:image/jpeg;base64,' + post.thumbnail_b64
                    : '';
                html += '<label class="post-option ' + bandClass + '">' +
                    '<input type="checkbox" name="postSelect" value="' + post.index + '" checked>' +
                    (postThumbSrc ? '<img src="' + postThumbSrc + '" class="post-thumb">' : '') +
                    '<span class="post-info">' +
                    'Poste ' + (post.index + 1) +
                    ' <span class="post-detail">(' + heightStr + ', ' + (post.score * 100).toFixed(0) + '%' +
                    (post.in_band ? ', en banda' : '') + ')</span>' +
                    '</span></label>';
            });
            html += '</div></div>';
        } else {
            html += '<div class="mb-3"><span style="color:var(--text-muted); font-size:0.85em;">' +
                '<i class="fas fa-info-circle"></i> No se detectaron postes en este frame</span></div>';
        }
    }

    // Botones
    html += '<div class="text-center" style="display:flex; gap:10px; justify-content:center; flex-wrap:wrap;">';
    if (!hasLockedRef && scanData.posts.length >= 2) {
        html += '<button class="btn btn-outline-primary" id="btnCalibrateOnly" style="padding:10px 22px;">' +
            '<i class="fas fa-thumbtack"></i> Solo calibrar postes</button>';
    }
    if (scanData.cows.length > 0) {
        html += '<button class="btn btn-analyze" id="btnConfirmSelection" style="padding:10px 28px;">' +
            '<i class="fas fa-check"></i> Confirmar y Analizar</button>';
    }
    html += '</div>';

    $('#selectionPanel').html(html);

    // Bind calibrate-only button
    $('#btnCalibrateOnly').off('click').on('click', function() {
        var checkedPosts = [];
        $('input[name="postSelect"]:checked').each(function() {
            checkedPosts.push(parseInt($(this).val()));
        });
        if (checkedPosts.length !== 2) {
            alert('Seleccioná exactamente 2 postes para calibrar.');
            return;
        }
        runCalibrateOnly(checkedPosts, frameNum);
    });

    // Bind "Re-calibrar" dentro del panel
    $('#btnClearReferenceInPanel').off('click').on('click', function() {
        clearCurrentReference();
        // Volver a mostrar el panel pero ya sin hasLockedRef
        showSelectionPanel(scanData, frameNum);
    });

    // Bind confirm button
    $('#btnConfirmSelection').off('click').on('click', function() {
        // Read cow selection
        var cowVal = $('input[name="cowSelect"]:checked').val();
        AppState.selectedCowIndex = cowVal !== undefined ? parseInt(cowVal) : 0;

        // Read post selection (solo si no hay ref fijada)
        var checkedPosts = [];
        if (!AppState.lockedReference) {
            $('input[name="postSelect"]:checked').each(function() {
                checkedPosts.push(parseInt($(this).val()));
            });
            if (scanData.posts.length > 0 && checkedPosts.length === 0) {
                alert('Necesitas al menos 1 poste seleccionado.');
                return;
            }
        }

        if (AppState.lockedReference) {
            AppState.selectedPostIndices = null;
        } else if (scanData.posts.length === 0) {
            AppState.selectedPostIndices = null;
        } else {
            AppState.selectedPostIndices = checkedPosts;
        }

        $('#selectionPanel').hide();
        $('#analysisLoader').show();
        runAnalyzeWithSelection(frameNum);
    });
}

function runAnalyzeWithSelection(frameNum) {
    var formData = new FormData();
    formData.append('frame_image_id', AppState.frameImageId);
    formData.append('cow_index', AppState.selectedCowIndex);
    if (AppState.selectedPostIndices !== null) {
        formData.append('post_indices', AppState.selectedPostIndices.join(','));
    }
    formData.append('breed', AppState.breed);
    formData.append('category', AppState.category);
    formData.append('age_range', AppState.age_range);
    // Enviar video_id para que el backend use la referencia fijada si existe
    if (AppState.videoId) {
        formData.append('video_id', AppState.videoId);
    }
    // Mandar tambien la ref inline (fallback si el backend reinicio)
    if (AppState.lockedReference) {
        formData.append('locked_reference_json', JSON.stringify({
            post1: AppState.lockedReference.post1,
            post2: AppState.lockedReference.post2,
        }));
    }

    $.ajax({
        type: 'POST',
        url: '/analyze_frame',
        data: formData,
        contentType: false,
        processData: false,
        timeout: 120000,
        success: function(data) {
            AppState.analyzing = false;
            AppState.currentAnalysis = data;
            AppState.currentAnalysis._frameNum = frameNum;
            displayAnalysis(data, frameNum);
        },
        error: function(xhr, status, error) {
            AppState.analyzing = false;
            $('#analysisLoader').hide();
            $('#analysisContent').show();
            $('#analysisDetails').html(
                '<div class="alert alert-danger">Error al analizar el frame: ' + error + '</div>'
            );
        }
    });
}

// ── Display Analysis Result ──

function displayAnalysis(data, frameNum) {
    $('#analysisLoader').hide();
    $('#analysisContent').show();
    $('#analysisFrameLabel').text('Frame ' + frameNum);

    if (!data.success) {
        $('#annotatedImage').hide();
        $('#analysisDetails').html(
            '<div class="alert alert-warning">' +
            '<i class="fas fa-exclamation-triangle"></i> ' + (data.error || 'Error desconocido') +
            '</div>'
        );
        $('#btnUseCalibration').prop('disabled', true);
        $('#btnUseKeypoint').prop('disabled', true);
        return;
    }

    // Show annotated image
    if (data.annotated_image) {
        $('#annotatedImage').attr('src', data.annotated_image).show();
    } else {
        $('#annotatedImage').hide();
    }

    var d = data.details || {};
    var detailsHtml = '<div class="details-grid">';

    // Postes Detectados / Promedio SOLO si NO hay referencia fijada (ya no son relevantes)
    if (!d.locked_ref_used) {
        var postesClass = d.postes_detected >= 2 ? 'detail-success' : (d.postes_detected >= 1 ? 'detail-warning' : 'detail-danger');
        var postesValue = '' + (d.postes_detected || 0);
        if (d.postes_heights_px && d.postes_heights_px.length > 0) {
            var heightsStr = d.postes_heights_px.map(function(h) { return h.toFixed(0) + 'px'; }).join(', ');
            postesValue += ' (' + heightsStr + ')';
        }
        detailsHtml += '<div class="detail-item ' + postesClass + '">' +
            '<div class="label">Postes Detectados</div>' +
            '<div class="value">' + postesValue + '</div></div>';

        if (d.postes_heights_px && d.postes_heights_px.length >= 2) {
            var avgH = d.postes_heights_px.reduce(function(a, b) { return a + b; }, 0) / d.postes_heights_px.length;
            detailsHtml += '<div class="detail-item detail-success">' +
                '<div class="label">Promedio Postes</div>' +
                '<div class="value">' + avgH.toFixed(1) + ' px = 50cm</div></div>';
        }
    }

    // Cow height
    if (d.cow_height_cm) {
        detailsHtml += '<div class="detail-item detail-success">' +
            '<div class="label">Altura Vaca</div>' +
            '<div class="value">' + d.cow_height_cm.toFixed(1) + ' cm</div></div>';
    }

    // Animal bbox height
    if (d.animal_bbox_height_px) {
        detailsHtml += '<div class="detail-item">' +
            '<div class="label">Altura BBox (px)</div>' +
            '<div class="value">' + d.animal_bbox_height_px.toFixed(0) + ' px</div></div>';
    }

    // cm/px
    if (d.cm_per_px) {
        detailsHtml += '<div class="detail-item">' +
            '<div class="label">Escala</div>' +
            '<div class="value">' + d.cm_per_px.toFixed(5) + ' cm/px</div></div>';
    }

    // Keypoints
    var kpClass = d.keypoints_found ? 'detail-success' : 'detail-danger';
    detailsHtml += '<div class="detail-item ' + kpClass + '">' +
        '<div class="label">Keypoints</div>' +
        '<div class="value">' + (d.keypoints_found ? 'Detectados' : 'No detectados') + '</div></div>';

    // dist1 (BL)
    if (d.dist1_px) {
        detailsHtml += '<div class="detail-item">' +
            '<div class="label">BL (dist1)</div>' +
            '<div class="value">' + d.dist1_px.toFixed(1) + ' px</div></div>';
    }

    // dist2 (Girth)
    if (d.dist2_px) {
        detailsHtml += '<div class="detail-item">' +
            '<div class="label">Girth (dist2)</div>' +
            '<div class="value">' + d.dist2_px.toFixed(1) + ' px</div></div>';
    }

    // Weight from backend (if available)
    if (data.weight_kg) {
        detailsHtml += '<div class="detail-item detail-success">' +
            '<div class="label">Peso (backend)</div>' +
            '<div class="value">' + data.weight_kg.toFixed(2) + ' kg</div></div>';
    }

    detailsHtml += '</div>';

    // Message
    if (d.message) {
        detailsHtml += '<div class="mt-2"><small class="text-muted"><i class="fas fa-info-circle"></i> ' + d.message + '</small></div>';
    }

    // Sección de Referencia Fijada
    var refHtml = '';
    if (d.locked_ref_used) {
        refHtml = '<div class="alert alert-info mt-2 mb-0 py-2 px-3" style="font-size:0.9em;">' +
            '<i class="fas fa-thumbtack"></i> <strong>Usando referencia fijada</strong> del video. ' +
            '<button class="btn btn-sm btn-outline-warning ms-2" id="btnClearReference">' +
            '<i class="fas fa-times"></i> Re-calibrar</button></div>';
    } else if (d.rectangle_ref && d.rectangle_ref.post1 && d.rectangle_ref.post2) {
        refHtml = '<div class="alert alert-success mt-2 mb-0 py-2 px-3" style="font-size:0.9em;">' +
            '<i class="fas fa-ruler"></i> <strong>Rectángulo de escala detectado</strong>. ' +
            '<button class="btn btn-sm btn-primary ms-2" id="btnLockReference">' +
            '<i class="fas fa-thumbtack"></i> Fijar referencia para todo el video</button></div>';
    }
    detailsHtml += refHtml;

    $('#analysisDetails').html(detailsHtml);

    // Bind botón Fijar referencia
    $('#btnLockReference').off('click').on('click', function() {
        lockCurrentReference(d.rectangle_ref);
    });
    $('#btnClearReference').off('click').on('click', function() {
        clearCurrentReference();
    });

    // Enable/disable selection buttons
    // Calibration: needs 2 postes + cow detected (cow_height_cm as portable ruler)
    var canCalibrate = d.postes_detected >= 2 && d.cow_height_cm && d.animal_bbox_height_px;
    $('#btnUseCalibration').prop('disabled', !canCalibrate);
    if (!canCalibrate) {
        $('#btnUseCalibration').attr('title', 'Necesita 2 postes + vaca detectada para calibrar');
    } else {
        $('#btnUseCalibration').attr('title', 'Usar altura de vaca (' + d.cow_height_cm.toFixed(1) + ' cm) como referencia');
    }

    // Keypoints: needs cow + keypoints detected + dist1 + dist2 + animal_bbox_height
    var canKeypoint = d.keypoints_found && d.dist1_px && d.dist2_px && d.animal_bbox_height_px;
    $('#btnUseKeypoint').prop('disabled', !canKeypoint);
    if (!canKeypoint) {
        $('#btnUseKeypoint').attr('title', 'Necesita vaca + keypoints (BL y Girth) detectados');
    } else {
        $('#btnUseKeypoint').attr('title', 'Guardar distancias de este frame para calcular peso');
    }
}

// ── Selection Functions ──

function selectAsCalibration() {
    if (!AppState.currentAnalysis || !AppState.currentAnalysis.success) return;
    var d = AppState.currentAnalysis.details || {};
    if (!d.cow_height_cm || !d.animal_bbox_height_px) return;

    AppState.calibrationFrame = {
        frameNum: AppState.currentAnalysis._frameNum,
        data: AppState.currentAnalysis,
        postIndices: AppState.selectedPostIndices  // remember which posts were used (null = all)
    };

    updateSelectionSummary();
    updateCalculateButton();
}

function selectAsKeypoint() {
    if (!AppState.currentAnalysis || !AppState.currentAnalysis.success) return;
    var d = AppState.currentAnalysis.details || {};
    if (!d.keypoints_found || !d.dist1_px || !d.dist2_px || !d.animal_bbox_height_px) return;

    var frameNum = AppState.currentAnalysis._frameNum;

    // Don't add duplicate frames
    for (var i = 0; i < AppState.keypointFrames.length; i++) {
        if (AppState.keypointFrames[i].frameNum === frameNum) {
            alert('Este frame ya esta seleccionado como keypoint (Frame ' + frameNum + ').');
            return;
        }
    }

    AppState.keypointFrames.push({
        frameNum: frameNum,
        data: AppState.currentAnalysis
    });

    updateSelectionSummary();
    updateCalculateButton();
}

function removeCalibration() {
    AppState.calibrationFrame = null;
    updateSelectionSummary();
    updateCalculateButton();
    // Also clear results
    $('#resultsCard').hide();
}

function removeKeypoint(index) {
    AppState.keypointFrames.splice(index, 1);
    updateSelectionSummary();
    updateCalculateButton();
    // Also clear results
    $('#resultsCard').hide();
}

// ── UI Update Functions ──

function updateSelectionSummary() {
    // Calibration summary
    if (AppState.calibrationFrame) {
        var cf = AppState.calibrationFrame;
        var cd = cf.data.details || {};
        $('#calibrationSummary').html(
            '<span class="selection-badge badge-calibration">' +
            '<i class="fas fa-ruler-combined icon-calibration"></i> ' +
            'Frame ' + cf.frameNum +
            ' (altura = ' + (cd.cow_height_cm ? cd.cow_height_cm.toFixed(1) : '?') + ' cm, ' +
            (cd.postes_detected || 0) + ' postes)' +
            ' <button class="btn-remove" onclick="removeCalibration()" title="Quitar">&times;</button>' +
            '</span>'
        );
    } else {
        $('#calibrationSummary').html('<span style="color: var(--text-muted);">No seleccionado - analiza un frame con 2 postes + vaca</span>');
    }

    // Keypoints summary
    if (AppState.keypointFrames.length > 0) {
        var kpHtml = '';
        AppState.keypointFrames.forEach(function(kf, idx) {
            var kd = kf.data.details || {};
            kpHtml += '<span class="selection-badge badge-keypoint">' +
                '<i class="fas fa-crosshairs icon-keypoint"></i> ' +
                'Frame ' + kf.frameNum +
                ' (BL=' + (kd.dist1_px ? kd.dist1_px.toFixed(0) : '?') + 'px, ' +
                'Girth=' + (kd.dist2_px ? kd.dist2_px.toFixed(0) : '?') + 'px)' +
                ' <button class="btn-remove" onclick="removeKeypoint(' + idx + ')" title="Quitar">&times;</button>' +
                '</span>';
        });
        $('#keypointsSummary').html(kpHtml);
    } else {
        $('#keypointsSummary').html('<span style="color: var(--text-muted);">Ninguno seleccionado - analiza frames con vaca + keypoints</span>');
    }
}

function updateCalculateButton() {
    var canCalculate = AppState.calibrationFrame !== null && AppState.keypointFrames.length > 0;
    $('#btnCalculateWeight').prop('disabled', !canCalculate);
    if (canCalculate) {
        $('#btnCalculateWeight').html(
            '<i class="fas fa-calculator"></i> Calcular Peso (' + AppState.keypointFrames.length + ' frames)'
        );
    } else {
        var missing = [];
        if (!AppState.calibrationFrame) missing.push('calibracion');
        if (AppState.keypointFrames.length === 0) missing.push('keypoints');
        $('#btnCalculateWeight').html(
            '<i class="fas fa-calculator"></i> Calcular Peso (falta: ' + missing.join(', ') + ')'
        );
    }
}

// ── Weight Calculation (client-side) ──

function calculateWeights() {
    if (!AppState.calibrationFrame || AppState.keypointFrames.length === 0) {
        alert('Necesitas seleccionar un frame de calibracion y al menos un frame de keypoints.');
        return;
    }

    var calib = AppState.calibrationFrame;
    var fixedHeightCm = calib.data.details.cow_height_cm;

    if (!fixedHeightCm || fixedHeightCm <= 0) {
        alert('El frame de calibracion no tiene una altura valida.');
        return;
    }

    // Read current breed/category/age
    var breed = $('#breed').val() || 'desconocido';
    var category = $('#category').val() || 'desconocido';
    var age = $('#age_range').val() || 'desconocido';

    var k_breed = BREED_K[breed] || 1.0;
    var k_category = CATEGORY_K[category] || 1.0;
    var k_age = AGE_K[age] || 1.0;
    var multiplier = k_breed * k_category * k_age;

    var results = [];
    AppState.keypointFrames.forEach(function(kf) {
        var d = kf.data.details;
        if (!d || !d.dist1_px || !d.dist2_px || !d.animal_bbox_height_px) return;
        if (d.animal_bbox_height_px <= 0) return;

        var cm_per_px = fixedHeightCm / d.animal_bbox_height_px;
        var dist1_cm = d.dist1_px * cm_per_px;
        var dist2_cm = d.dist2_px * cm_per_px;

        // Schaeffer formula: Weight(kg) = (BL * GirthVert^2 * lb_to_kg) / 300 * multiplier
        var lb = 0.45359237;
        var weight = (dist1_cm * dist2_cm * dist2_cm * lb) / 300.0 * multiplier;

        results.push({
            frame: kf.frameNum,
            dist1_px: d.dist1_px,
            dist2_px: d.dist2_px,
            animal_height_px: d.animal_bbox_height_px,
            cm_per_px: cm_per_px,
            dist1_cm: dist1_cm,
            dist2_cm: dist2_cm,
            weight_kg: weight
        });
    });

    renderResults(results, fixedHeightCm, multiplier, breed, category, age);
}

// ── Render Results ──

function renderResults(results, fixedHeightCm, multiplier, breed, category, age) {
    if (results.length === 0) {
        $('#resultsContent').html(
            '<div class="alert alert-warning">' +
            '<i class="fas fa-exclamation-triangle"></i> No se pudieron calcular pesos. ' +
            'Verifica que los frames de keypoints tengan dist1, dist2 y altura del animal.' +
            '</div>'
        );
        $('#resultsCard').fadeIn(300);
        return;
    }

    var weights = results.map(function(r) { return r.weight_kg; });
    var avgWeight = weights.reduce(function(a, b) { return a + b; }, 0) / weights.length;
    var minWeight = Math.min.apply(null, weights);
    var maxWeight = Math.max.apply(null, weights);

    // Standard deviation
    var mean = avgWeight;
    var variance = weights.reduce(function(sum, w) { return sum + (w - mean) * (w - mean); }, 0) / weights.length;
    var stdDev = Math.sqrt(variance);

    var html = '';

    // Summary stat cards
    html += '<div class="results-summary">' +
        '<div class="stat-card stat-primary">' +
        '<div class="stat-label">PROMEDIO</div>' +
        '<div class="stat-value">' + avgWeight.toFixed(1) + ' kg</div>' +
        '</div>' +
        '<div class="stat-card">' +
        '<div class="stat-label">RANGO</div>' +
        '<div class="stat-value">' + minWeight.toFixed(1) + ' - ' + maxWeight.toFixed(1) + ' kg</div>' +
        '</div>' +
        '<div class="stat-card">' +
        '<div class="stat-label">DESV. STD</div>' +
        '<div class="stat-value">' + stdDev.toFixed(1) + ' kg</div>' +
        '</div>' +
        '<div class="stat-card">' +
        '<div class="stat-label">FRAMES</div>' +
        '<div class="stat-value">' + results.length + '</div>' +
        '</div>' +
        '</div>';

    // Parameters
    html += '<div class="params-card">' +
        '<strong><i class="fas fa-sliders-h"></i> Parametros:</strong> ' +
        'Altura fija: ' + fixedHeightCm.toFixed(1) + ' cm (del frame de calibracion) | ' +
        'Raza: ' + breed + ' (K=' + (BREED_K[breed] || 1.0).toFixed(2) + ') | ' +
        'Categoria: ' + category + ' (K=' + (CATEGORY_K[category] || 1.0).toFixed(2) + ') | ' +
        'Edad: ' + age + ' (K=' + (AGE_K[age] || 1.0).toFixed(2) + ') | ' +
        'Multiplicador total: ' + multiplier.toFixed(4) +
        '</div>';

    // Table
    html += '<table class="results-table">' +
        '<thead><tr>' +
        '<th>Frame</th>' +
        '<th>Altura (px)</th>' +
        '<th>cm/px</th>' +
        '<th>BL (px)</th>' +
        '<th>BL (cm)</th>' +
        '<th>Girth (px)</th>' +
        '<th>Girth (cm)</th>' +
        '<th>Peso (kg)</th>' +
        '</tr></thead><tbody>';

    results.forEach(function(r) {
        var weightClass = '';
        if (Math.abs(r.weight_kg - maxWeight) < 0.01) weightClass = 'row-max';
        else if (Math.abs(r.weight_kg - minWeight) < 0.01) weightClass = 'row-min';

        html += '<tr class="' + weightClass + '">' +
            '<td>' + r.frame + '</td>' +
            '<td>' + r.animal_height_px.toFixed(0) + '</td>' +
            '<td>' + r.cm_per_px.toFixed(5) + '</td>' +
            '<td>' + r.dist1_px.toFixed(1) + '</td>' +
            '<td>' + r.dist1_cm.toFixed(2) + '</td>' +
            '<td>' + r.dist2_px.toFixed(1) + '</td>' +
            '<td>' + r.dist2_cm.toFixed(2) + '</td>' +
            '<td><strong>' + r.weight_kg.toFixed(2) + '</strong></td>' +
            '</tr>';
    });

    // Average row
    var avgDist1Cm = results.reduce(function(a, r) { return a + r.dist1_cm; }, 0) / results.length;
    var avgDist2Cm = results.reduce(function(a, r) { return a + r.dist2_cm; }, 0) / results.length;
    html += '<tr class="row-average">' +
        '<td>PROMEDIO</td>' +
        '<td>-</td>' +
        '<td>-</td>' +
        '<td>-</td>' +
        '<td>' + avgDist1Cm.toFixed(2) + '</td>' +
        '<td>-</td>' +
        '<td>' + avgDist2Cm.toFixed(2) + '</td>' +
        '<td><strong>' + avgWeight.toFixed(2) + '</strong></td>' +
        '</tr>';

    html += '</tbody></table>';

    // Legend
    html += '<div class="results-legend">' +
        '<span class="legend-badge legend-max">Rojo</span> = Peso maximo | ' +
        '<span class="legend-badge legend-min">Verde</span> = Peso minimo | ' +
        'Formula: (BL_cm x Girth_cm&sup2; x 0.4536) / 300 x K' +
        '</div>';

    $('#resultsContent').html(html);
    $('#resultsCard').fadeIn(300);

    // Scroll to results
    $('html, body').animate({
        scrollTop: $('#resultsCard').offset().top - 20
    }, 500);
}

// ── Video → Modelo 3D ──

function startVideo3D(mode) {
    if (AppState.modelo3dActive) return;
    mode = mode || 'hibrido';

    var calib = AppState.calibrationFrame;
    if (!calib || !calib.data || !calib.data.details || !calib.data.details.cm_per_px) {
        alert('Necesitas calibrar primero (2 postes) para obtener la escala cm/px.');
        return;
    }

    var cm_per_px = calib.data.details.cm_per_px;
    var cowHeightCm = calib.data.details.cow_height_cm;
    if (!cowHeightCm) {
        alert('No se encontro la altura calibrada de la vaca. Recalibra con 2 postes.');
        return;
    }
    var frameInterval = parseInt($('#modelo3dInterval').val()) || 30;
    var vacaName = $('#modelo3dName').val() || 'vaca_video';

    AppState.modelo3dActive = true;
    AppState.modelo3dMode = mode;
    AppState.modelo3dResults = [];
    AppState.modelo3dAbortController = new AbortController();

    // Show card and progress with mode label
    var modeLabel = mode === 'sfm' ? '(Multi-frame)' : '(Hibrido)';
    $('#modelo3dModeLabel').text(modeLabel);
    $('#modelo3dResultsCard').fadeIn(300);
    $('#modelo3dProgress').show();
    $('#modelo3dSummary').hide();
    $('#btnCancelModelo3D').show();
    $('#modelo3dProgressBar').css('width', '0%');
    $('#modelo3dProgressText').text('Iniciando ' + modeLabel + '...');
    $('#modelo3dProgressCount').text('0 / ?');
    $('#btnModelo3DHibrido').prop('disabled', true);
    $('#btnModelo3DSfm').prop('disabled', true);

    var formData = new FormData();
    formData.append('video', AppState.videoFile);
    formData.append('cow_height_cm', cowHeightCm);
    formData.append('frame_interval', frameInterval);
    formData.append('vaca_name', vacaName);
    formData.append('mode', mode);

    fetch('/api/video_modelo3d', {
        method: 'POST',
        body: formData,
        signal: AppState.modelo3dAbortController.signal
    }).then(function(response) {
        if (!response.ok) throw new Error('Server error: ' + response.status);

        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';

        function processChunk() {
            return reader.read().then(function(result) {
                if (result.done) {
                    if (AppState.modelo3dActive) finishModelo3D(null);
                    return;
                }

                buffer += decoder.decode(result.value, {stream: true});
                var parts = buffer.split('\n\n');
                buffer = parts.pop() || '';

                for (var i = 0; i < parts.length; i++) {
                    var eventStr = parts[i].trim();
                    if (!eventStr) continue;

                    var eventType = '', eventData = '';
                    var eventLines = eventStr.split('\n');
                    for (var j = 0; j < eventLines.length; j++) {
                        var line = eventLines[j];
                        if (line.indexOf('event: ') === 0) eventType = line.substring(7);
                        else if (line.indexOf('data: ') === 0) eventData = line.substring(6);
                    }
                    if (eventType && eventData) handleModelo3DEvent(eventType, eventData);
                }

                if (AppState.modelo3dActive) return processChunk();
            });
        }
        return processChunk();
    }).catch(function(err) {
        if (err.name === 'AbortError') {
            finishModelo3D(null, true);
        } else {
            console.error('[MODELO3D] Error:', err);
            finishModelo3D(null, false, err.message);
        }
    });

    $('html, body').animate({ scrollTop: $('#modelo3dResultsCard').offset().top - 20 }, 500);
}

function handleModelo3DEvent(eventType, dataStr) {
    try { var data = JSON.parse(dataStr); } catch(e) { return; }

    switch (eventType) {
        case 'started':
            AppState.modelo3dTotalFrames = data.frames_to_process;
            var durInfo = data.duration_sec ? ' (video: ' + data.duration_sec + 's, ' + data.fps + 'fps)' : '';
            $('#modelo3dProgressText').text('Extrayendo frames...' + durInfo);
            $('#modelo3dProgressCount').text('0 / ' + data.frames_to_process);
            console.log('[MODELO3D] Video: ' + data.total_frames + ' frames, ' + data.fps + 'fps, ' + data.duration_sec + 's, to_process=' + data.frames_to_process);
            break;

        case 'extracting':
            var pct = data.total > 0 ? Math.round(data.extracted / data.total * 50) : 0; // Phase 1 = 0-50%
            $('#modelo3dProgressBar').css('width', pct + '%');
            $('#modelo3dProgressCount').text(data.extracted + ' / ' + data.total);
            $('#modelo3dProgressText').text('Extrayendo frame ' + data.extracted + '/' + data.total + ' (' + data.accepted + ' aceptados)');
            break;

        case 'sfm_progress':
            var pct2 = 50 + (data.total_steps > 0 ? Math.round(data.step / data.total_steps * 50) : 0); // Phase 2 = 50-100%
            $('#modelo3dProgressBar').css('width', pct2 + '%');
            $('#modelo3dProgressCount').text(data.step + ' / ' + data.total_steps);
            var mLabel = AppState.modelo3dMode === 'sfm' ? 'Multi-frame' : 'Hibrido';
            $('#modelo3dProgressText').text(mLabel + ': ' + data.message);
            break;

        case 'complete':
            finishModelo3D(data);
            break;

        case 'error':
            finishModelo3D(null, false, data.message);
            break;
    }
}

function finishModelo3D(summary, cancelled, errorMsg) {
    AppState.modelo3dActive = false;
    $('#btnCancelModelo3D').hide();
    $('#btnModelo3DHibrido').prop('disabled', false);
    $('#btnModelo3DSfm').prop('disabled', false);

    if (cancelled) {
        $('#modelo3dProgressText').text('Cancelado.');
        return;
    }

    if (errorMsg) {
        $('#modelo3dProgress').hide();
        $('#modelo3dSummary').show().html(
            '<div class="screening-error-banner"><i class="fas fa-exclamation-triangle"></i> Error: ' + errorMsg + '</div>'
        );
        return;
    }

    $('#modelo3dProgress').hide();

    if (!summary) {
        $('#modelo3dSummary').show().html(
            '<div class="text-center py-3" style="color:var(--text-muted);">No se generaron resultados validos.</div>'
        );
        return;
    }

    renderModelo3DResults(summary);
}

function renderModelo3DResults(summary) {
    // Fill summary cards
    $('#m3dVolumen').text(summary.volumen_litros + ' L');
    $('#m3dPeso').text(summary.peso_kg + ' kg');
    $('#m3dAltura').text(summary.alto_cm + ' cm');
    $('#m3dPuntos').text(summary.num_points);

    // peso_kg ahora ya es del barril; la columna duplicada queda oculta
    $('#m3dPesoBarrilCol').hide();

    $('#modelo3dSummary').show();

    // Store PLY info for viewer button
    AppState.modelo3dPlyId = summary.ply_id;
    AppState.modelo3dPly3d = summary.ply_3d;
    AppState.modelo3dSummaryData = summary;

    // Refresh model list in 3D viewer so the new model appears
    if (typeof loadModelosDisponibles === 'function') {
        loadModelosDisponibles(summary.ply_id);
    }

    // Auto-increment name for next run (e.g. vaca_video → vaca_video_2)
    var currentName = $('#modelo3dName').val() || 'vaca_video';
    var match = currentName.match(/^(.+?)_(\d+)$/);
    if (match) {
        $('#modelo3dName').val(match[1] + '_' + (parseInt(match[2]) + 1));
    } else {
        $('#modelo3dName').val(currentName + '_2');
    }

    // Show "Ver Modelo 3D" button
    $('#btnVerModelo3DResult').off('click').on('click', function() {
        var url = '/api/modelo3d/' + summary.ply_id + '/' + summary.ply_3d;
        if (typeof window.viewer3dLoadModel === 'function') {
            window.viewer3dLoadModel(url, {
                peso_kg: summary.peso_kg,
                volumen_litros: summary.volumen_litros,
                largo_cm: summary.largo_cm,
                alto_cm: summary.alto_cm,
            }, summary.ply_id);
            $('html, body').animate({ scrollTop: $('#viewer3dCard').offset().top - 20 }, 500);
        } else {
            alert('El visor 3D no esta disponible.');
        }
    });

}

function cancelModelo3D() {
    if (AppState.modelo3dAbortController) {
        AppState.modelo3dAbortController.abort();
    }
}

// ====================================================================
// Document Ready - Event Handlers
// ====================================================================

$(document).ready(function() {

    // ── Card collapse toggles (click en chevron de cada header) ──
    $(document).on('click', '.card-collapse-toggle', function(e) {
        e.stopPropagation();
        var target = $(this).data('target');
        if (!target) return;
        var $body = $('#' + target);
        var $icon = $(this).find('i');
        $body.slideToggle(200);
        $icon.toggleClass('fa-chevron-up fa-chevron-down');
    });

    // ── Video Upload ──

    $('#videoUpload').on('change', function() {
        var file = this.files[0];
        if (!file) return;

        if (!file.type.startsWith('video/')) {
            alert('Por favor selecciona un archivo de video.');
            return;
        }

        initVideoPlayer(file);
    });

    // Redibujar overlay al redimensionar ventana y al hacer resize del video
    window.addEventListener('resize', function() {
        drawReferenceOverlay();
    });
    var vidEl = document.getElementById('videoPlayer');
    if (vidEl && 'ResizeObserver' in window) {
        new ResizeObserver(function() { drawReferenceOverlay(); }).observe(vidEl);
    }

    // ── Video Controls ──

    $('#btnSkipBack').on('click', function() { seekFrame(-10); });
    $('#btnPrev').on('click', function() { seekFrame(-1); });
    $('#btnPlayPause').on('click', function() { togglePlayPause(); });
    $('#btnNext').on('click', function() { seekFrame(1); });
    $('#btnSkipForward').on('click', function() { seekFrame(10); });
    $('#btnLiveAnalyze').on('click', function() { toggleLiveAnalyze(); });
    $('#btnDetectPassings').on('click', function() { toggleDetectPassings(); });
    $('#btnSave21Frames').on('click', function() { saveFramesAround(); });
    $('#btnViewSavedFrames').on('click', function() { showSavedFrames(); });
    $('#savedFramesClose').on('click', function() { $('#savedFramesModal').hide(); });
    $('#savedFramesRefresh').on('click', function() { refreshSavedFolders(); });
    $('#savedFramesModal').on('click', function(e) {
        if (e.target.id === 'savedFramesModal') $(this).hide();
    });

    // Frame slider
    $('#frameSlider').on('input', function() {
        var video = document.getElementById('videoPlayer');
        if (!video || !video.duration) return;
        video.pause();
        $('#playPauseIcon').removeClass('fa-pause').addClass('fa-play');
        var frameNum = parseInt($(this).val());
        video.currentTime = frameNum / AppState.fps;
        updateFrameCounter();
    });

    // Keyboard shortcuts
    $(document).on('keydown', function(e) {
        // Only if video is loaded and no input is focused
        if (!AppState.videoUrl) return;
        if ($(e.target).is('input, select, textarea')) return;

        switch(e.key) {
            case 'ArrowLeft':
                e.preventDefault();
                seekFrame(e.shiftKey ? -10 : -1);
                break;
            case 'ArrowRight':
                e.preventDefault();
                seekFrame(e.shiftKey ? 10 : 1);
                break;
            case ' ':
                e.preventDefault();
                togglePlayPause();
                break;
            case 'a':
            case 'A':
                e.preventDefault();
                analyzeCurrentFrame();
                break;
        }
    });

    // ── Analyze Button ──

    $('#btnAnalyze').on('click', function() {
        analyzeCurrentFrame();
    });

    // ── Selection Buttons ──

    $('#btnUseCalibration').on('click', function() {
        selectAsCalibration();
    });

    $('#btnUseKeypoint').on('click', function() {
        selectAsKeypoint();
    });

    // ── Calculate Weight ──

    $('#btnCalculateWeight').on('click', function() {
        calculateWeights();
    });

    // ── Dropdown changes → clear results ──

    $('#breed, #category, #age_range').on('change', function() {
        AppState.breed = $('#breed').val() || 'desconocido';
        AppState.category = $('#category').val() || 'desconocido';
        AppState.age_range = $('#age_range').val() || 'desconocido';
        // If we have results, recalculate with new multiplier
        if ($('#resultsCard').is(':visible') && AppState.calibrationFrame && AppState.keypointFrames.length > 0) {
            calculateWeights();
        }
    });

    // ── Modelo 3D desde Video ──

    $('#btnModelo3DHibrido').on('click', function() {
        startVideo3D('hibrido');
    });

    $('#btnModelo3DSfm').on('click', function() {
        startVideo3D('sfm');
    });

    $('#btnCancelModelo3D').on('click', function() {
        cancelModelo3D();
    });

    // ── Initialize UI ──

    updateSelectionSummary();
    updateCalculateButton();
});
