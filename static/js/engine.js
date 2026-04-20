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
    // Batch screening
    screeningResults: [],
    screeningActive: false,
    screeningAbortController: null,
    screeningSortCol: null,
    screeningSortAsc: true,
    screeningTableResults: []
};

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
    AppState.screeningResults = [];
    AppState.screeningActive = false;
    if (AppState.screeningAbortController) {
        AppState.screeningAbortController.abort();
        AppState.screeningAbortController = null;
    }
    $('#analysisCard').hide();
    $('#resultsCard').hide();
    $('#screeningCard').hide();
    updateSelectionSummary();

    video.onloadedmetadata = function() {
        // Estimate FPS: try to get from video, default 30
        // HTML5 video doesn't expose FPS directly, use requestVideoFrameCallback if available
        AppState.fps = 30;
        updateFrameCounter();

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

                if (scanData.cows.length === 0) {
                    // No cows detected
                    AppState.analyzing = false;
                    $('#analysisLoader').hide();
                    $('#analysisContent').show();
                    $('#analysisDetails').html(
                        '<div class="alert alert-danger"><i class="fas fa-times-circle"></i> No se detectaron vacas en este frame.</div>'
                    );
                    return;
                }

                // Always show selection panel so user can see/pick cows and posts
                AppState.selectedCowIndex = 0;
                AppState.selectedPostIndices = null;
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

    // Post section — always show if any posts
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

    html += '<div class="text-center">' +
        '<button class="btn btn-analyze" id="btnConfirmSelection" style="padding:10px 28px;">' +
        '<i class="fas fa-check"></i> Confirmar y Analizar</button></div>';

    $('#selectionPanel').html(html);

    // Bind confirm button
    $('#btnConfirmSelection').off('click').on('click', function() {
        // Read cow selection
        var cowVal = $('input[name="cowSelect"]:checked').val();
        AppState.selectedCowIndex = cowVal !== undefined ? parseInt(cowVal) : 0;

        // Read post selection
        var checkedPosts = [];
        $('input[name="postSelect"]:checked').each(function() {
            checkedPosts.push(parseInt($(this).val()));
        });

        if (scanData.posts.length > 0 && checkedPosts.length === 0) {
            alert('Necesitas al menos 1 poste seleccionado.');
            return;
        }

        // Siempre enviar los postes seleccionados explícitamente
        if (scanData.posts.length === 0) {
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

    // Postes detected + individual heights
    var postesClass = d.postes_detected >= 2 ? 'detail-success' : (d.postes_detected >= 1 ? 'detail-warning' : 'detail-danger');
    var postesValue = '' + (d.postes_detected || 0);
    if (d.postes_heights_px && d.postes_heights_px.length > 0) {
        var heightsStr = d.postes_heights_px.map(function(h) { return h.toFixed(0) + 'px'; }).join(', ');
        postesValue += ' (' + heightsStr + ')';
    }
    detailsHtml += '<div class="detail-item ' + postesClass + '">' +
        '<div class="label">Postes Detectados</div>' +
        '<div class="value">' + postesValue + '</div></div>';

    // Average post height (when 2+ posts)
    if (d.postes_heights_px && d.postes_heights_px.length >= 2) {
        var avgH = d.postes_heights_px.reduce(function(a, b) { return a + b; }, 0) / d.postes_heights_px.length;
        detailsHtml += '<div class="detail-item detail-success">' +
            '<div class="label">Promedio Postes</div>' +
            '<div class="value">' + avgH.toFixed(1) + ' px = 50cm</div></div>';
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

    $('#analysisDetails').html(detailsHtml);

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
    // Update batch screening section visibility/state
    updateScreeningSection();

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


// ── Batch Screening ──

function updateScreeningSection() {
    // Show batch screening section whenever a video is loaded
    if (AppState.videoFile) {
        $('#batchScreeningSection').show();
    } else {
        $('#batchScreeningSection').hide();
        return;
    }

    // Enable button only when calibration has cm_per_px
    var hasCalibration = AppState.calibrationFrame &&
        AppState.calibrationFrame.data &&
        AppState.calibrationFrame.data.details &&
        AppState.calibrationFrame.data.details.cm_per_px;

    $('#btnScreenVideo').prop('disabled', !hasCalibration);
    $('#btnModelo3DHibrido').prop('disabled', !hasCalibration);
    $('#btnModelo3DSfm').prop('disabled', !hasCalibration);
    if (hasCalibration) {
        $('#screeningHint').text('Escala calibrada (' +
            AppState.calibrationFrame.data.details.cm_per_px.toFixed(5) +
            ' cm/px). Listo para screening.');
        $('#modelo3dOptions').show();
    } else {
        $('#screeningHint').text('Calibra primero (2 postes) para habilitar el screening automatico.');
        $('#modelo3dOptions').hide();
    }
}

function startBatchScreening() {
    if (AppState.screeningActive) return;

    var calib = AppState.calibrationFrame;
    if (!calib || !calib.data || !calib.data.details || !calib.data.details.cm_per_px) {
        alert('Necesitas calibrar primero (2 postes) para obtener la escala cm/px.');
        return;
    }

    var cm_per_px = calib.data.details.cm_per_px;
    var frameInterval = parseInt($('#screenInterval').val()) || 30;

    // Confirm if video seems very long
    var totalFrames = getTotalFrames();
    var framesToProcess = Math.ceil(totalFrames / frameInterval);
    if (framesToProcess > 5000) {
        if (!confirm('Este video tiene ~' + framesToProcess + ' frames a procesar. Puede tomar mucho tiempo. Continuar?')) {
            return;
        }
    }

    AppState.screeningActive = true;
    AppState.screeningResults = [];
    AppState.screeningAbortController = new AbortController();

    // Show screening card and progress
    $('#screeningCard').fadeIn(300);
    $('#screeningProgress').show();
    $('#screeningSummary').hide().empty();
    $('#screeningGallery').hide().empty();
    $('#btnCancelScreening').show();
    $('#screeningProgressBar').css('width', '0%');
    $('#screeningProgressText').text('Iniciando screening...');
    $('#screeningProgressCount').text('0 / ?');

    // Disable screening button during processing
    $('#btnScreenVideo').prop('disabled', true);

    var minCowScore = parseFloat($('#screenMinScore').val()) || 0.75;

    // Build FormData with the actual video file
    var formData = new FormData();
    formData.append('video', AppState.videoFile);
    formData.append('cm_per_px', cm_per_px);
    formData.append('frame_interval', frameInterval);
    formData.append('min_cow_score', minCowScore);
    formData.append('breed', $('#breed').val() || 'desconocido');
    formData.append('category', $('#category').val() || 'desconocido');
    formData.append('age_range', $('#age_range').val() || 'desconocido');

    // Pass the same post indices used during calibration
    if (calib.postIndices !== null && calib.postIndices !== undefined) {
        formData.append('post_indices', calib.postIndices.join(','));
    }

    fetch('/batch_screen', {
        method: 'POST',
        body: formData,
        signal: AppState.screeningAbortController.signal
    }).then(function(response) {
        if (!response.ok) {
            throw new Error('Server error: ' + response.status);
        }

        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';

        function processChunk() {
            return reader.read().then(function(result) {
                if (result.done) {
                    // Stream ended without complete event
                    if (AppState.screeningActive) {
                        finishScreening(null);
                    }
                    return;
                }

                buffer += decoder.decode(result.value, {stream: true});

                // Parse SSE events from buffer using \n\n as event delimiter
                // (safe for large data lines like base64 thumbnails that span chunks)
                var parts = buffer.split('\n\n');
                // Last element may be incomplete — keep in buffer
                buffer = parts.pop() || '';

                for (var i = 0; i < parts.length; i++) {
                    var eventStr = parts[i].trim();
                    if (!eventStr) continue;

                    var eventType = '';
                    var eventData = '';
                    var eventLines = eventStr.split('\n');
                    for (var j = 0; j < eventLines.length; j++) {
                        var line = eventLines[j];
                        if (line.indexOf('event: ') === 0) {
                            eventType = line.substring(7);
                        } else if (line.indexOf('data: ') === 0) {
                            eventData = line.substring(6);
                        }
                    }

                    if (eventType && eventData) {
                        handleScreeningEvent(eventType, eventData);
                    }
                }

                if (AppState.screeningActive) {
                    return processChunk();
                }
            });
        }

        return processChunk();
    }).catch(function(err) {
        if (err.name === 'AbortError') {
            finishScreening(null, true);
        } else {
            console.error('[SCREENING] Error:', err);
            finishScreening(null, false, err.message);
        }
    });

    // Scroll to screening card
    $('html, body').animate({
        scrollTop: $('#screeningCard').offset().top - 20
    }, 500);
}

function handleScreeningEvent(eventType, dataStr) {
    try {
        var data = JSON.parse(dataStr);
    } catch (e) {
        console.warn('[SCREENING] Failed to parse event data:', dataStr);
        return;
    }

    switch (eventType) {
        case 'started':
            $('#screeningProgressText').text('Procesando frames...');
            $('#screeningProgressCount').text('0 / ' + data.frames_to_process);
            break;

        case 'frame_result':
            AppState.screeningResults.push(data);
            updateScreeningProgress(data.processed, data.total);
            break;

        case 'frame_skip':
            updateScreeningProgress(data.processed, data.total);
            break;

        case 'complete':
            finishScreening(data.summary);
            break;

        case 'error':
            finishScreening(null, false, data.message);
            break;
    }
}

function updateScreeningProgress(processed, total) {
    var pct = total > 0 ? Math.round(processed / total * 100) : 0;
    $('#screeningProgressBar').css('width', pct + '%');
    $('#screeningProgressCount').text(processed + ' / ' + total);
    var detected = AppState.screeningResults.length;
    var skipped = processed - detected;
    $('#screeningProgressText').text(
        'Procesando... ' + detected + ' detectados, ' + skipped + ' descartados'
    );
}

function finishScreening(summary, cancelled, errorMsg) {
    AppState.screeningActive = false;
    $('#btnCancelScreening').hide();
    updateScreeningSection();

    if (cancelled) {
        $('#screeningProgressText').text('Screening cancelado.');
        // Still render partial results
        if (AppState.screeningResults.length > 0) {
            renderScreeningResults(null);
        }
        return;
    }

    if (errorMsg) {
        $('#screeningProgress').hide();
        $('#screeningSummary').show().html(
            '<div class="screening-error-banner">' +
            '<i class="fas fa-exclamation-triangle"></i> Error: ' + errorMsg +
            '</div>'
        );
        if (AppState.screeningResults.length > 0) {
            renderScreeningResults(null);
        }
        return;
    }

    $('#screeningProgress').hide();
    renderScreeningResults(summary);
}

function renderScreeningResults(summary) {
    var results = AppState.screeningResults;

    if (results.length === 0) {
        $('#screeningSummary').show().html(
            '<div class="text-center py-3" style="color: var(--text-muted);">' +
            '<i class="fas fa-search" style="font-size:2em; display:block; margin-bottom:8px;"></i>' +
            'No se encontraron keypoints en ningun frame del video.' +
            '</div>'
        );
        $('#screeningGallery').hide();
        return;
    }

    // Separate valid / outlier
    var validResults = results.filter(function(r) { return r.in_range; });
    var outlierResults = results.filter(function(r) { return !r.in_range; });

    // Get calibration cm_per_px for converting px to cm
    var calibCmPerPx = null;
    if (AppState.calibrationFrame && AppState.calibrationFrame.data && AppState.calibrationFrame.data.details) {
        calibCmPerPx = AppState.calibrationFrame.data.details.cm_per_px;
    }

    // Build summary
    var summaryHtml = '<div class="screening-summary">';

    if (summary) {
        summaryHtml += buildStatCard('PROMEDIO', summary.avg_weight ? summary.avg_weight.toFixed(1) + ' kg' : '-', true);
        summaryHtml += buildStatCard('MEDIANA', summary.median_weight ? summary.median_weight.toFixed(1) + ' kg' : '-');
        summaryHtml += buildStatCard('DESV. STD', summary.std_dev !== undefined ? summary.std_dev.toFixed(1) + ' kg' : '-');
        summaryHtml += buildStatCard('VALIDOS', summary.valid_count + ' / ' + summary.detected_count);
        summaryHtml += buildStatCard('OUTLIERS', '' + summary.outlier_count);
        summaryHtml += buildStatCard('RANGO', summary.min_weight && summary.max_weight
            ? summary.min_weight.toFixed(0) + '-' + summary.max_weight.toFixed(0) + ' kg' : '-');
    } else {
        // Compute from local results
        var weights = validResults.length > 0
            ? validResults.map(function(r) { return r.weight_kg; })
            : results.map(function(r) { return r.weight_kg; });
        var avg = weights.reduce(function(a, b) { return a + b; }, 0) / weights.length;

        summaryHtml += buildStatCard('PROMEDIO', avg.toFixed(1) + ' kg', true);
        summaryHtml += buildStatCard('FRAMES', '' + results.length);
        summaryHtml += buildStatCard('VALIDOS', '' + validResults.length);
        summaryHtml += buildStatCard('OUTLIERS', '' + outlierResults.length);
    }

    summaryHtml += '</div>';

    if (outlierResults.length > 0 && validResults.length === 0) {
        summaryHtml += '<div class="screening-error-banner">' +
            '<i class="fas fa-exclamation-triangle"></i> Todos los frames tienen peso fuera del rango esperado.' +
            '</div>';
    }

    $('#screeningSummary').html(summaryHtml).show();

    // ── Results Table (like manual analysis) ──
    var tableResults = validResults.length > 0 ? validResults : results;
    AppState.screeningTableResults = tableResults.slice();
    AppState.screeningSortCol = null;
    AppState.screeningSortAsc = true;
    var galleryHtml = '';

    galleryHtml += '<h6 class="mt-3 mb-2" style="font-weight:600; color:var(--text-dark);">' +
        '<i class="fas fa-table"></i> Detalle por Frame (' + tableResults.length + ' validos)</h6>';

    galleryHtml += '<div style="overflow-x:auto;">';
    galleryHtml += '<table class="results-table" id="screeningDetailTable">' +
        '<thead><tr>' +
        buildSortableTh('Frame', 'frame_num') +
        '<th>Score</th>' +
        buildSortableTh('Altura (px)', 'animal_bbox_height_px') +
        '<th>cm/px</th>' +
        buildSortableTh('BL (px)', 'dist1_px') +
        '<th>BL (cm)</th>' +
        buildSortableTh('Girth (px)', 'dist2_px') +
        '<th>Girth (cm)</th>' +
        buildSortableTh('Peso (kg)', 'weight_kg') +
        '<th>Rango</th>' +
        '</tr></thead><tbody id="screeningDetailBody">';

    galleryHtml += buildScreeningTableBody(tableResults, calibCmPerPx);
    galleryHtml += '</tbody></table></div>';

    // Legend
    galleryHtml += '<div class="results-legend">' +
        '<span class="legend-badge legend-max">Rojo</span> = Peso maximo | ' +
        '<span class="legend-badge legend-min">Verde</span> = Peso minimo | ' +
        'Formula: (BL_cm x Girth_cm&sup2; x 0.4536) / 300 x K' +
        '</div>';

    // ── Gallery of annotated frames ──
    galleryHtml += '<h6 class="mt-4 mb-2" style="font-weight:600; color:var(--text-dark);">' +
        '<i class="fas fa-images"></i> Galeria de Frames (' + validResults.length + ' validos)</h6>';

    galleryHtml += '<div class="screening-gallery">';
    validResults.forEach(function(r) {
        galleryHtml += buildFrameCard(r, true);
    });
    galleryHtml += '</div>';

    // Outlier section
    if (outlierResults.length > 0) {
        galleryHtml += '<div class="screening-outlier-section">' +
            '<button class="screening-outlier-toggle" onclick="$(\'#screeningOutliers\').toggle();">' +
            '<i class="fas fa-exclamation-circle"></i> ' + outlierResults.length +
            ' frame' + (outlierResults.length !== 1 ? 's' : '') + ' fuera de rango (click para ver)</button>' +
            '<div id="screeningOutliers" style="display:none;">' +
            '<div class="screening-gallery" style="margin-top:10px;">';
        outlierResults.forEach(function(r) {
            galleryHtml += buildFrameCard(r, false);
        });
        galleryHtml += '</div></div></div>';
    }

    // Export button
    galleryHtml += '<div class="text-center">' +
        '<button class="btn btn-export-csv" onclick="exportScreeningCSV()">' +
        '<i class="fas fa-download"></i> Exportar CSV</button></div>';

    $('#screeningGallery').html(galleryHtml).show();
}

function buildStatCard(label, value, primary) {
    var cls = primary ? 'stat-card stat-primary' : 'stat-card';
    return '<div class="' + cls + '">' +
        '<div class="stat-label">' + label + '</div>' +
        '<div class="stat-value">' + value + '</div></div>';
}

function buildSortableTh(label, field) {
    var isActive = AppState.screeningSortCol === field;
    var arrow = isActive ? (AppState.screeningSortAsc ? '▲' : '▼') : '↕';
    var activeClass = isActive ? ' sort-active' : '';
    return '<th class="sortable-th' + activeClass + '" onclick="sortScreeningTable(\'' + field + '\')" ' +
        'title="Ordenar por ' + label + '">' +
        label + '<span class="sort-arrow">' + arrow + '</span></th>';
}

function buildScreeningTableBody(rows, calibCmPerPx) {
    var html = '';
    var allWeights = rows.map(function(r) { return r.weight_kg; });
    var tblMin = allWeights.length > 0 ? Math.min.apply(null, allWeights) : 0;
    var tblMax = allWeights.length > 0 ? Math.max.apply(null, allWeights) : 0;

    rows.forEach(function(r) {
        var useCmPx = r.cm_per_px || calibCmPerPx || 0;
        var dist1Cm = r.dist1_px ? (r.dist1_px * useCmPx) : 0;
        var dist2Cm = r.dist2_px ? (r.dist2_px * useCmPx) : 0;
        var rowClass = '';
        if (allWeights.length > 1) {
            if (Math.abs(r.weight_kg - tblMax) < 0.01) rowClass = 'row-max';
            else if (Math.abs(r.weight_kg - tblMin) < 0.01) rowClass = 'row-min';
        }

        html += '<tr class="' + rowClass + '">' +
            '<td>' + r.frame_num + '</td>' +
            '<td>' + (r.cow_score ? (r.cow_score * 100).toFixed(0) + '%' : '-') + '</td>' +
            '<td>' + (r.animal_bbox_height_px ? r.animal_bbox_height_px.toFixed(0) : '-') + '</td>' +
            '<td>' + (useCmPx ? useCmPx.toFixed(5) : '-') + '</td>' +
            '<td>' + (r.dist1_px ? r.dist1_px.toFixed(1) : '-') + '</td>' +
            '<td>' + (dist1Cm ? dist1Cm.toFixed(2) : '-') + '</td>' +
            '<td>' + (r.dist2_px ? r.dist2_px.toFixed(1) : '-') + '</td>' +
            '<td>' + (dist2Cm ? dist2Cm.toFixed(2) : '-') + '</td>' +
            '<td><strong>' + r.weight_kg.toFixed(2) + '</strong></td>' +
            '<td>' + (r.in_range ? '<span style="color:var(--success);">OK</span>' : '<span style="color:var(--danger);">Fuera</span>') + '</td>' +
            '</tr>';
    });

    // Average row
    if (rows.length > 0) {
        var avgW = allWeights.reduce(function(a, b) { return a + b; }, 0) / allWeights.length;
        var avgD1 = rows.reduce(function(a, r) { return a + (r.dist1_px || 0); }, 0) / rows.length;
        var avgD2 = rows.reduce(function(a, r) { return a + (r.dist2_px || 0); }, 0) / rows.length;
        var useCmPxAvg = calibCmPerPx || (rows[0].cm_per_px || 0);
        html += '<tr class="row-average">' +
            '<td>PROM.</td>' +
            '<td>-</td><td>-</td><td>-</td>' +
            '<td>' + avgD1.toFixed(1) + '</td>' +
            '<td>' + (avgD1 * useCmPxAvg).toFixed(2) + '</td>' +
            '<td>' + avgD2.toFixed(1) + '</td>' +
            '<td>' + (avgD2 * useCmPxAvg).toFixed(2) + '</td>' +
            '<td><strong>' + avgW.toFixed(2) + '</strong></td>' +
            '<td>-</td></tr>';
    }

    return html;
}

function sortScreeningTable(field) {
    if (AppState.screeningSortCol === field) {
        AppState.screeningSortAsc = !AppState.screeningSortAsc;
    } else {
        AppState.screeningSortCol = field;
        AppState.screeningSortAsc = true;
    }

    var asc = AppState.screeningSortAsc;
    AppState.screeningTableResults.sort(function(a, b) {
        var va = a[field] || 0;
        var vb = b[field] || 0;
        return asc ? (va - vb) : (vb - va);
    });

    // Get calibration cm_per_px
    var calibCmPerPx = null;
    if (AppState.calibrationFrame && AppState.calibrationFrame.data && AppState.calibrationFrame.data.details) {
        calibCmPerPx = AppState.calibrationFrame.data.details.cm_per_px;
    }

    // Rebuild table body
    $('#screeningDetailBody').html(buildScreeningTableBody(AppState.screeningTableResults, calibCmPerPx));

    // Update sort arrows and active class in headers
    $('#screeningDetailTable thead .sortable-th').each(function() {
        var onclick = $(this).attr('onclick') || '';
        var isThis = onclick.indexOf("'" + field + "'") !== -1;
        $(this).toggleClass('sort-active', isThis);
        $(this).find('.sort-arrow').text(isThis ? (asc ? '▲' : '▼') : '↕');
    });
}

function buildFrameCard(result, inRange) {
    var cardClass = inRange ? 'in-range' : 'out-of-range';
    var badgeClass = inRange ? 'badge-ok' : 'badge-outlier';
    var thumbSrc = result.annotated_thumb
        ? 'data:image/jpeg;base64,' + result.annotated_thumb
        : '';

    var html = '<div class="screening-frame-card ' + cardClass + '"';
    if (thumbSrc) {
        html += ' onclick="showScreeningFullImage(this)"';
    }
    html += '>';

    if (thumbSrc) {
        html += '<img src="' + thumbSrc + '" class="screening-frame-thumb" alt="Frame ' + result.frame_num + '">';
    }

    html += '<span class="screening-frame-label">F' + result.frame_num + '</span>';
    html += '<span class="screening-weight-badge ' + badgeClass + '">' +
        result.weight_kg.toFixed(1) + ' kg</span>';

    html += '<div class="screening-frame-info">';
    if (result.cow_score) html += '<strong>' + (result.cow_score * 100).toFixed(0) + '%</strong> | ';
    if (result.dist1_px) html += 'BL=' + result.dist1_px.toFixed(0) + 'px ';
    if (result.dist2_px) html += 'G=' + result.dist2_px.toFixed(0) + 'px';
    html += '</div>';

    html += '</div>';
    return html;
}

function showScreeningFullImage(cardEl) {
    var img = $(cardEl).find('.screening-frame-thumb');
    if (!img.length) return;

    var overlay = $('<div class="screening-modal-overlay"></div>');
    var fullImg = $('<img>').attr('src', img.attr('src'));
    overlay.append(fullImg);
    overlay.on('click', function() { overlay.remove(); });
    $('body').append(overlay);
}

function exportScreeningCSV() {
    var results = AppState.screeningResults;
    if (results.length === 0) return;

    var csv = 'frame_num,cow_score,weight_kg,in_range,animal_height_px,cm_per_px,dist1_px,dist2_px\n';
    results.forEach(function(r) {
        csv += r.frame_num + ',' +
            (r.cow_score || '') + ',' +
            r.weight_kg + ',' +
            (r.in_range ? 'si' : 'no') + ',' +
            (r.animal_bbox_height_px || '') + ',' +
            (r.cm_per_px || '') + ',' +
            (r.dist1_px || '') + ',' +
            (r.dist2_px || '') + '\n';
    });

    var blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'screening_' + new Date().toISOString().slice(0, 10) + '.csv';
    a.click();
    URL.revokeObjectURL(url);
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
    updateScreeningSection();

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

    // Peso barril: show for both hybrid and multi-frame modes
    if (summary.peso_barril_kg) {
        $('#m3dPesoBarril').text(summary.peso_barril_kg + ' kg');
        $('#m3dPesoBarrilCol').show();
    } else {
        $('#m3dPesoBarrilCol').hide();
    }

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

function cancelScreening() {
    if (AppState.screeningAbortController) {
        AppState.screeningAbortController.abort();
    }
}

// ====================================================================
// Document Ready - Event Handlers
// ====================================================================

$(document).ready(function() {

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

    // ── Video Controls ──

    $('#btnSkipBack').on('click', function() { seekFrame(-10); });
    $('#btnPrev').on('click', function() { seekFrame(-1); });
    $('#btnPlayPause').on('click', function() { togglePlayPause(); });
    $('#btnNext').on('click', function() { seekFrame(1); });
    $('#btnSkipForward').on('click', function() { seekFrame(10); });

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

    // ── Batch Screening ──

    $('#btnScreenVideo').on('click', function() {
        startBatchScreening();
    });

    $('#btnCancelScreening').on('click', function() {
        cancelScreening();
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
