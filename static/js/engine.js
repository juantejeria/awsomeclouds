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
    analyzing: false
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
    $('#frameCounter').text('Frame: ' + current + ' / ' + total);
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
    $('#analysisCard').hide();
    $('#resultsCard').hide();
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

    // Show analysis card with loader
    $('#analysisCard').fadeIn(300);
    $('#analysisLoader').show();
    $('#analysisContent').hide();
    $('#analysisFrameLabel').text('Frame ' + frameNum);

    // Read current breed/category/age from dropdowns
    AppState.breed = $('#breed').val() || 'desconocido';
    AppState.category = $('#category').val() || 'desconocido';
    AppState.age_range = $('#age_range').val() || 'desconocido';

    captureFrame().then(function(blob) {
        var formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');
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
    }).catch(function(err) {
        AppState.analyzing = false;
        $('#analysisLoader').hide();
        alert('Error al capturar frame: ' + err);
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
            '<div class="value">' + avgH.toFixed(1) + ' px = 122cm</div></div>';
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
        data: AppState.currentAnalysis
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

    // ── Initialize UI ──

    updateSelectionSummary();
    updateCalculateButton();
});
