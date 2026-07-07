// Extraído verbatim de static/js/engine.js — flujo "Batch screening" (REMOVIDO 2026-07-06)
// Los contenedores #screeningCard/#screeningProgress/#screeningGallery/#screeningSummary
// NO están acá: siguen vivos, los usa el flujo "Detectar pasadas" (processPassingLoop).

// ===== Campos de AppState (engine.js ~54-60) =====
    // Batch screening
    screeningResults: [],
    screeningActive: false,
    screeningAbortController: null,
    screeningSortCol: null,
    screeningSortAsc: true,
    screeningTableResults: [],

// ===== Reset al cargar nuevo video (initVideoPlayer, ~1678-1688) =====
    AppState.screeningResults = [];
    AppState.screeningActive = false;
    if (AppState.screeningAbortController) {
        AppState.screeningAbortController.abort();
        AppState.screeningAbortController = null;
    }

// ===== Ocultar botón cancelar al iniciar barrido (startDetectPassings, ~338) =====
    $('#btnCancelScreening').hide();

// ===== Llamada desde updateSelectionSummary (~2287-2288) =====
    // Update batch screening section visibility/state
    updateScreeningSection();

// ===== Bloque principal (~2517-3041) =====
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

    $('#screeningGallery').html(galleryHtml).css('display', 'flex');
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

// ===== Llamada desde finishModelo3D (~3183) =====
    updateScreeningSection();

// ===== cancelScreening (~3265-3269) =====
function cancelScreening() {
    if (AppState.screeningAbortController) {
        AppState.screeningAbortController.abort();
    }
}

// ===== Bindings en $(document).ready (~3400-3408) =====
    // ── Batch Screening ──

    $('#btnScreenVideo').on('click', function() {
        startBatchScreening();
    });

    $('#btnCancelScreening').on('click', function() {
        cancelScreening();
    });
