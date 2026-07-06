// ===== generate3DFromResult (botón Generar modelo 3D) =====
async function generate3DFromResult() {
    var cowName = ($('#resultCowName').val() || '').trim();
    if (!cowName) {
        alert('Poné un nombre para la vaca.');
        return;
    }
    if (!AppState.passingResults.length) {
        alert('No hay resultados.');
        return;
    }

    // Contornos del barril (galería del barril) sin outliers (ancho ≥30%
    // por debajo de la media). El modelo 3D se arma exclusivamente desde
    // la fusión multi-frame de inliers.
    var contours = [];
    AppState.passingResults.forEach(function(r) {
        if (r.cow_height_cm) return;
        if (r.barril_outlier) return;
        var c = r.barril_contour_norm;
        if (!c || !c.heights_cm || !c.width_cm) return;
        if (c.heights_cm.length !== c.n_samples) return;
        contours.push({
            n_samples: c.n_samples,
            width_cm: c.width_cm,
            heights_cm: c.heights_cm,
            tops_cm: c.tops_cm,
            bottoms_cm: c.bottoms_cm,
        });
    });
    if (contours.length < 2) {
        alert('Se necesitan al menos 2 frames del barril sin poste solapando y con ancho dentro de tolerancia para generar el modelo 3D. ' +
              'Hoy hay ' + contours.length + '.');
        return;
    }

    // Frame piloto para la malla VISUAL (silueta + textura): tiene que ser
    // un frame del BARRIL (sin poste solapando, cuerpo limpio), porque si
    // tomamos un frame de altura el poste rojo aparece en la textura del
    // modelo 3D. Elegimos el más cercano al ancho del consenso así la malla
    // refleja al animal promedio. Si no hay consenso, cae al primer válido.
    var target = null;
    var bestDiff = Infinity;
    var refWidth = (AppState.lastConsensusContour && AppState.lastConsensusContour.width_cm) || 0;
    AppState.passingResults.forEach(function(r) {
        if (r.cow_height_cm) return;            // skip frames de altura
        if (!r.barril_contour_norm || !r.barril_contour_norm.width_cm) return;
        if (!r.cm_per_px) return;
        if (refWidth > 0) {
            var diff = Math.abs(r.barril_contour_norm.width_cm - refWidth);
            if (diff < bestDiff) { bestDiff = diff; target = r; }
        } else if (!target) {
            target = r;
        }
    });
    if (!target) {
        alert('No hay frames del barril (sin poste solapando) para construir la malla visual.');
        return;
    }

    var $btn = $('#btnGenerate3D').prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Generando…');

    try {
        // ── Paso 1: visual 3D desde frame representativo (silueta + colores) ──
        // Escribe _3d.ply y _lateral.ply con forma de vaca y textura de la foto.
        // El volumen reportado es el volumen encerrado de la malla _3d.ply.
        var video = document.getElementById('videoPlayer');
        await _seekVideo(video, target.frameNum / AppState.fps);
        var blob = await captureVideoFrameBlob(video);

        var fd = new FormData();
        fd.append('frame', blob, '3d_frame.jpg');
        fd.append('cow_name', cowName);
        fd.append('altura_cm', (AppState.lastAvgH || 0).toFixed(1));
        // Largo promedio del box del barril (cm), medido en la misma pasada.
        fd.append('largo_cm', (AppState.lastAvgLargo || 0).toFixed(1));
        // Dirección de marcha = hacia dónde mira el barril (las manos van al
        // frente, en el sentido del movimiento). Sirve para ubicar la sección
        // del diámetro "detrás de las manos".
        var _dir = _detectMovementDir(AppState.passingResults || []);
        var barrilDir = _dir.valid ? (_dir.dx > 0 ? 'right' : 'left') : 'unknown';
        AppState.lastBarrilDir = barrilDir;
        fd.append('barril_dir', barrilDir);
        // Pasamos el consenso como barril_L de referencia (lo sobrescribe
        // igual el paso 2, pero mantiene coherencia si el paso 2 fallara).
        fd.append('barril_L', (AppState.lastConsensusB || AppState.lastAvgB || 0).toFixed(1));
        if (AppState.videoId) fd.append('video_id', AppState.videoId);
        if (AppState.lockedReference) {
            fd.append('locked_reference_json', JSON.stringify({
                post1: AppState.lockedReference.post1,
                post2: AppState.lockedReference.post2,
                original_coords: AppState.lockedReference.original_coords,
            }));
        }

        var respFrame = await fetch('/generate_3d_from_frame', { method: 'POST', body: fd });
        var dataFrame = await respFrame.json();
        if (!dataFrame.success) {
            alert('Error generando visual 3D: ' + (dataFrame.error || 'desconocido'));
            return;
        }

        // ── Paso 2: finalizar resumen ──
        // Reporta vol_barril_litros = volumen encerrado del _3d.ply ya escrito
        // y guarda metadatos del consenso (frames usados, ancho). NO toca PLYs.
        var respCons = await fetch('/generate_3d_consensus', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                cow_name: cowName,
                altura_cm: AppState.lastAvgH || 0,
                largo_cm: AppState.lastAvgLargo || 0,
                barril_dir: barrilDir,
                contours: contours,
            }),
        });
        var dataCons = await respCons.json();
        if (!dataCons.success) {
            alert('Error calculando consenso: ' + (dataCons.error || 'desconocido'));
            return;
        }
        if (dataCons.barril_consenso_L != null) {
            AppState.lastConsensusB = dataCons.barril_consenso_L;
        }

        if (typeof window.loadModelosDisponibles === 'function') {
            window.loadModelosDisponibles(dataCons.model_id || cowName);
        }
        var $viewer = $('#viewer3dCard');
        if ($viewer.length) {
            $('html, body').animate({ scrollTop: $viewer.offset().top - 20 }, 400);
        }
    } catch (err) {
        alert('Error: ' + err);
    } finally {
        $btn.prop('disabled', false).html('<i class="fas fa-cube"></i> Generar modelo 3D');
    }
}



// ===== variantes B/C/E del consenso =====
        var resB = _computeBarrilConsensus(sourceContours, 'p75');
        var resC = _computeBarrilConsensus(sourceContours, 'maxW_p75');
        var resE = _computeBarrilConsensus(sourceContours, 'envelope');
        AppState.consensusVariants = {
            A_median: result ? result.volumeL : null,
            B_p75: resB ? resB.volumeL : null,
            C_maxW_p75: resC ? resC.volumeL : null,
            E_envelope: resE ? resE.volumeL : null,

// ===== processFolder + renderProcessedFrameThumb (Procesar carpeta → 3D) =====
async function processFolder() {
    var folder = prompt('Nombre de la carpeta en checkpoints/6mayo/ (= nombre del video sin extensión):');
    if (!folder) return;
    // El nombre del individuo = nombre de la carpeta (= nombre del video).
    var cowName = folder.replace(/[^A-Za-z0-9_-]/g, '_');
    if (!cowName) { alert('Nombre de carpeta inválido para usar como individuo.'); return; }
    var alturaStr = prompt('Altura del animal (cm):', '92.5');
    if (!alturaStr) return;
    var altura = parseFloat(alturaStr);
    if (isNaN(altura) || altura <= 0) { alert('Altura inválida.'); return; }

    var $btn = $('#btnProcessFolder');
    var $lbl = $('#btnProcessFolderLabel');
    var orig = $lbl.text();
    $btn.prop('disabled', true);

    try {
        // 1. Listar archivos + cargar context.json (locked_reference) de la carpeta
        var fd0 = new FormData();
        fd0.append('folder', folder);
        var resp = await fetch('/list_saved_frames', { method: 'POST', body: fd0 });
        var data = await resp.json();
        if (!data.success) { alert('Error: ' + data.error); return; }
        var frames = data.frames || [];
        if (!frames.length) { alert('Carpeta vacía.'); return; }

        // Resolver locked_reference: priorizar la del context.json (= la que
        // estaba activa al guardar los frames). Si no hay, usar la actual.
        var lockedRef = (data.context && data.context.locked_reference) || AppState.lockedReference;
        if (!lockedRef) {
            alert('No hay locked_reference: ni en el context.json de la carpeta ni en la sesión actual. Cargá un video y marcá los postes, o regenerá la carpeta con el botón "Guardar 21 frames" después de marcar postes.');
            return;
        }
        console.log('[processFolder]', frames.length, 'frames; locked_reference desde:', data.context && data.context.locked_reference ? 'context.json' : 'sesión actual');

        // 2. Reset state como si arrancara una nueva pasada
        AppState.passingResults = [];
        AppState.passingStats = { analyzed: 0, detected: 0, in_rect: 0, out_rect: 0, no_cow: 0 };
        $('#screeningGallery').empty();

        // 3. Procesar cada frame con /detect_cow_fast
        var refJson = JSON.stringify({
            post1: lockedRef.post1, post2: lockedRef.post2,
            original_coords: lockedRef.original_coords,
        });
        for (var i = 0; i < frames.length; i++) {
            var fr = frames[i];
            $lbl.text('Procesando ' + (i + 1) + '/' + frames.length);
            var imgResp = await fetch('/saved_frame/' + folder + '/' + fr.file_name);
            var blob = await imgResp.blob();
            var fd = new FormData();
            fd.append('frame', blob, fr.file_name);
            if (AppState.videoId) fd.append('video_id', AppState.videoId);
            fd.append('locked_reference_json', refJson);
            var detResp = await fetch('/detect_cow_fast', { method: 'POST', body: fd });
            var d = await detResp.json();
            AppState.passingStats.analyzed++;
            if (d && d.success && d.detected) {
                AppState.passingStats.detected++;
                var thumbUrl = await renderProcessedFrameThumb(blob, d, fr.offset);
                AppState.passingResults.push({
                    frameNum: fr.frame_num,
                    valid: !!d.within_rectangle,
                    within_rectangle: !!d.within_rectangle,
                    cow_height_cm: d.cow_height_cm,
                    cm_per_px: d.cm_per_px,
                    animal_bbox_original: d.animal_bbox_original,
                    video_w: d.video_w, video_h: d.video_h,
                    p: d.p, t_floor: d.t_floor,
                    silueta_bottom_used: !!d.silueta_bottom_used,
                    barril_top_used: !!d.barril_top_used,
                    barril_post_overlap: !!d.barril_post_overlap,
                    barril_volumen_litros: d.barril_volumen_litros,
                    barril_contour_norm: d.barril_contour_norm,
                    barril_cols_rellenadas: d.barril_cols_rellenadas,
                    bbox_aligned_with_floor: !!d.bbox_aligned_with_floor,
                    annotated_image: thumbUrl,
                    folder_offset: fr.offset,
                });
                // Marcar el frame con offset=0 como "central" (cruce simulado)
                if (fr.offset === 0 && d.cow_height_cm == null) {
                    // Si el central no tiene altura, le asignamos la dada por el usuario
                    AppState.passingResults[AppState.passingResults.length - 1].cow_height_cm = altura;
                    AppState.passingResults[AppState.passingResults.length - 1].counted_in_avg = true;
                }
            } else {
                AppState.passingStats.no_cow++;
            }
        }

        // 4. Finalizar (asigna passing_idx basado en folder_offset)
        $lbl.text('Calculando consenso...');
        AppState.passingResults.sort(function(a, b) { return a.frameNum - b.frameNum; });
        AppState.passingResults.forEach(function(r) {
            r.passing_idx = (r.folder_offset != null) ? r.folder_offset : null;
            r.barril_eligible = true;  // todos los 21 son elegibles por construcción
        });

        // 5. Generar modelo 3D consenso usando el frame central + altura dada
        var centralR = AppState.passingResults.find(function(r) { return r.folder_offset === 0; });
        if (!centralR) { alert('No se encontró frame central (offset=0).'); return; }

        // Re-fetch del blob central para mandarlo a /generate_3d_from_frame
        var centralFile = frames.find(function(f) { return f.offset === 0; });
        var centralBlob = await (await fetch('/saved_frame/' + folder + '/' + centralFile.file_name)).blob();

        // 6. Computar consenso (4 variantes)
        // Reusa la lógica de finalizePassingResults para obtener la silueta
        finalizePassingResults();

        // 7. Generar PLYs desde el frame central
        $lbl.text('Generando PLYs 3D...');
        var fd3d = new FormData();
        fd3d.append('frame', centralBlob, centralFile.file_name);
        fd3d.append('cow_name', cowName);
        fd3d.append('altura_cm', String(altura));
        fd3d.append('barril_L', String(AppState.consensusVariants && AppState.consensusVariants.E_envelope || 0));
        if (AppState.videoId) fd3d.append('video_id', AppState.videoId);
        fd3d.append('locked_reference_json', refJson);
        var resp3d = await fetch('/generate_3d_from_frame', { method: 'POST', body: fd3d });
        var d3d = await resp3d.json();
        if (!d3d.success) { alert('Error generando 3D: ' + d3d.error); return; }

        var cv = AppState.consensusVariants || {};
        alert('Modelo 3D generado: ' + cowName + '\n' +
              'Frames procesados: ' + AppState.passingResults.length + '\n' +
              'Barril: ' + (cv.A_median || '–').toFixed(1) + ' L\n' +
              'Carpeta: ' + folder);
    } catch (e) {
        console.error('[processFolder] err', e);
        alert('Error: ' + e.message);
    } finally {
        $btn.prop('disabled', false);
        $lbl.text(orig);
    }
}

async function renderProcessedFrameThumb(blob, data, offset) {
    var img = await createImageBitmap(blob);
    var THUMB_W = 520;
    var scale = THUMB_W / img.width;
    var W = Math.round(img.width * scale);
    var H = Math.round(img.height * scale);
    var canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, W, H);
    // Marca offset arriba a la izquierda
    ctx.fillStyle = offset === 0 ? '#1976d2' : '#37474f';
    ctx.fillRect(8, 8, 56, 22);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 14px sans-serif';
    var lbl = offset === 0 ? '0 cruce' : (offset > 0 ? '+' + offset : String(offset));
    ctx.fillText(lbl, 12, 24);
    return canvas.toDataURL('image/jpeg', 0.82);
}

