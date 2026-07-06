// ===== Sección principal (mapa de trayectoria, línea de mínimos, regla Fase C) =====
// ── Mapa de trayectoria de pezuñas ──
// Acumula los feet_points detectados en cada frame del barrido y los pinta
// sobre un canvas único, color-codificados por índice de frame (azul→rojo)
// para visualizar la trayectoria temporal de los pies.
// Detecta dirección del movimiento comparando el centro X del bbox del primer
// frame con el del último frame que tienen bbox válido. > 0 → derecha, < 0 → izquierda.
function _segIntersect(A, B, C, D) {
    var d1x = B.x - A.x, d1y = B.y - A.y;
    var d2x = D.x - C.x, d2y = D.y - C.y;
    var denom = d1x * d2y - d1y * d2x;
    if (Math.abs(denom) < 1e-9) return null;
    var t = ((C.x - A.x) * d2y - (C.y - A.y) * d2x) / denom;
    var u = ((C.x - A.x) * d1y - (C.y - A.y) * d1x) / denom;
    if (t < 0 || t > 1 || u < 0 || u > 1) return null;
    return { x: A.x + t * d1x, y: A.y + t * d1y };
}

function _filterFeetByMode(r, mode, movingRight) {
    if (mode === 'all') return r.feet_points || [];
    var bb = r.animal_bbox_original;
    if (!bb || !Array.isArray(r.feet_points)) return [];
    var cx = (bb[0] + bb[2]) / 2;
    return r.feet_points.filter(function(fp) {
        if (mode === 'back') return movingRight ? (fp.x < cx) : (fp.x > cx);
        if (mode === 'front') return movingRight ? (fp.x > cx) : (fp.x < cx);
        return true;
    });
}

function renderFeetTrajectoryMap() {
    var results = AppState.passingResults || [];
    var mode = $('#feetFilterMode').val() || 'back';
    var dirInfo = _detectMovementDir(results);
    var movingRight = dirInfo.dx > 0;

    // Aplicar filtro según modo a una copia (no mutamos passingResults)
    var filtered = results.map(function(r) {
        var fp = (mode === 'all' || !dirInfo.valid)
            ? (r.feet_points || [])
            : _filterFeetByMode(r, mode, movingRight);
        return Object.assign({}, r, { _feet_filtered: fp });
    });

    var withFeet = filtered.filter(function(r) { return r._feet_filtered.length > 0; });
    var totalFeetAll = withFeet.reduce(function(a, r) { return a + r._feet_filtered.length; }, 0);
    console.log('[FEET MAP] frames=' + filtered.length + ' frames_with_feet=' + withFeet.length +
                ' total_points=' + totalFeetAll +
                ' mode=' + mode + ' dir_dx=' + dirInfo.dx.toFixed(0) +
                ' dir_valid=' + dirInfo.valid);

    // Frame de referencia: el del cruce si existe, si no el del medio
    var ref = null;
    for (var i = 0; i < results.length; i++) {
        if (results[i].cow_height_cm != null && results[i].annotated_image) {
            ref = results[i];
            break;
        }
    }
    if (!ref) {
        for (var j = 0; j < results.length; j++) {
            if (results[j].annotated_image) { ref = results[j]; break; }
        }
    }
    if (!ref) {
        $('#feetTrajectoryInfo').html('<em>No hay frames con vaca detectada en este barrido.</em>');
        $('#feetTrajectoryWrap').empty();
        $('#feetTrajectoryCard').show();
        return;
    }
    if (withFeet.length === 0) {
        $('#feetTrajectoryInfo').html(
            '<em>Ningún frame devolvió picos de pezuñas.</em> ' +
            '(silueta_seg no marcó pies o los umbrales de find_peaks son altos — revisar logs Flask)'
        );
        $('#feetTrajectoryWrap').empty();
        $('#feetTrajectoryCard').show();
        return;
    }

    // Canvas blanco grande, manteniendo el aspect del video original
    var srcW = ref.video_w || (results[0] && results[0].video_w);
    var srcH = ref.video_h || (results[0] && results[0].video_h);
    if (!srcW || !srcH) { $('#feetTrajectoryCard').hide(); return; }
    var MAP_W = 1100;
    var sc = MAP_W / srcW;
    var MAP_H = Math.round(srcH * sc);
    var canvas = document.createElement('canvas');
    canvas.width = MAP_W; canvas.height = MAP_H;
    var ctx = canvas.getContext('2d');
    // Fondo blanco
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, MAP_W, MAP_H);

    // Referencia fija: rectángulo de postes + cintas + línea del piso
    var oc = AppState.lockedReference && AppState.lockedReference.original_coords;
    if (oc) {
        var pL = oc.post1.cx < oc.post2.cx ? oc.post1 : oc.post2;
        var pR = oc.post1.cx < oc.post2.cx ? oc.post2 : oc.post1;
        // Top + laterales en amarillo apagado
        ctx.strokeStyle = 'rgba(204,164,0,0.9)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pL.cx*sc, pL.top_tape*sc); ctx.lineTo(pR.cx*sc, pR.top_tape*sc);
        ctx.moveTo(pL.cx*sc, pL.top_tape*sc); ctx.lineTo(pL.cx*sc, pL.floor*sc);
        ctx.moveTo(pR.cx*sc, pR.top_tape*sc); ctx.lineTo(pR.cx*sc, pR.floor*sc);
        ctx.stroke();
        // Línea del piso en azul
        ctx.strokeStyle = 'rgba(33,150,243,0.95)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pL.cx*sc, pL.floor*sc);
        ctx.lineTo(pR.cx*sc, pR.floor*sc);
        ctx.stroke();
        // Cintas rojas inclinadas (la franja de 110cm)
        ctx.strokeStyle = 'rgba(244,67,54,0.9)';
        ctx.lineWidth = 1;
        [pL, pR].forEach(function(p) {
            var tx = (p.top_tape_x !== undefined ? p.top_tape_x : p.cx) * sc;
            var ty = p.top_tape * sc;
            var bx = p.cx * sc;
            var by = p.floor * sc;
            ctx.beginPath();
            ctx.moveTo(tx, ty);
            ctx.lineTo(bx, by);
            ctx.stroke();
        });
        // Rotated rect en celeste cuando hay tilt
        [pL, pR].forEach(function(p) {
            if (!Array.isArray(p.rot_corners) || p.rot_corners.length !== 4) return;
            if (Math.abs(p.angle_deg || 0) < 0.5) return;
            ctx.strokeStyle = 'rgba(100,220,255,0.85)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            p.rot_corners.forEach(function(c, i) {
                var x = c[0] * sc, y = c[1] * sc;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.closePath();
            ctx.stroke();
        });
    }

    // Puntos de pezuñas color-codificados por orden temporal (filtrados según modo)
    var n = filtered.length;
    filtered.forEach(function(r, idx) {
        if (!r._feet_filtered || r._feet_filtered.length === 0) return;
        var t = n > 1 ? idx / (n - 1) : 0;
        var rC = Math.round(33 + t * (244 - 33));
        var gC = Math.round(150 + t * (67 - 150));
        var bC = Math.round(243 + t * (54 - 243));
        ctx.fillStyle = 'rgba(' + rC + ',' + gC + ',' + bC + ',0.95)';
        r._feet_filtered.forEach(function(fp) {
            ctx.beginPath();
            ctx.arc(fp.x * sc, fp.y * sc, 1.8, 0, Math.PI*2);
            ctx.fill();
        });
    });

    // ── Línea media (media móvil binned sobre los puntos filtrados) ──
    // Divide el rango de X en bins, computa la media (x, y) por bin, conecta.
    // No es recta: sigue la tendencia central del cloud de pezuñas.
    var allPts = [];
    filtered.forEach(function(r) {
        if (r._feet_filtered && r._feet_filtered.length > 0) {
            r._feet_filtered.forEach(function(fp) {
                allPts.push({ x: fp.x, y: fp.y });
            });
        }
    });
    if (allPts.length >= 4) {
        allPts.sort(function(a, b) { return a.x - b.x; });
        var minX = allPts[0].x;
        var maxX = allPts[allPts.length - 1].x;
        var rangeX = maxX - minX;
        if (rangeX > 0) {
            // Cantidad de bins escala con el ancho del rango y la cantidad de puntos
            var NUM_BINS = Math.max(6, Math.min(24, Math.floor(allPts.length / 2)));
            var binW = rangeX / NUM_BINS;
            var bins = [];
            for (var bi = 0; bi < NUM_BINS; bi++) bins.push({ sx: 0, sy: 0, n: 0 });
            allPts.forEach(function(p) {
                var idxB = Math.min(NUM_BINS - 1, Math.floor((p.x - minX) / binW));
                bins[idxB].sx += p.x;
                bins[idxB].sy += p.y;
                bins[idxB].n++;
            });
            var line = [];
            bins.forEach(function(b) {
                if (b.n > 0) line.push({ x: b.sx / b.n, y: b.sy / b.n });
            });
            if (line.length >= 2) {
                // Curva suave con quadratic curves (control = punto medio)
                ctx.strokeStyle = 'rgba(33, 33, 33, 0.85)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(line[0].x * sc, line[0].y * sc);
                for (var li = 1; li < line.length - 1; li++) {
                    var mx = (line[li].x + line[li + 1].x) / 2;
                    var my = (line[li].y + line[li + 1].y) / 2;
                    ctx.quadraticCurveTo(line[li].x * sc, line[li].y * sc, mx * sc, my * sc);
                }
                ctx.lineTo(line[line.length - 1].x * sc, line[line.length - 1].y * sc);
                ctx.stroke();
                // Pequeños puntos en cada media-bin para que se vea de qué pasa la curva
                ctx.fillStyle = 'rgba(33, 33, 33, 0.6)';
                line.forEach(function(p) {
                    ctx.beginPath();
                    ctx.arc(p.x * sc, p.y * sc, 1.5, 0, Math.PI * 2);
                    ctx.fill();
                });
            }

            // ── Línea de mínimos locales ──
            // En cada bin nos quedamos con el punto de mayor Y (más cerca del
            // piso = pezuña plantada). Después filtramos por prominencia para
            // descartar bins cuyo máximo no es claramente un valle local.
            var minBins = [];
            for (var mi = 0; mi < NUM_BINS; mi++) minBins.push(null);
            allPts.forEach(function(p) {
                var idxM = Math.min(NUM_BINS - 1, Math.floor((p.x - minX) / binW));
                if (minBins[idxM] === null || p.y > minBins[idxM].y) {
                    minBins[idxM] = { x: p.x, y: p.y };
                }
            });
            var minLine = minBins.filter(function(b) { return b !== null; });
            // Prominencia: descarta puntos cuya Y es menor que el promedio de
            // sus vecinos menos un umbral relativo al rango Y total.
            if (minLine.length >= 3) {
                var ys = minLine.map(function(p) { return p.y; });
                var yMin = Math.min.apply(null, ys);
                var yMax = Math.max.apply(null, ys);
                var prom = Math.max(2, (yMax - yMin) * 0.15);
                var filteredMin = minLine.filter(function(p, i) {
                    if (i === 0 || i === minLine.length - 1) return true;
                    var avgNb = (minLine[i - 1].y + minLine[i + 1].y) / 2;
                    return p.y >= avgNb - prom;
                });
                if (filteredMin.length >= 2) minLine = filteredMin;
            }
            if (minLine.length >= 2) {
                // Línea verde uniendo los mínimos
                ctx.strokeStyle = 'rgba(46, 125, 50, 0.95)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(minLine[0].x * sc, minLine[0].y * sc);
                for (var ki = 1; ki < minLine.length; ki++) {
                    ctx.lineTo(minLine[ki].x * sc, minLine[ki].y * sc);
                }
                ctx.stroke();
                // Marcadores en cada mínimo
                ctx.fillStyle = 'rgba(46, 125, 50, 0.95)';
                ctx.strokeStyle = 'rgba(255,255,255,0.9)';
                ctx.lineWidth = 1;
                minLine.forEach(function(p) {
                    ctx.beginPath();
                    ctx.arc(p.x * sc, p.y * sc, 3, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                });
            }
            // Exponer para Fase C (escala / regla)
            AppState.feetMapMinimaLine = minLine.slice();

            // ── Cruce línea-piso × línea-mínimos → escala (Fase C.1) ──
            // Calcula la primera intersección geométrica entre el segmento del
            // piso (pL.floor → pR.floor) y la polilínea de mínimos. En ese
            // punto se evalúa cm_per_px interpolado entre los dos postes
            // (poste1 = 110cm/tape_px_1, poste2 = 110cm/tape_px_2).
            // Sin fallback: si no hay cruce, no se deriva escala.
            if (oc && minLine.length >= 2) {
                var floorA = { x: pL.cx, y: pL.floor };
                var floorB = { x: pR.cx, y: pR.floor };
                var cross = null;
                for (var si = 0; si < minLine.length - 1; si++) {
                    var ix = _segIntersect(floorA, floorB, minLine[si], minLine[si + 1]);
                    if (ix) { cross = ix; break; }
                }
                if (cross) {
                    // t a lo largo del piso (proyección sobre el segmento)
                    var fx = floorB.x - floorA.x;
                    var fy = floorB.y - floorA.y;
                    var fl2 = fx * fx + fy * fy;
                    var t = fl2 > 0
                        ? ((cross.x - floorA.x) * fx + (cross.y - floorA.y) * fy) / fl2
                        : 0;
                    t = Math.max(0, Math.min(1, t));
                    var s1 = VARA_CM / pL.tape_px;
                    var s2 = VARA_CM / pR.tape_px;
                    var cmPerPx = (1 - t) * s1 + t * s2;
                    AppState.rulerScale = {
                        cm_per_px: cmPerPx,
                        anchor: { x: cross.x, y: cross.y },
                        t_floor: t,
                        source: 'feet-floor-cross'
                    };
                    // Dibujar el cruce
                    ctx.fillStyle = 'rgba(229, 57, 53, 0.95)';
                    ctx.strokeStyle = 'rgba(255,255,255,0.95)';
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.arc(cross.x * sc, cross.y * sc, 5, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                    ctx.font = 'bold 12px sans-serif';
                    ctx.fillStyle = 'rgba(229, 57, 53, 1)';
                    var lbl = 'escala: ' + cmPerPx.toFixed(4) + ' cm/px';
                    ctx.fillText(lbl, cross.x * sc + 8, cross.y * sc - 8);
                } else {
                    AppState.rulerScale = null;
                }
            } else {
                AppState.rulerScale = null;
            }
        }
    }

    // Flecha de dirección detectada en la esquina superior derecha
    if (dirInfo.valid) {
        var arrowY = 22, arrowR = 12, arrowX = MAP_W - 60;
        ctx.font = 'bold 13px sans-serif';
        ctx.fillStyle = 'rgba(0,0,0,0.85)';
        var dlbl = (movingRight ? '→ ' : '← ') + 'dirección';
        var dlw = ctx.measureText(dlbl).width;
        ctx.fillText(dlbl, MAP_W - dlw - 12, arrowY);
    }

    // Leyenda de gradiente
    var legendW = 200, legendH = 8, lx = 12, ly = MAP_H - 22;
    var grad = ctx.createLinearGradient(lx, ly, lx + legendW, ly);
    grad.addColorStop(0, 'rgb(33,150,243)');
    grad.addColorStop(1, 'rgb(244,67,54)');
    ctx.fillStyle = grad;
    ctx.fillRect(lx, ly, legendW, legendH);
    ctx.strokeStyle = 'rgba(0,0,0,0.4)';
    ctx.lineWidth = 1;
    ctx.strokeRect(lx, ly, legendW, legendH);
    ctx.font = 'bold 11px sans-serif';
    ctx.fillStyle = 'rgba(0,0,0,0.85)';
    ctx.fillText('frame inicial', lx, ly - 3);
    var endLbl = 'frame final';
    var ew = ctx.measureText(endLbl).width;
    ctx.fillText(endLbl, lx + legendW - ew, ly - 3);

    var modeLabel = mode === 'back' ? 'Patas traseras'
                  : mode === 'front' ? 'Patas delanteras'
                  : 'Todas las patas';
    var dirLabel = dirInfo.valid
        ? (movingRight ? 'derecha (→)' : 'izquierda (←)')
        : '<span style="color:#e65100;">indeterminada (Δcx=' + dirInfo.dx.toFixed(0) + 'px)</span>';
    var modeWarn = (!dirInfo.valid && mode !== 'all')
        ? ' <em style="color:#e65100;">— sin dirección clara, mostrando todas</em>'
        : '';
    var minLineLbl = (AppState.feetMapMinimaLine && AppState.feetMapMinimaLine.length >= 2)
        ? ' · <span style="color:#2e7d32;">●</span> línea verde = mínimos locales (pezuñas plantadas)'
        : '';
    var scaleLbl = AppState.rulerScale
        ? ' · <span style="color:#e53935;">●</span> cruce piso × mínimos → escala <strong>' +
          AppState.rulerScale.cm_per_px.toFixed(4) + ' cm/px</strong>'
        : (AppState.feetMapMinimaLine && AppState.feetMapMinimaLine.length >= 2
            ? ' · <em style="color:#e65100;">sin cruce piso × mínimos</em>'
            : '');
    $('#feetTrajectoryInfo').html(
        '<strong>' + modeLabel + '</strong>' + modeWarn + ' · ' +
        'Dirección: ' + dirLabel + ' · ' +
        '<strong>' + withFeet.length + '</strong> frames · ' +
        '<strong>' + totalFeetAll + '</strong> puntos · ' +
        'color = orden temporal (azul → rojo)' + minLineLbl + scaleLbl
    );
    var pngDataUrl = canvas.toDataURL('image/png');
    // Exponer para que otros flujos (processFolder) puedan persistirlo.
    AppState.feetMapPngDataUrl = pngDataUrl;
    AppState.feetMapPayload = {
        video_w: srcW, video_h: srcH,
        n_frames: results.length,
        n_frames_with_feet: withFeet.length,
        total_feet_points: totalFeetAll,
        filter_mode: mode,
        movement_dx: dirInfo.dx,
        movement_dir: dirInfo.valid ? (movingRight ? 'right' : 'left') : 'unknown',
        locked_reference: AppState.lockedReference || null,
        frames: results.map(function(r) {
            return {
                frameNum: r.frameNum,
                passing_idx: (r.passing_idx !== undefined) ? r.passing_idx : null,
                folder_offset: (r.folder_offset !== undefined) ? r.folder_offset : null,
                cow_height_cm: r.cow_height_cm || null,
                within_rectangle: !!r.within_rectangle,
                animal_bbox_original: r.animal_bbox_original || null,
                feet_points: r.feet_points || []
            };
        })
    };
    $('#feetTrajectoryWrap').empty().append(
        $('<img>').attr('src', pngDataUrl)
                  .css({ 'max-width': '100%', 'height': 'auto',
                         'border-radius': '6px', 'border': '1px solid #ccc',
                         'background': '#fff' })
    );

    // Nombre base (video + timestamp)
    var ts = new Date();
    var pad = function(v) { return (v < 10 ? '0' : '') + v; };
    var stamp = ts.getFullYear() + pad(ts.getMonth()+1) + pad(ts.getDate()) +
                '_' + pad(ts.getHours()) + pad(ts.getMinutes()) + pad(ts.getSeconds());
    var videoBase = (AppState.videoFile && AppState.videoFile.name)
        ? AppState.videoFile.name.replace(/\.[^.]+$/, '').replace(/[^\w\-]/g, '_')
        : 'video';
    var baseName = 'feet_map_' + videoBase + '_' + stamp;

    $('#btnDownloadFeetMap').show().off('click').on('click', function() {
        var a = document.createElement('a');
        a.href = pngDataUrl;
        a.download = baseName + '.png';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    $('#btnDownloadFeetJson').show().off('click').on('click', function() {
        var payload = {
            video: (AppState.videoFile && AppState.videoFile.name) || null,
            video_w: srcW, video_h: srcH,
            timestamp: ts.toISOString(),
            locked_reference: AppState.lockedReference || null,
            n_frames: results.length,
            n_frames_with_feet: withFeet.length,
            total_feet_points: totalFeetAll,
            frames: results.map(function(r) {
                return {
                    frameNum: r.frameNum,
                    passing_idx: (r.passing_idx !== undefined) ? r.passing_idx : null,
                    cow_height_cm: r.cow_height_cm || null,
                    within_rectangle: !!r.within_rectangle,
                    animal_bbox_original: r.animal_bbox_original || null,
                    feet_points: r.feet_points || []
                };
            })
        };
        var blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = baseName + '.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(function() { URL.revokeObjectURL(url); }, 1000);
    });

    $('#feetTrajectoryCard').show();
    renderRulerCard();
}

// ── Regla de medición (Fase C.2) ──
// Usa AppState.rulerScale (cm_per_px en el cruce piso × mínimos) para medir
// distancias en cualquier frame del video o sobre un thumbnail procesado.
// Los puntos se guardan en AppState.rulerPoints en coords del video original.

function renderRulerCard() {
    if (!AppState.rulerScale) {
        $('#rulerCard').hide();
        return;
    }
    $('#rulerCard').show();

    var mode = $('#rulerSourceMode').val() || 'video';
    // Poblar el dropdown de thumbnails con todas las pasadas (frameNum + altura si la hay)
    var $pick = $('#rulerThumbPick');
    if (mode === 'thumb') {
        $pick.show();
        // Repobrar solo si la lista cambió
        var results = AppState.passingResults || [];
        var optsKey = results.map(function(r) { return r.frameNum; }).join(',');
        if ($pick.data('opts-key') !== optsKey) {
            $pick.empty();
            results.forEach(function(r, i) {
                var lbl = 'Frame ' + r.frameNum +
                    (r.cow_height_cm != null ? ' (' + r.cow_height_cm.toFixed(1) + 'cm)' : '');
                $pick.append($('<option>').val(i).text(lbl));
            });
            $pick.data('opts-key', optsKey);
        }
    } else {
        $pick.hide();
    }

    var infoBase = 'Escala: <strong>' + AppState.rulerScale.cm_per_px.toFixed(4) +
        ' cm/px</strong> (ancla en ' +
        AppState.rulerScale.anchor.x.toFixed(0) + ',' +
        AppState.rulerScale.anchor.y.toFixed(0) + ')';
    $('#rulerInfo').html(infoBase + ' · Clickeá dos puntos sobre el frame para medir.');

    // Si todavía no hay snapshot, capturar el frame actual del video automáticamente
    if (!AppState.rulerSourceCanvas) {
        if (mode === 'thumb' && AppState.passingResults && AppState.passingResults.length > 0) {
            rulerLoadThumb(0);
        } else {
            rulerCaptureVideoFrame();
        }
    } else {
        rulerDraw();
    }
}

// Captura el frame actual del <video> en un canvas y lo carga como fuente.
function rulerCaptureVideoFrame() {
    var video = document.getElementById('videoPlayer');
    if (!video || !video.videoWidth) {
        alert('No hay video cargado.');
        return;
    }
    var c = document.createElement('canvas');
    c.width = video.videoWidth;
    c.height = video.videoHeight;
    c.getContext('2d').drawImage(video, 0, 0);
    AppState.rulerSourceCanvas = c;
    AppState.rulerSourceMeta = { w_orig: c.width, h_orig: c.height,
                                  frameNum: getCurrentFrameNum() };
    AppState.rulerPoints = [];
    rulerDraw();
}

// Carga un thumbnail procesado (de AppState.passingResults) como fuente.
function rulerLoadThumb(idx) {
    var r = AppState.passingResults[idx];
    if (!r || !r.annotated_image) return;
    var img = new Image();
    img.onload = function() {
        var c = document.createElement('canvas');
        // Usar dimensiones del video original para mantener escala consistente
        var wo = r.video_w || img.width;
        var ho = r.video_h || img.height;
        c.width = wo;
        c.height = ho;
        c.getContext('2d').drawImage(img, 0, 0, wo, ho);
        AppState.rulerSourceCanvas = c;
        AppState.rulerSourceMeta = { w_orig: wo, h_orig: ho, frameNum: r.frameNum };
        AppState.rulerPoints = [];
        rulerDraw();
    };
    img.src = r.annotated_image;
}

// Dibuja el snapshot + puntos + línea de medición.
function rulerDraw() {
    var src = AppState.rulerSourceCanvas;
    if (!src || !AppState.rulerScale) {
        $('#rulerWrap').empty();
        return;
    }
    var DISP_W = 900;
    var sc = DISP_W / src.width;
    var DISP_H = Math.round(src.height * sc);
    var canvas = document.createElement('canvas');
    canvas.width = DISP_W;
    canvas.height = DISP_H;
    canvas.style.cursor = 'crosshair';
    var ctx = canvas.getContext('2d');
    ctx.drawImage(src, 0, 0, DISP_W, DISP_H);

    // Marcar el ancla de escala
    var anc = AppState.rulerScale.anchor;
    ctx.fillStyle = 'rgba(229, 57, 53, 0.95)';
    ctx.strokeStyle = 'rgba(255,255,255,0.95)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(anc.x * sc, anc.y * sc, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    var pts = AppState.rulerPoints || [];
    // Puntos clickeados
    pts.forEach(function(p, i) {
        ctx.fillStyle = i === 0 ? 'rgba(33,150,243,1)' : 'rgba(76,175,80,1)';
        ctx.strokeStyle = 'rgba(255,255,255,1)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x * sc, p.y * sc, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    });

    // Línea entre puntos + distancia
    var distLbl = '';
    if (pts.length === 2) {
        ctx.strokeStyle = 'rgba(255, 193, 7, 1)';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(pts[0].x * sc, pts[0].y * sc);
        ctx.lineTo(pts[1].x * sc, pts[1].y * sc);
        ctx.stroke();
        var dx = pts[1].x - pts[0].x;
        var dy = pts[1].y - pts[0].y;
        var distPx = Math.sqrt(dx * dx + dy * dy);
        var distCm = distPx * AppState.rulerScale.cm_per_px;
        // Label en el punto medio
        var mx = ((pts[0].x + pts[1].x) / 2) * sc;
        var my = ((pts[0].y + pts[1].y) / 2) * sc - 8;
        var txt = distCm.toFixed(1) + ' cm  (' + distPx.toFixed(0) + ' px)';
        ctx.font = 'bold 14px sans-serif';
        var tw = ctx.measureText(txt).width;
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(mx - tw / 2 - 6, my - 16, tw + 12, 20);
        ctx.fillStyle = 'rgba(255,255,255,1)';
        ctx.fillText(txt, mx - tw / 2, my);
        distLbl = ' · Distancia: <strong>' + distCm.toFixed(2) + ' cm</strong> (' +
                  distPx.toFixed(1) + ' px)';
    } else if (pts.length === 1) {
        distLbl = ' · 1 punto fijado · click para el segundo.';
    }

    // Click handler en coords originales del video
    canvas.addEventListener('click', function(ev) {
        var rect = canvas.getBoundingClientRect();
        var cx = (ev.clientX - rect.left) * (canvas.width / rect.width);
        var cy = (ev.clientY - rect.top) * (canvas.height / rect.height);
        var ox = cx / sc;
        var oy = cy / sc;
        if (AppState.rulerPoints.length >= 2) {
            AppState.rulerPoints = [{ x: ox, y: oy }];
        } else {
            AppState.rulerPoints.push({ x: ox, y: oy });
        }
        rulerDraw();
    });

    $('#rulerWrap').empty().append(canvas);

    var src_meta = AppState.rulerSourceMeta || {};
    var srcLbl = src_meta.frameNum != null
        ? 'Frame ' + src_meta.frameNum
        : 'Snapshot';
    var infoBase = 'Escala: <strong>' + AppState.rulerScale.cm_per_px.toFixed(4) +
        ' cm/px</strong> · Fuente: ' + srcLbl;
    $('#rulerInfo').html(infoBase + distLbl);
}

function rulerResetPoints() {
    AppState.rulerPoints = [];
    rulerDraw();
}



// ===== campo feet_points en passingResults =====
                    feet_points: Array.isArray(data.feet_points) ? data.feet_points : [],


// ===== dibujo de pezuñas en overlay =====
        // Pezuñas detectadas (puntos verdes) — siempre, válido o no
        if (Array.isArray(data.feet_points) && data.feet_points.length > 0) {
            ctx.fillStyle = 'rgba(76,175,80,1)';
            ctx.strokeStyle = 'rgba(0,0,0,0.85)';
            ctx.lineWidth = 1;
            data.feet_points.forEach(function(fp) {
                ctx.beginPath();
                ctx.arc(fp.x * scale, fp.y * scale, 2.5, 0, Math.PI*2);
                ctx.fill();
                ctx.stroke();
            });
        }



// ===== processFolder paso 8: guardar feet_map =====
        // 8. Guardar el mapa de pezuñas en la carpeta (PNG + JSON)
        var feetSavedMsg = '';
        if (AppState.feetMapPngDataUrl) {
            try {
                var saveResp = await fetch('/save_feet_map', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        folder: folder,
                        png_data_url: AppState.feetMapPngDataUrl,
                        payload: AppState.feetMapPayload || {}
                    })
                });
                var saveData = await saveResp.json();
                if (saveData.success) {
                    feetSavedMsg = '\nMapa de pezuñas: feet_map.png guardado';
                    console.log('[processFolder] feet_map guardado en', saveData.png_path);
                } else {
                    console.warn('[processFolder] no se pudo guardar feet_map:', saveData.error);
                }
            } catch (e) {
                console.warn('[processFolder] error guardando feet_map:', e);
            }
        }



// ===== modal frames guardados: badge feet map =====
    if (data.feet_map_exists) metaTxt += ' · mapa de pezuñas ✓';


// ===== modal frames guardados: imagen feet map =====
    var $feetWrap = $('#savedFeetMapWrap');
    if (data.feet_map_exists) {
        var url = '/saved_feet_map/' + folder + '?t=' + Date.now();
        $feetWrap.show().html(
            '<div style="font-weight:600; font-size:0.85em; margin-bottom:4px;">Trayectoria de pezuñas guardada:</div>' +
            '<img src="' + url + '" style="max-width:100%; height:auto; border:1px solid #ccc; border-radius:6px; background:#fff;">'
        );
    } else {
        $feetWrap.hide().empty();
    }



// ===== handlers feetFilterMode + regla =====
    $('#feetFilterMode').on('change', function() {
        if (AppState.passingResults && AppState.passingResults.length > 0) {
            renderFeetTrajectoryMap();
        }
    });
    $('#rulerSourceMode').on('change', function() {
        renderRulerCard();
        var mode = $(this).val();
        if (mode === 'thumb') {
            var idx = parseInt($('#rulerThumbPick').val()) || 0;
            rulerLoadThumb(idx);
        } else {
            // Capturar el frame actual del video al cambiar a modo video
            rulerCaptureVideoFrame();
        }
    });
    $('#rulerThumbPick').on('change', function() {
        var idx = parseInt($(this).val());
        if (!isNaN(idx)) rulerLoadThumb(idx);
    });
    $('#btnRulerCapture').on('click', function() {
        var mode = $('#rulerSourceMode').val();
        if (mode === 'thumb') {
            var idx = parseInt($('#rulerThumbPick').val()) || 0;
            rulerLoadThumb(idx);
        } else {
            rulerCaptureVideoFrame();
        }
    });
    $('#btnRulerReset').on('click', function() { rulerResetPoints(); });


// ===== campos AppState =====
    passingStats: { analyzed: 0, detected: 0, in_rect: 0, out_rect: 0, no_cow: 0 },
    // Mapa de pezuñas: línea de mínimos locales (Fase B) y escala derivada (Fase C)
    feetMapMinimaLine: [],
    rulerScale: null,           // { cm_per_px, anchor: {x, y}, source: 'feet-floor-cross' | ... }
    rulerPoints: [],            // [{x, y}] en coords del video original
    rulerSourceCanvas: null,    // HTMLCanvasElement con el snapshot actual
    rulerSourceMeta: null       // { w_orig, h_orig, frameNum }
};