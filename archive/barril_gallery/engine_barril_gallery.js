// ── Galería de "frames del barril usados para el 3D" ──
// Disjunta de la galería de altura: acá van solo los frames donde el poste
// cercano NO solapa con el bbox de la vaca, así que la mask del barril sale
// limpia (sin la muesca del poste). Esos frames alimentan el consenso
// multi-frame del 3D.

function renderBarril3DGallery() {
    var $header = $('#barril3dHeader');
    var $gal = $('#barril3dGallery');
    $gal.empty();
    $header.empty();

    var eligibles = [];
    AppState.passingResults.forEach(function(r, i) {
        // 21 frames alrededor del cruce: -10..+10 (incluye el 0).
        // El frame 0 tiene poste superpuesto pero la malla se repara
        // automáticamente (ver _reparar_mascara_oclusion en backend).
        if (!r.barril_eligible) return;
        var c = r.barril_contour_norm;
        if (!c || !c.tops_cm || !c.bottoms_cm) return;
        if (!r.annotated_image || !r.animal_bbox_original || !r.cm_per_px) return;
        eligibles.push({ r: r, idx: i });
    });

    if (!eligibles.length) {
        $header.hide();
        $gal.hide();
        return;
    }

    var nOutliers = eligibles.filter(function(it) { return it.r.barril_outlier; }).length;
    var nInliers = eligibles.length - nOutliers;
    var outlierSpan = nOutliers > 0
        ? ' · <span style="color:#e65100;"><strong>' + nOutliers + ' descartado(s)</strong> por ancho ≥30% bajo la media</span>'
        : '';
    $header.html(
        '<h6 style="margin-bottom:8px;"><i class="fas fa-dharmachakra"></i> ' +
        'Frames del barril (' + nInliers + ' usados para el modelo 3D)' + outlierSpan + '</h6>' +
        '<div style="color:#666; font-size:0.85em; margin-bottom:10px;">' +
        'Frames sin poste solapando. Se descarta cualquier barril cuyo ancho ' +
        'esté ≥30% por debajo del ancho medio de la pasada (mascara cortada o ' +
        'cuerpo parcial). Después del descarte se recalcula la media. El ' +
        'contorno naranja es la silueta usada.</div>'
    ).show();
    $gal.css('display', 'flex').show();

    eligibles.forEach(function(item) { appendBarril3DThumbnail(item.idx); });
}

function appendBarril3DThumbnail(idx) {
    var r = AppState.passingResults[idx];
    _buildBarril3DThumbUrl(r).then(function(url) {
        _injectBarril3DCard(idx, url);
    });
}

// Construye (asincrónico) un data URL con el contorno del barril (tops+bottoms
// ya reparados) dibujado sobre el thumbnail del frame. Reusable por la galería
// y por la descarga de resultado.
function _buildBarril3DThumbUrl(r) {
    return new Promise(function(resolve) {
        if (!r || !r.annotated_image) { resolve(null); return; }
        var img = new Image();
        img.onload = function() {
            var canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);

            var c = r.barril_contour_norm;
            var bb = r.animal_bbox_original;
            if (!c || !c.tops_cm || !c.bottoms_cm || !bb || !r.cm_per_px) {
                resolve(canvas.toDataURL('image/jpeg', 0.82));
                return;
            }
            var s = (r.video_w && r.video_w > 0) ? (img.width / r.video_w) : 1;
            var x1 = bb[0] * s;
            var x2 = bb[2] * s;
            var y_floor = bb[3] * s;
            var px_per_cm = 1 / r.cm_per_px;
            var px_scale = px_per_cm * s;
            var N = c.n_samples;

            ctx.beginPath();
            for (var i = 0; i < N; i++) {
                var xi = x1 + (x2 - x1) * i / (N - 1);
                var yi = y_floor - c.tops_cm[i] * px_scale;
                if (i === 0) ctx.moveTo(xi, yi);
                else ctx.lineTo(xi, yi);
            }
            for (var j = N - 1; j >= 0; j--) {
                var xj = x1 + (x2 - x1) * j / (N - 1);
                var yj = y_floor - c.bottoms_cm[j] * px_scale;
                ctx.lineTo(xj, yj);
            }
            ctx.closePath();
            ctx.fillStyle = 'rgba(255, 152, 0, 0.18)';
            ctx.fill();
            ctx.strokeStyle = 'rgba(255, 87, 34, 0.95)';
            ctx.lineWidth = 2;
            ctx.stroke();

            resolve(canvas.toDataURL('image/jpeg', 0.82));
        };
        img.onerror = function() { resolve(r.annotated_image); };
        img.src = r.annotated_image;
    });
}

function _injectBarril3DCard(idx, thumbUrl) {
    var r = AppState.passingResults[idx];
    var c = r.barril_contour_norm;
    var widthStr = c.width_cm ? c.width_cm.toFixed(1) + ' cm' : 'N/A';
    var volStr = (r.barril_volumen_litros != null) ? r.barril_volumen_litros.toFixed(1) + ' L' : 'N/A';
    var rellenoStr = (r.barril_cols_rellenadas != null && r.barril_cols_rellenadas > 0)
        ? ' · <span style="color:#7b1fa2;">rellenadas: ' + r.barril_cols_rellenadas + '</span>' : '';

    var isOutlier = !!r.barril_outlier;
    var borderColor = isOutlier ? '#ff9800' : '#ff7043';
    var devPctStr = (r.barril_width_dev_pct != null) ? r.barril_width_dev_pct.toFixed(1) + '%' : '–';
    var statusTag = '';
    if (isOutlier) {
        var reasonTxt = '';
        var reasonTitle = '';
        if (r.barril_outlier_reason === 'width_low') {
            reasonTxt = 'ancho bajo (' + devPctStr + ')';
            reasonTitle = 'Ancho del barril ≥30% por debajo de la mediana — máscara cortada por el poste';
        } else if (r.barril_outlier_reason === 'width_high') {
            reasonTxt = 'ancho alto (+' + devPctStr + ')';
            reasonTitle = 'Ancho del barril ≥30% sobre la mediana — postura estirada';
        } else if (r.barril_outlier_reason === 'oclusion') {
            reasonTxt = 'oclusión (' + (r.barril_cols_rellenadas || 0) + ' cols)';
            reasonTitle = 'Más de 15 columnas reparadas por oclusión severa del poste';
        } else {
            reasonTxt = 'descartado';
        }
        statusTag = '<span style="color:#e65100; font-weight:700;" title="' + reasonTitle + '"> · ⚠ ' + reasonTxt + '</span>';
    }

    var idxStr = (r.passing_idx != null)
        ? (r.passing_idx === 0 ? '0 (cruce)' : (r.passing_idx > 0 ? '+' + r.passing_idx : r.passing_idx))
        : '–';
    var idxBadge = '<span style="display:inline-block; min-width:32px; padding:1px 6px; ' +
        'background:' + (r.passing_idx === 0 ? '#1976d2' : '#37474f') + '; color:#fff; ' +
        'border-radius:10px; font-weight:700; margin-right:6px; font-size:0.85em;">' +
        idxStr + '</span>';

    var html = '<div class="col-md-6 col-12 mb-2" id="barril3d-card-' + idx + '">' +
        '<div style="padding:4px; cursor:pointer; border:2px solid ' + borderColor + '; border-radius:6px; background:#fff; overflow:hidden;" ' +
        'onclick="showPassingDetail(' + idx + ')">' +
        '<img src="' + thumbUrl + '" style="display:block; width:100%; height:auto; border-radius:4px;">' +
        '<div style="font-size:0.85em; padding:4px 6px; line-height:1.3;">' +
        idxBadge +
        '<strong>Frame ' + r.frameNum + '</strong> · ' +
        'Vol: <strong>' + volStr + '</strong> · ' +
        'Ancho: <strong>' + widthStr + '</strong>' + rellenoStr + statusTag +
        '</div></div></div>';
    $('#barril3dGallery').append(html);
}

