"""
Generación de reportes visuales de inferencia en video.
Muestra todos los frames con detección de animal, el recorte usado
y la confianza del modelo de reconocimiento.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from .infer import MultiVideoResult, TrackResult, VideoResult


def _image_to_base64(img: Image.Image, quality: int = 85) -> str:
    """Convierte una imagen PIL a data URI base64."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _draw_detection_on_frame(
    frame_rgb: np.ndarray,
    bbox: list[float],
    bbox_padded: list[float],
    animal_type: str,
    yolo_conf: float,
    label: str,
    recog_conf: float,
) -> Image.Image:
    """Dibuja bounding box y etiquetas sobre un frame."""
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        try:
            font = ImageFont.load_default()
            font_small = font
        except Exception:
            font = None
            font_small = None

    x1, y1, x2, y2 = bbox
    px1, py1, px2, py2 = bbox_padded

    # Dibujar bbox original (verde)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

    # Dibujar bbox con padding (amarillo, punteado simulado con líneas más finas)
    draw.rectangle([px1, py1, px2, py2], outline=(255, 255, 0), width=1)

    # Etiqueta YOLO (arriba del bbox)
    yolo_label = f"YOLO: {animal_type} {yolo_conf:.0%}"
    if font:
        draw.rectangle(
            [x1, y1 - 22, x1 + len(yolo_label) * 8, y1],
            fill=(0, 0, 0, 200),
        )
        draw.text((x1 + 2, y1 - 20), yolo_label, fill=(0, 255, 0), font=font_small)

    # Etiqueta de reconocimiento (abajo del bbox)
    recog_label = f"ID: {label} ({recog_conf:.1%})"
    if font:
        draw.rectangle(
            [x1, y2, x1 + len(recog_label) * 8, y2 + 22],
            fill=(0, 0, 0, 200),
        )
        color = (0, 200, 255) if recog_conf >= 0.5 else (255, 100, 100)
        draw.text((x1 + 2, y2 + 2), recog_label, fill=color, font=font_small)

    return img


def generate_video_report(
    video_result: "VideoResult",
    video_filename: str = "video",
    dataset_name: str = "",
    threshold: float = 0.70,
    max_frames_in_report: int = 60,
) -> str:
    """
    Genera un reporte HTML completo con los resultados del análisis de video.

    Args:
        video_result: Resultado de predict_video_with_detection()
        video_filename: Nombre del archivo de video analizado
        dataset_name: Nombre del dataset/establecimiento usado
        threshold: Umbral de confianza usado
        max_frames_in_report: Máximo de frames a incluir en el reporte

    Returns:
        Contenido HTML del reporte (string)
    """
    pred = video_result.prediction
    # Usar solo los top frames del individuo ganador (ya ordenados por confianza)
    detections = video_result.top_detections[:max_frames_in_report]

    # Generar sección de frames con detecciones
    frames_html = ""
    for i, det in enumerate(detections):
        # Imagen anotada del frame completo
        annotated = _draw_detection_on_frame(
            det.frame_rgb,
            det.bbox,
            det.bbox_padded,
            det.animal_type,
            det.yolo_confidence,
            det.label,
            det.recognition_confidence,
        )
        # Redimensionar para el reporte (máximo 480px de ancho)
        max_w = 480
        if annotated.width > max_w:
            ratio = max_w / annotated.width
            annotated = annotated.resize(
                (max_w, int(annotated.height * ratio)), Image.LANCZOS
            )

        frame_b64 = _image_to_base64(annotated)

        # Imagen del recorte
        crop_img = Image.fromarray(det.crop_rgb)
        crop_size = 160
        if crop_img.width > crop_size or crop_img.height > crop_size:
            crop_img.thumbnail((crop_size, crop_size), Image.LANCZOS)
        crop_b64 = _image_to_base64(crop_img)

        # Color de la barra de confianza
        conf_pct = det.recognition_confidence * 100
        if conf_pct >= threshold * 100:
            bar_color = "#4caf50"
            status = "Reconocido"
        elif conf_pct >= threshold * 50:
            bar_color = "#ff9800"
            status = "Baja confianza"
        else:
            bar_color = "#f44336"
            status = "Desconocido"

        frames_html += f"""
        <div class="frame-card">
            <div class="frame-header">
                <span class="frame-num">Frame #{det.frame_idx}</span>
                <span class="frame-status" style="background-color: {bar_color};">{status}</span>
            </div>
            <div class="frame-content">
                <div class="frame-image">
                    <img src="{frame_b64}" alt="Frame {det.frame_idx}" />
                </div>
                <div class="frame-details">
                    <div class="crop-section">
                        <span class="detail-label">Recorte analizado:</span>
                        <img src="{crop_b64}" alt="Crop" class="crop-img" />
                    </div>
                    <div class="metrics">
                        <div class="metric-row">
                            <span class="detail-label">YOLO Detección:</span>
                            <span class="detail-value">{det.animal_type} ({det.yolo_confidence:.0%})</span>
                        </div>
                        <div class="metric-row">
                            <span class="detail-label">Individuo:</span>
                            <span class="detail-value" style="font-weight: bold;">{det.label}</span>
                        </div>
                        <div class="metric-row">
                            <span class="detail-label">Confianza:</span>
                            <span class="detail-value">{det.recognition_confidence:.1%}</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {conf_pct:.1f}%; background-color: {bar_color};"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """

    # Estadísticas de distribución de predicciones por clase
    class_votes: dict[str, list[float]] = {}
    for det in video_result.frame_detections:
        if det.label not in class_votes:
            class_votes[det.label] = []
        class_votes[det.label].append(det.recognition_confidence)

    class_stats_html = ""
    for label in sorted(class_votes, key=lambda k: -len(class_votes[k])):
        confs = class_votes[label]
        avg_conf = np.mean(confs)
        count = len(confs)
        pct = count / len(video_result.frame_detections) * 100 if video_result.frame_detections else 0
        class_stats_html += f"""
        <div class="stat-item">
            <span class="stat-label">{label}</span>
            <span class="stat-value">{count} frames ({pct:.0f}%) - conf. prom: {avg_conf:.1%}</span>
        </div>
        """

    # Resultado final
    if pred.decision == "known":
        result_color = "#4caf50"
        result_bg = "#e8f5e9"
        result_text = f"IDENTIFICADO: {pred.label}"
    else:
        result_color = "#f44336"
        result_bg = "#ffebee"
        result_text = f"DESCONOCIDO (mejor: {pred.label})"

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Inferencia - {video_filename}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; color: #333; background: #f5f5f5; padding: 20px;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{
            background: white; padding: 24px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
        }}
        h1 {{ color: #2c5530; margin-bottom: 8px; font-size: 1.6em; }}
        .header-info {{ color: #666; font-size: 0.9em; }}
        .header-info span {{ margin-right: 20px; }}
        .result-banner {{
            background: {result_bg}; border-left: 5px solid {result_color};
            padding: 20px; border-radius: 8px; margin-bottom: 20px;
        }}
        .result-banner h2 {{ color: {result_color}; font-size: 1.5em; margin-bottom: 4px; }}
        .result-banner .conf {{ font-size: 1.1em; color: #555; }}
        .result-banner .conf-detail {{ font-size: 0.95em; color: #777; margin-top: 4px; }}
        .stats-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px; margin-bottom: 20px;
        }}
        .stat-card {{
            background: white; padding: 16px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-top: 3px solid #4a7c59;
        }}
        .stat-card h3 {{ color: #2c5530; font-size: 0.95em; margin-bottom: 8px; }}
        .stat-card .big-number {{ font-size: 2em; font-weight: bold; color: #4a7c59; }}
        .stat-card .sub {{ color: #888; font-size: 0.85em; }}
        .section {{
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
        }}
        .section h2 {{ color: #2c5530; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #e0e0e0; }}
        .stat-item {{
            display: flex; justify-content: space-between; padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        .stat-label {{ font-weight: 500; }}
        .stat-value {{ color: #4a7c59; font-weight: 600; }}
        .frame-card {{
            background: white; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 16px;
            overflow: hidden;
        }}
        .frame-header {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 16px; background: #f9f9f9; border-bottom: 1px solid #eee;
        }}
        .frame-num {{ font-weight: 600; color: #333; }}
        .frame-status {{
            color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.85em;
        }}
        .frame-content {{ display: flex; gap: 16px; padding: 16px; flex-wrap: wrap; }}
        .frame-image {{ flex: 1; min-width: 280px; }}
        .frame-image img {{ width: 100%; border-radius: 4px; }}
        .frame-details {{ flex: 0 0 200px; min-width: 180px; }}
        .crop-section {{ margin-bottom: 12px; }}
        .crop-img {{ max-width: 160px; border-radius: 4px; border: 2px solid #4a7c59; margin-top: 4px; }}
        .detail-label {{ font-size: 0.85em; color: #888; display: block; }}
        .detail-value {{ font-size: 0.95em; }}
        .metric-row {{ margin-bottom: 6px; }}
        .confidence-bar {{
            width: 100%; height: 8px; background: #e0e0e0; border-radius: 4px;
            margin-top: 4px; overflow: hidden;
        }}
        .confidence-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
        .legend {{
            background: #fff9c4; padding: 12px 16px; border-radius: 6px;
            margin-bottom: 16px; font-size: 0.9em;
        }}
        .legend ul {{ margin-left: 16px; margin-top: 4px; }}
        .footer {{
            text-align: center; color: #999; font-size: 0.8em; padding: 20px;
        }}
        @media print {{
            body {{ background: white; }}
            .frame-card {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Reporte de Inferencia en Video</h1>
            <div class="header-info">
                <span>Archivo: <strong>{video_filename}</strong></span>
                <span>Dataset: <strong>{dataset_name}</strong></span>
                <span>Fecha: <strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</strong></span>
            </div>
        </div>

        <div class="result-banner">
            <h2>{result_text}</h2>
            <div class="conf">
                Confianza (top {len(video_result.top_detections)} frames): {pred.confidence:.1%} | Umbral: {threshold:.0%}
            </div>
            <div class="conf-detail">
                {video_result.winning_count} de {video_result.frames_with_detections} frames
                votaron por <strong>{video_result.winning_label}</strong>
                (concordancia: {video_result.agreement_ratio:.0%})
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Frames Extraídos</h3>
                <div class="big-number">{video_result.total_frames_extracted}</div>
                <div class="sub">Total del video</div>
            </div>
            <div class="stat-card">
                <h3>Con Detección</h3>
                <div class="big-number">{video_result.frames_with_detections}</div>
                <div class="sub">{video_result.frames_with_detections / max(video_result.total_frames_extracted, 1) * 100:.0f}% de los frames</div>
            </div>
            <div class="stat-card">
                <h3>Sin Detección</h3>
                <div class="big-number">{video_result.frames_without_detections}</div>
                <div class="sub">Descartados (sin animal)</div>
            </div>
            <div class="stat-card">
                <h3>Frames {video_result.winning_label}</h3>
                <div class="big-number">{video_result.winning_count}</div>
                <div class="sub">Concordancia: {video_result.agreement_ratio:.0%}</div>
            </div>
            <div class="stat-card">
                <h3>Conf. Top {len(video_result.top_detections)}</h3>
                <div class="big-number">{video_result.winning_class_avg_conf:.0%}</div>
                <div class="sub">Promedio mejores frames</div>
            </div>
        </div>

        <div class="section">
            <h2>Distribución de Predicciones por Clase</h2>
            {class_stats_html if class_stats_html else '<p style="color: #999;">Sin detecciones</p>'}
        </div>

        <div class="section">
            <h2>Mejores Frames de {video_result.winning_label} ({len(detections)} de {video_result.winning_count} del individuo)</h2>
            <div class="legend">
                <strong>Leyenda:</strong>
                <ul>
                    <li><strong style="color: #0f0;">Rectángulo verde:</strong> Bounding box de YOLO (animal detectado)</li>
                    <li><strong style="color: #ff0;">Rectángulo amarillo:</strong> Región expandida usada para recorte</li>
                    <li><strong>Recorte:</strong> Sección de la imagen enviada al modelo de reconocimiento</li>
                </ul>
            </div>
            {frames_html if frames_html else '<p style="color: #999;">No se detectaron animales en ningún frame.</p>'}
        </div>

        <div class="footer">
            <p>Reporte generado por SRFPP - Sistema de Reconocimiento Facial y Predicción de Peso</p>
        </div>
    </div>
</body>
</html>
    """
    return html


def _render_track_frames_html(
    tr: "TrackResult",
    threshold: float,
    max_frames: int = 60,
) -> str:
    """Render top detection frames for a single track as HTML cards."""
    detections = tr.top_detections[:max_frames]
    frames_html = ""
    for det in detections:
        annotated = _draw_detection_on_frame(
            det.frame_rgb,
            det.bbox,
            det.bbox_padded,
            det.animal_type,
            det.yolo_confidence,
            det.label,
            det.recognition_confidence,
        )
        max_w = 480
        if annotated.width > max_w:
            ratio = max_w / annotated.width
            annotated = annotated.resize(
                (max_w, int(annotated.height * ratio)), Image.LANCZOS
            )
        frame_b64 = _image_to_base64(annotated)

        crop_img = Image.fromarray(det.crop_rgb)
        crop_size = 160
        if crop_img.width > crop_size or crop_img.height > crop_size:
            crop_img.thumbnail((crop_size, crop_size), Image.LANCZOS)
        crop_b64 = _image_to_base64(crop_img)

        conf_pct = det.recognition_confidence * 100
        if conf_pct >= threshold * 100:
            bar_color = "#4caf50"
            status = "Reconocido"
        elif conf_pct >= threshold * 50:
            bar_color = "#ff9800"
            status = "Baja confianza"
        else:
            bar_color = "#f44336"
            status = "Desconocido"

        frames_html += f"""
        <div class="frame-card">
            <div class="frame-header">
                <span class="frame-num">Frame #{det.frame_idx}</span>
                <span class="frame-status" style="background-color: {bar_color};">{status}</span>
            </div>
            <div class="frame-content">
                <div class="frame-image">
                    <img src="{frame_b64}" alt="Frame {det.frame_idx}" />
                </div>
                <div class="frame-details">
                    <div class="crop-section">
                        <span class="detail-label">Recorte analizado:</span>
                        <img src="{crop_b64}" alt="Crop" class="crop-img" />
                    </div>
                    <div class="metrics">
                        <div class="metric-row">
                            <span class="detail-label">YOLO Deteccion:</span>
                            <span class="detail-value">{det.animal_type} ({det.yolo_confidence:.0%})</span>
                        </div>
                        <div class="metric-row">
                            <span class="detail-label">Individuo:</span>
                            <span class="detail-value" style="font-weight: bold;">{det.label}</span>
                        </div>
                        <div class="metric-row">
                            <span class="detail-label">Confianza:</span>
                            <span class="detail-value">{det.recognition_confidence:.1%}</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {conf_pct:.1f}%; background-color: {bar_color};"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    return frames_html


def _render_track_class_stats_html(tr: "TrackResult") -> str:
    """Render class distribution stats for a single track."""
    class_votes: dict[str, list[float]] = {}
    for det in tr.frame_detections:
        if det.label not in class_votes:
            class_votes[det.label] = []
        class_votes[det.label].append(det.recognition_confidence)

    html = ""
    for label in sorted(class_votes, key=lambda k: -len(class_votes[k])):
        confs = class_votes[label]
        avg_conf = np.mean(confs)
        count = len(confs)
        pct = count / len(tr.frame_detections) * 100 if tr.frame_detections else 0
        html += f"""
        <div class="stat-item">
            <span class="stat-label">{label}</span>
            <span class="stat-value">{count} frames ({pct:.0f}%) - conf. prom: {avg_conf:.1%}</span>
        </div>
        """
    return html


def generate_multi_video_report(
    multi_result: "MultiVideoResult",
    video_filename: str = "video",
    dataset_name: str = "",
    threshold: float = 0.70,
    max_frames_in_report: int = 30,
) -> str:
    """
    Generate an HTML report for multi-animal tracking results.

    Args:
        multi_result: Result from predict_video_multi_animal()
        video_filename: Name of the video file analysed
        dataset_name: Dataset/establishment name
        threshold: Confidence threshold used
        max_frames_in_report: Max frames per track to include

    Returns:
        HTML string of the full report
    """
    n_tracks = len(multi_result.tracks)
    det_pct = (
        multi_result.frames_with_detections / max(multi_result.total_frames_extracted, 1) * 100
    )

    # Build per-track HTML sections
    tracks_html = ""
    for i, tr in enumerate(multi_result.tracks):
        pred = tr.prediction
        if pred.decision == "known":
            r_color = "#4caf50"
            r_bg = "#e8f5e9"
            r_text = f"IDENTIFICADO: {pred.label}"
        else:
            r_color = "#f44336"
            r_bg = "#ffebee"
            r_text = f"DESCONOCIDO (mejor: {pred.label})"

        frames_html = _render_track_frames_html(tr, threshold, max_frames_in_report)
        class_stats_html = _render_track_class_stats_html(tr)

        tracks_html += f"""
        <div class="section" style="border-left: 4px solid {r_color};">
            <h2>Animal {i + 1}: {tr.winning_label} (Track #{tr.track_id})</h2>
            <div class="result-banner" style="background: {r_bg}; border-left: 5px solid {r_color};">
                <h2 style="color: {r_color};">{r_text}</h2>
                <div class="conf">
                    Confianza (top {len(tr.top_detections)} frames): {pred.confidence:.1%} | Umbral: {threshold:.0%}
                </div>
                <div class="conf-detail">
                    {tr.winning_count} de {len(tr.frame_detections)} frames
                    votaron por <strong>{tr.winning_label}</strong>
                    (concordancia: {tr.agreement_ratio:.0%})
                </div>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Frames del Track</h3>
                    <div class="big-number">{len(tr.frame_detections)}</div>
                    <div class="sub">Detecciones totales</div>
                </div>
                <div class="stat-card">
                    <h3>Frames {tr.winning_label}</h3>
                    <div class="big-number">{tr.winning_count}</div>
                    <div class="sub">Concordancia: {tr.agreement_ratio:.0%}</div>
                </div>
                <div class="stat-card">
                    <h3>Conf. Top {len(tr.top_detections)}</h3>
                    <div class="big-number">{tr.winning_class_avg_conf:.0%}</div>
                    <div class="sub">Promedio mejores frames</div>
                </div>
                <div class="stat-card">
                    <h3>Rostro / Cuerpo</h3>
                    <div class="big-number">{tr.frames_with_face} / {tr.frames_with_body_only}</div>
                    <div class="sub">Detecciones face vs body</div>
                </div>
            </div>
            <div class="section" style="box-shadow: none; padding: 12px 0;">
                <h2>Distribucion de Predicciones</h2>
                {class_stats_html if class_stats_html else '<p style="color: #999;">Sin detecciones</p>'}
            </div>
            <div class="section" style="box-shadow: none; padding: 12px 0;">
                <h2>Mejores Frames ({len(tr.top_detections[:max_frames_in_report])} de {tr.winning_count})</h2>
                {frames_html if frames_html else '<p style="color: #999;">Sin frames.</p>'}
            </div>
        </div>
        """

    noise_note = ""
    if multi_result.noise_tracks_discarded > 0:
        noise_note = (
            f'<p style="color: #888; font-size: 0.9em;">'
            f'{multi_result.noise_tracks_discarded} rastreo(s) descartado(s) por ser muy corto(s).</p>'
        )

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Multi-Animal - {video_filename}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; color: #333; background: #f5f5f5; padding: 20px;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{
            background: white; padding: 24px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
        }}
        h1 {{ color: #2c5530; margin-bottom: 8px; font-size: 1.6em; }}
        .header-info {{ color: #666; font-size: 0.9em; }}
        .header-info span {{ margin-right: 20px; }}
        .summary-banner {{
            background: #e3f2fd; border-left: 5px solid #1976d2;
            padding: 20px; border-radius: 8px; margin-bottom: 20px;
        }}
        .summary-banner h2 {{ color: #1976d2; font-size: 1.5em; margin-bottom: 4px; }}
        .result-banner {{
            padding: 16px; border-radius: 8px; margin-bottom: 16px;
        }}
        .result-banner h2 {{ font-size: 1.3em; margin-bottom: 4px; }}
        .result-banner .conf {{ font-size: 1em; color: #555; }}
        .result-banner .conf-detail {{ font-size: 0.9em; color: #777; margin-top: 4px; }}
        .stats-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 14px; margin-bottom: 20px;
        }}
        .stat-card {{
            background: white; padding: 14px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-top: 3px solid #4a7c59;
        }}
        .stat-card h3 {{ color: #2c5530; font-size: 0.9em; margin-bottom: 6px; }}
        .stat-card .big-number {{ font-size: 1.8em; font-weight: bold; color: #4a7c59; }}
        .stat-card .sub {{ color: #888; font-size: 0.8em; }}
        .section {{
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
        }}
        .section h2 {{ color: #2c5530; margin-bottom: 14px; padding-bottom: 6px; border-bottom: 2px solid #e0e0e0; }}
        .stat-item {{
            display: flex; justify-content: space-between; padding: 6px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        .stat-label {{ font-weight: 500; }}
        .stat-value {{ color: #4a7c59; font-weight: 600; }}
        .frame-card {{
            background: white; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 14px;
            overflow: hidden;
        }}
        .frame-header {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 14px; background: #f9f9f9; border-bottom: 1px solid #eee;
        }}
        .frame-num {{ font-weight: 600; color: #333; }}
        .frame-status {{
            color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.85em;
        }}
        .frame-content {{ display: flex; gap: 14px; padding: 14px; flex-wrap: wrap; }}
        .frame-image {{ flex: 1; min-width: 260px; }}
        .frame-image img {{ width: 100%; border-radius: 4px; }}
        .frame-details {{ flex: 0 0 190px; min-width: 170px; }}
        .crop-section {{ margin-bottom: 10px; }}
        .crop-img {{ max-width: 150px; border-radius: 4px; border: 2px solid #4a7c59; margin-top: 4px; }}
        .detail-label {{ font-size: 0.85em; color: #888; display: block; }}
        .detail-value {{ font-size: 0.95em; }}
        .metric-row {{ margin-bottom: 6px; }}
        .confidence-bar {{
            width: 100%; height: 8px; background: #e0e0e0; border-radius: 4px;
            margin-top: 4px; overflow: hidden;
        }}
        .confidence-fill {{ height: 100%; border-radius: 4px; }}
        .footer {{
            text-align: center; color: #999; font-size: 0.8em; padding: 20px;
        }}
        @media print {{
            body {{ background: white; }}
            .frame-card {{ page-break-inside: avoid; }}
            .section {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Reporte Multi-Animal</h1>
            <div class="header-info">
                <span>Archivo: <strong>{video_filename}</strong></span>
                <span>Dataset: <strong>{dataset_name}</strong></span>
                <span>Fecha: <strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</strong></span>
            </div>
        </div>

        <div class="summary-banner">
            <h2>Se identificaron {n_tracks} animal(es)</h2>
            <p>{multi_result.total_frames_extracted} frames extraidos |
               {multi_result.frames_with_detections} con deteccion ({det_pct:.0f}%) |
               {multi_result.frames_without_detections} sin deteccion</p>
            {noise_note}
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Frames Extraidos</h3>
                <div class="big-number">{multi_result.total_frames_extracted}</div>
                <div class="sub">Total del video</div>
            </div>
            <div class="stat-card">
                <h3>Con Deteccion</h3>
                <div class="big-number">{multi_result.frames_with_detections}</div>
                <div class="sub">{det_pct:.0f}% de los frames</div>
            </div>
            <div class="stat-card">
                <h3>Sin Deteccion</h3>
                <div class="big-number">{multi_result.frames_without_detections}</div>
                <div class="sub">Descartados</div>
            </div>
            <div class="stat-card">
                <h3>Animales Rastreados</h3>
                <div class="big-number">{n_tracks}</div>
                <div class="sub">Tracks validos</div>
            </div>
        </div>

        {tracks_html}

        <div class="footer">
            <p>Reporte generado por SRFPP - Sistema de Reconocimiento Facial y Prediccion de Peso</p>
        </div>
    </div>
</body>
</html>
    """
    return html
