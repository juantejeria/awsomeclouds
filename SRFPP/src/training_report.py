"""
Generación de reportes de entrenamiento con visualizaciones de detecciones YOLO.
"""

from __future__ import annotations

import base64
import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .face_detection import _get_yolo_model
from .io_utils import read_json


def analyze_training_images(
    data_dir: Path,
    artifacts_dir: Path,
    max_samples_per_class: int = 20,
) -> dict[str, Any]:
    """
    Analiza las imágenes usadas en el entrenamiento y detecta animales con YOLO.
    
    Args:
        data_dir: Directorio del dataset (ImageFolder structure)
        artifacts_dir: Directorio de artifacts donde guardar el reporte
        max_samples_per_class: Máximo de muestras a analizar por clase
    
    Returns:
        Diccionario con estadísticas y detecciones
    """
    from torchvision import datasets
    
    # Cargar dataset sin transforms para obtener rutas originales
    base_ds = datasets.ImageFolder(str(data_dir), transform=None)
    
    # Cargar YOLO
    yolo_model = _get_yolo_model()
    
    # Clases de animales en COCO
    animal_classes = {
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 
        18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
        22: "zebra", 23: "giraffe"
    }
    
    stats = {
        "total_images": len(base_ds),
        "total_classes": len(base_ds.classes),
        "images_by_class": {},
        "detections_by_class": defaultdict(int),
        "images_with_detections": [],
        "images_without_detections": [],
        "detection_details": [],
    }
    
    # Analizar imágenes por clase
    for class_idx, class_name in enumerate(base_ds.classes):
        class_images = [
            (path, idx) 
            for idx, (path, label) in enumerate(base_ds.samples) 
            if label == class_idx
        ]
        
        stats["images_by_class"][class_name] = len(class_images)
        
        # Analizar muestras de esta clase
        samples_to_analyze = class_images[:max_samples_per_class]
        
        for img_path, dataset_idx in samples_to_analyze:
            img_path = Path(img_path)
            
            try:
                # Cargar imagen
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                
                # Detectar con YOLO
                results = yolo_model(img_array, conf=0.3, verbose=False)
                
                detections = []
                has_animal = False
                
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for box, conf, cls in zip(boxes, confidences, classes):
                            if cls in animal_classes:
                                has_animal = True
                                animal_name = animal_classes[cls]
                                stats["detections_by_class"][animal_name] += 1
                                
                                detections.append({
                                    "animal": animal_name,
                                    "confidence": float(conf),
                                    "bbox": [float(x) for x in box],  # [x1, y1, x2, y2]
                                    "center": [
                                        float((box[0] + box[2]) / 2),  # cx
                                        float((box[1] + box[3]) / 2),  # cy
                                    ],
                                })
                
                detection_info = {
                    "image_path": str(img_path.relative_to(data_dir)),
                    "class": class_name,
                    "detections": detections,
                    "has_detection": has_animal,
                }
                
                stats["detection_details"].append(detection_info)
                
                if has_animal:
                    stats["images_with_detections"].append(detection_info)
                else:
                    stats["images_without_detections"].append(detection_info)
                    
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                continue
    
    return stats


def draw_detections_on_image(
    image: Image.Image,
    detections: list[dict[str, Any]],
    show_bbox: bool = True,
    show_center: bool = True,
    center_color: tuple[int, int, int] = (255, 0, 0),  # Rojo
    bbox_color: tuple[int, int, int] = (0, 255, 0),  # Verde
) -> Image.Image:
    """
    Dibuja las detecciones de YOLO sobre una imagen.
    
    Args:
        image: Imagen PIL
        detections: Lista de detecciones con 'bbox', 'center', 'animal', 'confidence'
        show_bbox: Si True, dibuja bounding boxes
        show_center: Si True, dibuja punto en el centro
        center_color: Color RGB para el punto central
        bbox_color: Color RGB para el bounding box
    
    Returns:
        Imagen con las detecciones dibujadas
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Intentar cargar fuente, si falla usar default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    for det in detections:
        bbox = det["bbox"]
        center = det["center"]
        animal = det["animal"]
        conf = det["confidence"]
        
        x1, y1, x2, y2 = bbox
        
        # Dibujar bounding box
        if show_bbox:
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=bbox_color,
                width=3,
            )
        
        # Dibujar punto central (más grande para visibilidad)
        if show_center:
            radius = 8
            cx, cy = center
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=center_color,
                outline=(255, 255, 255),
                width=2,
            )
        
        # Etiqueta con nombre del animal y confianza
        label = f"{animal} {conf:.2f}"
        label_bbox = draw.textbbox((x1, y1 - 20), label, font=font) if font else None
        
        if label_bbox:
            # Fondo semitransparente para la etiqueta
            draw.rectangle(
                [x1, y1 - 25, label_bbox[2] + 5, y1],
                fill=(0, 0, 0, 180),
            )
            draw.text(
                (x1 + 2, y1 - 23),
                label,
                fill=(255, 255, 255),
                font=font,
            )
    
    return img


def generate_training_report(
    data_dir: Path,
    artifacts_dir: Path,
    max_samples_per_class: int = 20,
) -> Path:
    """
    Genera un reporte HTML completo del entrenamiento con visualizaciones.
    
    Args:
        data_dir: Directorio del dataset
        artifacts_dir: Directorio de artifacts
        max_samples_per_class: Máximo muestras a analizar por clase
    
    Returns:
        Ruta al archivo HTML generado
    """
    print("Analizando imágenes del entrenamiento...")
    stats = analyze_training_images(data_dir, artifacts_dir, max_samples_per_class)
    
    # Crear directorio para imágenes del reporte
    report_images_dir = artifacts_dir / "report_images"
    report_images_dir.mkdir(exist_ok=True)
    
    # Cargar configuración del entrenamiento
    config = read_json(artifacts_dir / "config.json")
    classes = read_json(artifacts_dir / "classes.json")
    
    # Generar visualizaciones
    print("Generando visualizaciones...")
    visualization_paths = []
    
    # Procesar imágenes con detecciones
    for det_info in stats["images_with_detections"][:50]:  # Limitar a 50 para no hacer el reporte muy pesado
        img_path = data_dir / det_info["image_path"]
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_with_detections = draw_detections_on_image(
                img,
                det_info["detections"],
                show_bbox=True,
                show_center=True,
            )
            
            # Guardar imagen visualizada (para referencia local)
            vis_filename = f"vis_{Path(det_info['image_path']).name}"
            vis_path = report_images_dir / vis_filename
            img_with_detections.save(vis_path, quality=90)
            
            # Convertir imagen a base64 para incluir en HTML
            img_buffer = io.BytesIO()
            img_with_detections.save(img_buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            img_data_uri = f"data:image/jpeg;base64,{img_base64}"
            
            visualization_paths.append({
                "path": f"report_images/{vis_filename}",  # Mantener para referencia
                "data_uri": img_data_uri,  # Base64 para HTML autónomo
                "original_path": det_info["image_path"],
                "class": det_info["class"],
                "detections": det_info["detections"],
            })
        except Exception as e:
            print(f"Error generando visualización para {img_path}: {e}")
            continue
    
    # Generar HTML
    html_content = _generate_html_report(stats, config, classes, visualization_paths, artifacts_dir)
    
    # Guardar HTML
    report_path = artifacts_dir / "training_report.html"
    report_path.write_text(html_content, encoding='utf-8')
    
    # Guardar también los datos en JSON para referencia
    json_path = artifacts_dir / "training_report_data.json"
    json_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    
    print(f"Reporte generado: {report_path}")
    return report_path


def _generate_html_report(
    stats: dict[str, Any],
    config: dict[str, Any],
    classes: list[str],
    visualization_paths: list[dict[str, Any]],
    artifacts_dir: Path,
) -> str:
    """Genera el contenido HTML del reporte."""
    
    # Calcular porcentajes
    total_analyzed = len(stats["detection_details"])
    with_detections = len(stats["images_with_detections"])
    without_detections = len(stats["images_without_detections"])
    detection_rate = (with_detections / total_analyzed * 100) if total_analyzed > 0 else 0
    
    # Generar sección de estadísticas por clase
    class_stats_html = ""
    for class_name in classes:
        count = stats["images_by_class"].get(class_name, 0)
        class_stats_html += f"""
        <div class="stat-item">
            <span class="stat-label">{class_name}:</span>
            <span class="stat-value">{count} imágenes</span>
        </div>
        """
    
    # Generar sección de detecciones por tipo de animal
    detection_stats_html = ""
    for animal, count in sorted(stats["detections_by_class"].items(), key=lambda x: -x[1]):
        detection_stats_html += f"""
        <div class="stat-item">
            <span class="stat-label">{animal}:</span>
            <span class="stat-value">{count} detecciones</span>
        </div>
        """
    
    # Generar grid de visualizaciones
    visualizations_html = ""
    for vis in visualization_paths:
        detections_text = ", ".join([
            f"{d['animal']} ({d['confidence']:.2f})" 
            for d in vis["detections"]
        ])
        # Usar data URI (base64) para que el HTML sea autónomo
        img_src = vis.get("data_uri", vis.get("path", ""))
        visualizations_html += f"""
        <div class="vis-item">
            <img src="{img_src}" alt="{vis['original_path']}" />
            <div class="vis-info">
                <strong>Clase:</strong> {vis['class']}<br>
                <strong>Detecciones:</strong> {detections_text}<br>
                <strong>Archivo:</strong> {vis['original_path']}
            </div>
        </div>
        """
    
    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Entrenamiento - Reconocimiento Facial de Vacas</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c5530;
            margin-bottom: 10px;
            border-bottom: 3px solid #4a7c59;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #4a7c59;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .header-info {{
            background: #f0f7f0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .header-info p {{
            margin: 5px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4a7c59;
        }}
        .stat-card h3 {{
            color: #2c5530;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        .stat-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .stat-item:last-child {{
            border-bottom: none;
        }}
        .stat-label {{
            font-weight: 500;
        }}
        .stat-value {{
            color: #4a7c59;
            font-weight: bold;
        }}
        .visualizations-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .vis-item {{
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            background: white;
        }}
        .vis-item img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .vis-info {{
            padding: 10px;
            font-size: 0.9em;
            background: #f9f9f9;
        }}
        .vis-info strong {{
            color: #2c5530;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🐄 Reporte de Entrenamiento</h1>
        
        <div class="header-info">
            <p><strong>Fecha de generación:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Establecimiento:</strong> {artifacts_dir.name}</p>
            <p><strong>Total de imágenes en dataset:</strong> {stats['total_images']}</p>
            <p><strong>Clases entrenadas:</strong> {stats['total_classes']}</p>
        </div>
        
        <h2>📊 Estadísticas Generales</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Análisis de Detecciones</h3>
                <div class="stat-item">
                    <span class="stat-label">Imágenes analizadas:</span>
                    <span class="stat-value">{total_analyzed}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Con detecciones:</span>
                    <span class="stat-value">{with_detections}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Sin detecciones:</span>
                    <span class="stat-value">{without_detections}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Tasa de detección:</span>
                    <span class="stat-value">{detection_rate:.1f}%</span>
                </div>
            </div>
            
            <div class="stat-card">
                <h3>Imágenes por Clase</h3>
                {class_stats_html}
            </div>
            
            <div class="stat-card">
                <h3>Detecciones por Tipo de Animal</h3>
                {detection_stats_html if detection_stats_html else '<p>No se detectaron animales</p>'}
            </div>
        </div>
        
        <h2>⚙️ Configuración del Entrenamiento</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Hiperparámetros</h3>
                <div class="stat-item">
                    <span class="stat-label">Épocas:</span>
                    <span class="stat-value">{config.get('epochs', 'N/A')}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Batch size:</span>
                    <span class="stat-value">{config.get('batch_size', 'N/A')}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Learning rate:</span>
                    <span class="stat-value">{config.get('lr', 'N/A')}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Tamaño de imagen:</span>
                    <span class="stat-value">{config.get('img_size', 'N/A')}x{config.get('img_size', 'N/A')}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Fracción de validación:</span>
                    <span class="stat-value">{config.get('val_frac', 'N/A')}</span>
                </div>
            </div>
            
            <div class="stat-card">
                <h3>Resultados</h3>
                <div class="stat-item">
                    <span class="stat-label">Mejor accuracy (val):</span>
                    <span class="stat-value">{config.get('best_val_acc', 0):.3f}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Tiempo de entrenamiento:</span>
                    <span class="stat-value">{config.get('train_seconds', 0):.1f}s</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Dispositivo usado:</span>
                    <span class="stat-value">{config.get('device_used', 'N/A')}</span>
                </div>
            </div>
        </div>
        
        <h2>🔍 Visualizaciones de Detecciones YOLO</h2>
        <div class="highlight">
            <strong>Nota:</strong> Las imágenes muestran las detecciones de YOLO con:
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li><strong>Punto rojo:</strong> Centro de la detección (el "ojo" de YOLO)</li>
                <li><strong>Rectángulo verde:</strong> Bounding box del animal detectado</li>
                <li><strong>Etiqueta:</strong> Tipo de animal y confianza de la detección</li>
            </ul>
        </div>
        
        <div class="visualizations-grid">
            {visualizations_html if visualizations_html else '<p>No hay visualizaciones disponibles</p>'}
        </div>
        
        <div class="footer">
            <p>Reporte generado automáticamente por el sistema de reconocimiento facial de vacas</p>
            <p>Total de visualizaciones: {len(visualization_paths)}</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html

