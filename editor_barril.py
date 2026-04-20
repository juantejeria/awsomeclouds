"""
Editor interactivo de barril para modelos 3D.
UI web donde el usuario dibuja líneas de corte libres sobre la silueta
para definir qué es "barril" (cuerpo sin patas, cabeza, cola).

El usuario puede agregar tantas líneas como quiera.
Cada línea divide la silueta — los píxeles del lado "exterior" se descartan.
"""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template_string, jsonify, request
from ultralytics import YOLO
from pathlib import Path
from generar_modelos3d_grandes import (
    detectar_vaca, segmentar_yolo_seg, volumen_por_rebanadas, parsear_nombre
)
import json
import base64

app = Flask(__name__)

# Globals
PROJECT = Path(__file__).parent
DATASET = PROJECT / 'checkpoints' / 'Dataset Modelo 3d "grandes" '
OUTPUT = PROJECT / 'output_modelos3d_grandes' / '_editor_barril'
OUTPUT.mkdir(parents=True, exist_ok=True)

cow_model = None
seg_model = None
alturas = {}
individuos_cache = {}


def cargar_modelos():
    global cow_model, seg_model, alturas
    if cow_model is None:
        print("Cargando modelos YOLO...")
        cow_model = YOLO(str(PROJECT / "models_yolo" / "cow.pt"))
        seg_model = YOLO(str(PROJECT / "yolov8n-seg.pt"))
        with open(PROJECT / "alturas_individuos.json") as f:
            alturas = json.load(f)['alturas_cm']
        print("Modelos cargados.")


def cargar_individuo(nombre):
    """Carga la mejor foto de un individuo y genera la máscara."""
    if nombre in individuos_cache:
        return individuos_cache[nombre]

    cargar_modelos()

    ind_dir = DATASET / nombre
    fotos_dir = None
    for sub in ind_dir.iterdir():
        if sub.is_dir() and sub.name.lower().startswith('3d_modelo'):
            fotos_dir = sub
            break

    if fotos_dir is None:
        return None

    fotos = sorted([f for f in fotos_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if not fotos:
        return None

    # Usar primera foto que funcione bien
    best = None
    best_area = 0
    for foto in fotos:
        img = cv2.imread(str(foto))
        if img is None:
            continue

        bbox = detectar_vaca(img, cow_model, cow_model)
        if bbox is None:
            # fallback
            coco = YOLO(str(PROJECT / "yolov8n.pt"))
            r = coco(img, conf=0.15, classes=[19], verbose=False)
            if r and len(r[0].boxes) > 0:
                boxes = r[0].boxes
                areas = [(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]) for b in boxes]
                bbox = boxes[int(np.argmax(areas))].xyxy[0].cpu().numpy().astype(int)
            else:
                continue

        mask, contorno = segmentar_yolo_seg(img, bbox, seg_model)
        if mask is None:
            continue

        area = np.count_nonzero(mask)
        if area > best_area:
            best_area = area
            best = {
                'img': img,
                'bbox': bbox,
                'mask': mask,
                'foto': foto.name,
            }

    if best is None:
        return None

    categoria, peso, meses = parsear_nombre(nombre)
    altura = alturas.get(nombre, 120)

    result = {
        'nombre': nombre,
        'peso': peso,
        'altura': altura,
        'img': best['img'],
        'bbox': best['bbox'],
        'mask': best['mask'],
        'foto': best['foto'],
    }
    individuos_cache[nombre] = result
    return result


def img_to_base64(img_bgr, quality=85):
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode('utf-8')


def mask_to_base64(mask):
    _, buf = cv2.imencode('.png', mask)
    return base64.b64encode(buf).decode('utf-8')


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Editor de Barril - {{ nombre }}</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: monospace; background: #1a1a2e; color: #eee; }
.header { background: #16213e; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 18px; }
.header .nav { display: flex; gap: 8px; }
.header .nav a { color: #0ff; text-decoration: none; padding: 4px 12px; border: 1px solid #0ff; border-radius: 4px; font-size: 13px; }
.header .nav a:hover { background: #0ff; color: #000; }
.header .nav a.active { background: #0ff; color: #000; }
.main { display: flex; height: calc(100vh - 48px); }
.canvas-container { flex: 1; position: relative; overflow: hidden; background: #111; }
canvas { cursor: crosshair; }
.sidebar { width: 320px; background: #16213e; padding: 16px; overflow-y: auto; }
.sidebar h3 { margin-bottom: 10px; color: #0ff; }
.info-box { background: #0d1b2a; padding: 12px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; line-height: 1.6; }
.info-box .label { color: #888; }
.info-box .value { color: #0f0; font-weight: bold; }
.info-box .value.error { color: #f55; }
.btn { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-family: monospace; font-size: 13px; margin: 4px 2px; }
.btn-primary { background: #0ff; color: #000; }
.btn-danger { background: #f44; color: #fff; }
.btn-success { background: #4f4; color: #000; }
.btn:hover { opacity: 0.8; }
.cuts-list { margin-top: 10px; }
.cut-item { background: #0d1b2a; padding: 8px; margin: 4px 0; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; }
.cut-item .cut-color { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.cut-item button { background: #f44; color: #fff; border: none; padding: 2px 8px; border-radius: 3px; cursor: pointer; font-size: 11px; }
.instructions { background: #1a1a3e; padding: 10px; border-radius: 6px; margin-bottom: 12px; font-size: 12px; line-height: 1.5; color: #aaa; }
.results { margin-top: 12px; }
</style>
</head>
<body>

<div class="header">
    <h1>Editor de Barril - {{ nombre }} ({{ peso }} kg)</h1>
    <div class="nav">
        {% for ind in individuos %}
        <a href="/editor/{{ ind }}" class="{{ 'active' if ind == nombre else '' }}">{{ ind.replace('vaca_','').replace('_36','') }}</a>
        {% endfor %}
    </div>
</div>

<div class="main">
    <div class="canvas-container" id="canvasContainer">
        <canvas id="canvas"></canvas>
    </div>

    <div class="sidebar">
        <div class="instructions">
            <b>Instrucciones:</b><br>
            - Click y arrastra para dibujar una linea de corte<br>
            - Cada linea elimina los pixeles del lado exterior<br>
            - Agrega tantas lineas como necesites<br>
            - El volumen del barril se recalcula en tiempo real
        </div>

        <div class="info-box">
            <span class="label">Peso real:</span> <span class="value">{{ peso }} kg</span><br>
            <span class="label">Altura:</span> <span class="value">{{ altura }} cm</span><br>
            <span class="label">Foto:</span> <span class="label">{{ foto }}</span>
        </div>

        <h3>Volumen</h3>
        <div class="info-box" id="volInfo">
            <span class="label">Vol silueta completa:</span> <span class="value" id="volTotal">-</span> L<br>
            <span class="label">Peso silueta (x1.03):</span> <span class="value" id="pesoTotal">-</span> kg<br>
            <span class="label">Error silueta vs real:</span> <span class="value" id="errorTotal">-</span><br>
            <hr style="border-color:#333; margin:6px 0">
            <span class="label">Vol barril (recortado):</span> <span class="value" id="volBarril">-</span> L<br>
            <span class="label">Peso barril (x1.03):</span> <span class="value" id="pesoBarril">-</span> kg<br>
        </div>

        <h3>Lineas de corte</h3>
        <button class="btn btn-danger" onclick="clearAllCuts()">Borrar todas</button>
        <button class="btn btn-primary" onclick="undoLastCut()">Deshacer ultima</button>
        <div class="cuts-list" id="cutsList"></div>

        <div class="results" style="margin-top: 20px;">
            <button class="btn btn-success" onclick="guardarRecorte()">Guardar recorte</button>
            <div id="saveStatus" style="margin-top: 8px; font-size: 12px;"></div>
        </div>
    </div>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('canvasContainer');

const imgData = "data:image/jpeg;base64,{{ img_b64 }}";
const maskData = "data:image/png;base64,{{ mask_b64 }}";
const nombre = "{{ nombre }}";
const pesoReal = {{ peso }};
const bboxData = {{ bbox }};

let img = new Image();
let maskImg = new Image();
let maskCanvas, maskCtx;
let originalMask;  // Uint8Array del mask original
let cuts = [];  // [{x1,y1,x2,y2,color}]
let drawing = false;
let startX, startY;
let scale = 1;
let offsetX = 0, offsetY = 0;

const COLORS = ['#ff4444','#44ff44','#4444ff','#ffff44','#ff44ff','#44ffff','#ff8844','#8844ff','#44ff88','#ff4488'];

img.onload = function() {
    console.log("Editor v3 - knife cut mode");
    maskImg.src = maskData;
};

maskImg.onload = function() {
    // Set canvas size
    resizeCanvas();

    // Create mask canvas
    maskCanvas = document.createElement('canvas');
    maskCanvas.width = img.width;
    maskCanvas.height = img.height;
    maskCtx = maskCanvas.getContext('2d');
    maskCtx.drawImage(maskImg, 0, 0, img.width, img.height);

    // Store original mask
    let mData = maskCtx.getImageData(0, 0, img.width, img.height);
    originalMask = new Uint8Array(img.width * img.height);
    for (let i = 0; i < originalMask.length; i++) {
        originalMask[i] = mData.data[i * 4] > 128 ? 1 : 0;
    }

    draw();
    recalcVolume();
};

img.src = imgData;

function resizeCanvas() {
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    // Fit image
    let sx = canvas.width / img.width;
    let sy = canvas.height / img.height;
    scale = Math.min(sx, sy) * 0.95;
    offsetX = (canvas.width - img.width * scale) / 2;
    offsetY = (canvas.height - img.height * scale) / 2;
}

window.addEventListener('resize', () => { resizeCanvas(); draw(); });

function toImgCoords(cx, cy) {
    return [(cx - offsetX) / scale, (cy - offsetY) / scale];
}

function toCanvasCoords(ix, iy) {
    return [ix * scale + offsetX, iy * scale + offsetY];
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image
    ctx.drawImage(img, offsetX, offsetY, img.width * scale, img.height * scale);

    // Draw current mask (barrel) as green overlay
    let currentMask = computeBarrelMask();
    let overlay = ctx.createImageData(img.width, img.height);
    for (let i = 0; i < currentMask.length; i++) {
        if (currentMask[i]) {
            overlay.data[i*4] = 0;
            overlay.data[i*4+1] = 220;
            overlay.data[i*4+2] = 0;
            overlay.data[i*4+3] = 100;
        }
        // Show removed parts in red (was in original but not in barrel)
        if (originalMask[i] && !currentMask[i]) {
            overlay.data[i*4] = 255;
            overlay.data[i*4+1] = 0;
            overlay.data[i*4+2] = 0;
            overlay.data[i*4+3] = 80;
        }
    }
    // Draw overlay on temp canvas then scale
    let tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = img.width;
    tmpCanvas.height = img.height;
    let tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.putImageData(overlay, 0, 0);
    ctx.drawImage(tmpCanvas, offsetX, offsetY, img.width * scale, img.height * scale);

    // Draw cut lines
    cuts.forEach((cut, idx) => {
        let [cx1, cy1] = toCanvasCoords(cut.x1, cut.y1);
        let [cx2, cy2] = toCanvasCoords(cut.x2, cut.y2);
        ctx.beginPath();
        ctx.moveTo(cx1, cy1);
        ctx.lineTo(cx2, cy2);
        ctx.strokeStyle = cut.color;
        ctx.lineWidth = 3;
        ctx.stroke();

        // Arrow showing which side gets removed
        let mx = (cx1+cx2)/2, my = (cy1+cy2)/2;
        let dx = cx2-cx1, dy = cy2-cy1;
        let len = Math.sqrt(dx*dx+dy*dy);
        // Normal pointing to the "remove" side (right of the line direction)
        let nx = dy/len * 15, ny = -dx/len * 15;
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(mx+nx, my+ny);
        ctx.strokeStyle = cut.color;
        ctx.lineWidth = 2;
        ctx.stroke();
        // Small circle at the "remove" end
        ctx.beginPath();
        ctx.arc(mx+nx, my+ny, 4, 0, Math.PI*2);
        ctx.fillStyle = cut.color;
        ctx.fill();
    });

    // Draw current line being drawn
    if (drawing) {
        let [cx1, cy1] = toCanvasCoords(startX, startY);
        ctx.beginPath();
        ctx.moveTo(cx1, cy1);
        ctx.lineTo(currentMouseX, currentMouseY);
        ctx.strokeStyle = COLORS[cuts.length % COLORS.length];
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

let currentMouseX = 0, currentMouseY = 0;

canvas.addEventListener('mousedown', (e) => {
    let rect = canvas.getBoundingClientRect();
    let cx = e.clientX - rect.left;
    let cy = e.clientY - rect.top;
    [startX, startY] = toImgCoords(cx, cy);
    drawing = true;
});

canvas.addEventListener('mousemove', (e) => {
    let rect = canvas.getBoundingClientRect();
    currentMouseX = e.clientX - rect.left;
    currentMouseY = e.clientY - rect.top;
    if (drawing) draw();
});

canvas.addEventListener('mouseup', (e) => {
    if (!drawing) return;
    drawing = false;
    let rect = canvas.getBoundingClientRect();
    let cx = e.clientX - rect.left;
    let cy = e.clientY - rect.top;
    let [endX, endY] = toImgCoords(cx, cy);

    // Minimum line length
    let dist = Math.sqrt((endX-startX)**2 + (endY-startY)**2);
    if (dist < 10) return;

    cuts.push({
        x1: startX, y1: startY,
        x2: endX, y2: endY,
        color: COLORS[cuts.length % COLORS.length]
    });

    draw();
    recalcVolume();
    updateCutsList();
});

function computeBarrelMask() {
    // Each cut line acts as a "knife" that erases pixels along the segment.
    // After all cuts, we keep only the largest connected component (= the barrel).
    // This way cutting across a leg disconnects it, and it gets discarded.
    let mask = new Uint8Array(originalMask);

    // Step 1: Erase pixels along each cut line (with thickness)
    let cutThickness = 3;
    cuts.forEach(cut => {
        let dx = cut.x2 - cut.x1;
        let dy = cut.y2 - cut.y1;
        let len = Math.sqrt(dx*dx + dy*dy);
        if (len < 1) return;
        let steps = Math.ceil(len);
        for (let s = 0; s <= steps; s++) {
            let t = s / steps;
            let cx = Math.round(cut.x1 + dx * t);
            let cy = Math.round(cut.y1 + dy * t);
            // Erase a small area around the point
            for (let oy = -cutThickness; oy <= cutThickness; oy++) {
                for (let ox = -cutThickness; ox <= cutThickness; ox++) {
                    let px = cx + ox;
                    let py = cy + oy;
                    if (px >= 0 && px < img.width && py >= 0 && py < img.height) {
                        mask[py * img.width + px] = 0;
                    }
                }
            }
        }
    });

    if (cuts.length === 0) return mask;

    // Step 2: Find connected components and keep only the largest
    let labels = new Int32Array(img.width * img.height);
    let labelId = 0;
    let labelSizes = {};

    for (let y = 0; y < img.height; y++) {
        for (let x = 0; x < img.width; x++) {
            let idx = y * img.width + x;
            if (mask[idx] === 0 || labels[idx] !== 0) continue;
            // BFS flood fill
            labelId++;
            let queue = [idx];
            labels[idx] = labelId;
            let size = 0;
            while (queue.length > 0) {
                let ci = queue.pop();
                size++;
                let cy2 = Math.floor(ci / img.width);
                let cx2 = ci % img.width;
                // 4-connected neighbors
                let neighbors = [
                    cy2 > 0 ? ci - img.width : -1,
                    cy2 < img.height-1 ? ci + img.width : -1,
                    cx2 > 0 ? ci - 1 : -1,
                    cx2 < img.width-1 ? ci + 1 : -1
                ];
                for (let ni of neighbors) {
                    if (ni >= 0 && mask[ni] && labels[ni] === 0) {
                        labels[ni] = labelId;
                        queue.push(ni);
                    }
                }
            }
            labelSizes[labelId] = size;
        }
    }

    // Find largest component
    let maxLabel = 0;
    let maxSize = 0;
    for (let [lid, sz] of Object.entries(labelSizes)) {
        if (sz > maxSize) {
            maxSize = sz;
            maxLabel = parseInt(lid);
        }
    }

    // Keep only the largest component
    for (let i = 0; i < mask.length; i++) {
        if (mask[i] && labels[i] !== maxLabel) {
            mask[i] = 0;
        }
    }

    return mask;
}

function recalcVolume() {
    let barrelMask = computeBarrelMask();

    // Send mask to server for volume calculation
    // Encode barrel mask as run-length for efficiency
    let runs = [];
    let count = 0;
    let current = barrelMask[0];
    for (let i = 0; i < barrelMask.length; i++) {
        if (barrelMask[i] === current) {
            count++;
        } else {
            runs.push([current, count]);
            current = barrelMask[i];
            count = 1;
        }
    }
    runs.push([current, count]);

    fetch('/calcular_volumen', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            nombre: nombre,
            width: img.width,
            height: img.height,
            mask_rle: runs
        })
    })
    .then(r => r.json())
    .then(data => {
        document.getElementById('volTotal').textContent = data.vol_total.toFixed(1);
        let pesoSilueta = data.vol_total * 1.03;
        document.getElementById('pesoTotal').textContent = pesoSilueta.toFixed(1);
        let errTotal = ((pesoSilueta - pesoReal) / pesoReal * 100);
        let errTotalEl = document.getElementById('errorTotal');
        errTotalEl.textContent = (errTotal >= 0 ? '+' : '') + errTotal.toFixed(0) + '%';
        errTotalEl.className = 'value' + (Math.abs(errTotal) > 20 ? ' error' : '');

        document.getElementById('volBarril').textContent = data.vol_barril.toFixed(1);
        document.getElementById('pesoBarril').textContent = (data.vol_barril * 1.03).toFixed(1);
    });
}

function updateCutsList() {
    let html = '';
    cuts.forEach((cut, idx) => {
        html += `<div class="cut-item">
            <span><span class="cut-color" style="background:${cut.color}"></span> Linea ${idx+1}</span>
            <button onclick="removeCut(${idx})">X</button>
        </div>`;
    });
    document.getElementById('cutsList').innerHTML = html;
}

function removeCut(idx) {
    cuts.splice(idx, 1);
    draw();
    recalcVolume();
    updateCutsList();
}

function clearAllCuts() {
    cuts = [];
    draw();
    recalcVolume();
    updateCutsList();
}

function undoLastCut() {
    if (cuts.length > 0) {
        cuts.pop();
        draw();
        recalcVolume();
        updateCutsList();
    }
}

function guardarRecorte() {
    let barrelMask = computeBarrelMask();
    let runs = [];
    let count = 0;
    let current = barrelMask[0];
    for (let i = 0; i < barrelMask.length; i++) {
        if (barrelMask[i] === current) {
            count++;
        } else {
            runs.push([current, count]);
            current = barrelMask[i];
            count = 1;
        }
    }
    runs.push([current, count]);

    fetch('/guardar_recorte', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            nombre: nombre,
            cuts: cuts,
            width: img.width,
            height: img.height,
            mask_rle: runs
        })
    })
    .then(r => r.json())
    .then(data => {
        document.getElementById('saveStatus').textContent = data.message;
        document.getElementById('saveStatus').style.color = '#4f4';
    });
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    """Lista de individuos disponibles."""
    cargar_modelos()
    individuos = []
    for ind_dir in sorted(DATASET.iterdir()):
        if ind_dir.is_dir() and ind_dir.name in alturas:
            individuos.append(ind_dir.name)

    html = """
    <html><head><title>Editor de Barril</title>
    <style>
    body { font-family: monospace; background: #1a1a2e; color: #eee; padding: 40px; }
    a { color: #0ff; text-decoration: none; display: block; padding: 10px; margin: 4px 0;
        background: #16213e; border-radius: 6px; }
    a:hover { background: #0d1b2a; }
    h1 { margin-bottom: 20px; }
    </style></head><body>
    <h1>Editor de Barril - Seleccionar individuo</h1>
    """
    for ind in individuos:
        cat, peso, meses = parsear_nombre(ind)
        html += f'<a href="/editor/{ind}">{ind} ({peso} kg)</a>'
    html += "</body></html>"
    return html


@app.route('/editor/<nombre>')
def editor(nombre):
    data = cargar_individuo(nombre)
    if data is None:
        return f"No se pudo cargar {nombre}", 404

    # Lista de individuos para navegación
    individuos = sorted([d.name for d in DATASET.iterdir() if d.is_dir() and d.name in alturas])

    img_b64 = img_to_base64(data['img'])
    mask_b64 = mask_to_base64(data['mask'])
    bbox = data['bbox'].tolist()

    return render_template_string(HTML_TEMPLATE,
        nombre=nombre,
        peso=data['peso'],
        altura=data['altura'],
        foto=data['foto'],
        img_b64=img_b64,
        mask_b64=mask_b64,
        bbox=bbox,
        individuos=individuos
    )


@app.route('/calcular_volumen', methods=['POST'])
def calcular_volumen():
    req = request.json
    nombre = req['nombre']
    width = req['width']
    height = req['height']
    mask_rle = req['mask_rle']

    data = cargar_individuo(nombre)
    if data is None:
        return jsonify({'error': 'individuo no encontrado'}), 404

    # Decode RLE mask
    barrel_mask = np.zeros(width * height, dtype=np.uint8)
    idx = 0
    for val, count in mask_rle:
        if val:
            barrel_mask[idx:idx+count] = 255
        idx += count
    barrel_mask = barrel_mask.reshape(height, width)

    # Calcular escala
    bbox = data['bbox']
    bbox_h = bbox[3] - bbox[1]
    escala = data['altura'] / bbox_h

    # Volumen total (máscara original)
    vol_total, _ = volumen_por_rebanadas(data['mask'], escala)

    # Volumen barril (máscara recortada)
    vol_barril, _ = volumen_por_rebanadas(barrel_mask, escala)

    return jsonify({
        'vol_total': round(vol_total, 1),
        'vol_barril': round(vol_barril, 1),
    })


@app.route('/guardar_recorte', methods=['POST'])
def guardar_recorte():
    req = request.json
    nombre = req['nombre']
    cuts = req['cuts']
    width = req['width']
    height = req['height']
    mask_rle = req['mask_rle']

    data = cargar_individuo(nombre)
    if data is None:
        return jsonify({'error': 'individuo no encontrado'}), 404

    # Decode barrel mask
    barrel_flat = np.zeros(width * height, dtype=np.uint8)
    idx = 0
    for val, count in mask_rle:
        if val:
            barrel_flat[idx:idx+count] = 255
        idx += count
    barrel_mask = barrel_flat.reshape(height, width)

    bbox = data['bbox']
    bbox_h = bbox[3] - bbox[1]
    escala = data['altura'] / bbox_h
    vol_barril, _ = volumen_por_rebanadas(barrel_mask, escala)
    vol_total, _ = volumen_por_rebanadas(data['mask'], escala)

    # Calcular proporciones relativas al bbox
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    bbox_w = bbox_x2 - bbox_x1

    # Guardar
    recorte = {
        'individuo': nombre,
        'peso_real_kg': data['peso'],
        'altura_cm': data['altura'],
        'foto': data['foto'],
        'cuts': cuts,
        'img_width': width,
        'img_height': height,
        'bbox': bbox.tolist(),
        'vol_total_litros': round(vol_total, 1),
        'vol_barril_litros': round(vol_barril, 1),
        'peso_barril_kg': round(vol_barril * 1.03, 1),
        'ratio_barril_total': round(vol_barril / vol_total, 3) if vol_total > 0 else 0,
    }

    out_file = OUTPUT / f"{nombre}_recorte.json"
    with open(out_file, 'w') as f:
        json.dump(recorte, f, indent=2, ensure_ascii=False)

    # Guardar imagen del barril
    img_rgb = data['img'].copy()
    dark = (img_rgb * 0.3).astype(np.uint8)
    dark[barrel_mask > 0] = img_rgb[barrel_mask > 0]
    contours, _ = cv2.findContours(barrel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dark, contours, -1, (0, 255, 255), 2)
    cv2.imwrite(str(OUTPUT / f"{nombre}_barril.jpg"), dark)

    return jsonify({
        'message': f'Guardado: {out_file.name} (vol={vol_barril:.0f}L, peso={vol_barril*1.03:.0f}kg)',
    })


if __name__ == '__main__':
    print("\n  Editor de Barril")
    print("  Abrir en: http://localhost:5050\n")
    app.run(host='0.0.0.0', port=5050, debug=False)
