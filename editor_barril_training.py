"""
Editor de Barril - Training Mode.
UI para anotar el barril en frames extraídos de videos.
Funciones: ver galería, editar (recortar con líneas), guardar, descartar.

Nota: "Descartar" borra el frame del índice y elimina sus archivos del disco
(con doble confirmación en la UI). No existe estado 'discarded'.
"""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template_string, jsonify, request, send_file
from pathlib import Path
from generar_modelos3d_grandes import volumen_por_rebanadas
import json
import base64
import threading

app = Flask(__name__)


@app.after_request
def add_ngrok_header(response):
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response


PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / 'output_modelos3d_grandes' / '_barril_training'
INDEX_FILE = DATA_DIR / 'frames_index.json'

# Datasets (campo 'source') que se MUESTRAN en la galería del editor. Solo filtra
# la vista; load_index() sigue devolviendo TODO (save/validate/discard reescriben
# el índice completo, así que no se debe filtrar ahí). Para mostrar más, agregar acá.
VISIBLE_SOURCES = {'6mayo', '12junio', '14mayo', '20mayo'}


def visible_frames():
    return [f for f in load_index() if f.get('source') in VISIBLE_SOURCES]


def load_index():
    with open(INDEX_FILE) as f:
        return json.load(f)


_index_lock = threading.RLock()


def save_index(data):
    # Escritura atómica + lock para evitar corrupción por peticiones concurrentes
    # (Flask corre con threaded=True). Se escribe a un temporal y luego os.replace.
    with _index_lock:
        tmp = str(INDEX_FILE) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, INDEX_FILE)


def img_to_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ═══════════════════════════════════════
# GALERÍA
# ═══════════════════════════════════════

GALLERY_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Barrel Training - Galeria</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:monospace; background:#1a1a2e; color:#eee; }
.header { background:#16213e; padding:12px 20px; display:flex; justify-content:space-between; align-items:center; }
.header h1 { font-size:16px; }
.stats { display:flex; gap:15px; font-size:13px; }
.stats .stat { padding:4px 10px; border-radius:4px; }
.stat-pending { background:#e67e22; color:#000; }
.stat-edited { background:#27ae60; color:#fff; }
.stat-validated { background:#9b59b6; color:#fff; }
.filters { background:#0d1b2a; padding:10px 20px; display:flex; gap:10px; align-items:center; }
.filters input, .filters button { padding:6px 12px; border-radius:4px; border:1px solid #444; background:#16213e; color:#eee; font-family:monospace; cursor:pointer; }
.filters input { cursor:text; min-width:220px; }
.filters button:hover { background:#0ff; color:#000; }
.gallery { display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); gap:10px; padding:15px; }
.card { background:#16213e; border-radius:8px; padding:12px; cursor:pointer; transition:transform 0.15s, background 0.15s; border:1px solid #243; }
.card:hover { transform:scale(1.02); background:#1d2b4a; border-color:#0ff; }
.card .name { font-size:13px; font-weight:bold; color:#0ff; word-break:break-all; margin-bottom:8px; }
.card .bar { height:6px; border-radius:3px; background:#0d1b2a; overflow:hidden; margin-bottom:8px; }
.card .bar > i { display:block; height:100%; background:#9b59b6; }
.chips { display:flex; flex-wrap:wrap; gap:5px; font-size:11px; }
.chip { padding:2px 7px; border-radius:3px; font-weight:bold; }
.status-pending, .chip-pending { background:#e67e22; color:#000; }
.status-edited, .chip-edited { background:#27ae60; color:#fff; }
.status-validated, .chip-validated { background:#9b59b6; color:#fff; }
.chip-cruz { background:#ffd700; color:#000; }
.chip-anca { background:#ff44ff; color:#000; }
.chip-total { background:#0d1b2a; color:#aaa; }
</style>
</head>
<body>
<div class="header">
    <h1>Barrel Training - {{ groups|length }} individuos / {{ total }} frames</h1>
    <div class="stats">
        <span class="stat stat-pending">Pendientes: {{ pending }}</span>
        <span class="stat stat-edited">Editados: {{ edited }}</span>
        <span class="stat stat-validated">Validados: {{ validated }}</span>
    </div>
</div>
<div class="filters">
    <label>Buscar:</label>
    <input id="search" type="text" placeholder="nombre del individuo..." oninput="applyFilter()">
    <button onclick="window.location.reload()">Refrescar</button>
</div>
<div class="gallery" id="gallery">
    {% for g in groups %}
    <div class="card" data-name="{{ g.individuo|lower }}" onclick="window.location='/individuo?ind={{ g.individuo|urlencode }}'">
        <div class="name">{{ g.individuo }}</div>
        <div class="bar"><i style="width:{{ (100*g.validated/g.total)|round(0,'floor') }}%"></i></div>
        <div class="chips">
            <span class="chip chip-total">{{ g.total }} frames</span>
            {% if g.pending %}<span class="chip chip-pending">⏳ {{ g.pending }}</span>{% endif %}
            {% if g.edited %}<span class="chip chip-edited">✎ {{ g.edited }}</span>{% endif %}
            {% if g.validated %}<span class="chip chip-validated">✓ {{ g.validated }}</span>{% endif %}
            <span class="chip chip-cruz">✛ {{ g.cruz }}</span>
            <span class="chip chip-anca">✛ {{ g.anca }}</span>
        </div>
    </div>
    {% endfor %}
</div>
<script>
function applyFilter() {
    let q = document.getElementById('search').value.toLowerCase().trim();
    document.querySelectorAll('.card').forEach(c => {
        c.style.display = (!q || c.dataset.name.includes(q)) ? '' : 'none';
    });
}
</script>
</body>
</html>
"""

# ═══════════════════════════════════════
# FRAMES DE UN INDIVIDUO (lista, sin miniatura)
# ═══════════════════════════════════════

INDIVIDUO_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>{{ individuo }} - frames</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:monospace; background:#1a1a2e; color:#eee; }
.header { background:#16213e; padding:12px 20px; display:flex; justify-content:space-between; align-items:center; gap:12px; }
.header h1 { font-size:15px; color:#0ff; word-break:break-all; }
.header a { padding:6px 14px; border:1px solid #444; border-radius:4px; background:#16213e; color:#eee; text-decoration:none; font-size:12px; }
.header a:hover { background:#0ff; color:#000; }
.stats { display:flex; gap:10px; font-size:12px; }
.chip { padding:3px 9px; border-radius:3px; font-weight:bold; }
.chip-pending { background:#e67e22; color:#000; }
.chip-edited { background:#27ae60; color:#fff; }
.chip-validated { background:#9b59b6; color:#fff; }
.chip-cruz { background:#ffd700; color:#000; }
.chip-anca { background:#ff44ff; color:#000; }
.list { padding:15px; display:flex; flex-direction:column; gap:6px; max-width:760px; }
.row { background:#16213e; border-radius:6px; padding:10px 14px; cursor:pointer; display:flex; justify-content:space-between; align-items:center; border:1px solid #243; transition:background 0.12s; }
.row:hover { background:#1d2b4a; border-color:#0ff; }
.row .fid { font-size:12px; }
.row .badges { display:flex; gap:6px; align-items:center; }
.status-pending { background:#e67e22; color:#000; }
.status-edited { background:#27ae60; color:#fff; }
.status-validated { background:#9b59b6; color:#fff; }
.status { padding:2px 8px; border-radius:3px; font-size:11px; font-weight:bold; }
</style>
</head>
<body>
<div class="header">
    <h1>{{ individuo }}</h1>
    <div class="stats">
        <span class="chip chip-pending">⏳ {{ pending }}</span>
        <span class="chip chip-edited">✎ {{ edited }}</span>
        <span class="chip chip-validated">✓ {{ validated }}</span>
        <span class="chip chip-cruz">✛ {{ cruz }}</span>
        <span class="chip chip-anca">✛ {{ anca }}</span>
    </div>
    <a href="/gallery">← Individuos</a>
</div>
<div class="list">
    {% for f in frames %}
    <div class="row" onclick="window.location='/edit/{{ f.id }}'">
        <span class="fid">Frame #{{ f.frame_idx }}</span>
        <span class="badges">
            {% if f.cruz %}<span style="color:#ffd700;font-weight:bold;">✛</span>{% endif %}
            {% if f.anca %}<span style="color:#ff44ff;font-weight:bold;">✛</span>{% endif %}
            <span class="status status-{{ f.status }}">{{ f.status }}</span>
        </span>
    </div>
    {% endfor %}
</div>
</body>
</html>
"""

# ═══════════════════════════════════════
# EDITOR
# ═══════════════════════════════════════

EDITOR_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Editar - {{ frame.id }}</title>
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:monospace; background:#1a1a2e; color:#eee; }
.header { background:#16213e; padding:10px 20px; display:flex; justify-content:space-between; align-items:center; }
.header h1 { font-size:14px; }
.nav-btns { display:flex; gap:6px; }
.nav-btns a, .nav-btns button { padding:6px 14px; border:1px solid #444; border-radius:4px; background:#16213e; color:#eee; text-decoration:none; font-family:monospace; font-size:12px; cursor:pointer; }
.nav-btns a:hover, .nav-btns button:hover { background:#0ff; color:#000; }
.btn-save { background:#27ae60 !important; color:#fff !important; border-color:#27ae60 !important; }
.btn-discard { background:#e74c3c !important; color:#fff !important; border-color:#e74c3c !important; }
.main { display:flex; height:calc(100vh - 44px); }
.canvas-container { flex:1; position:relative; overflow:hidden; background:#111; }
canvas { cursor:crosshair; }
.sidebar { width:280px; background:#16213e; padding:12px; overflow-y:auto; }
.sidebar h3 { margin:8px 0 6px; color:#0ff; font-size:13px; }
.info-box { background:#0d1b2a; padding:8px; border-radius:6px; margin-bottom:8px; font-size:12px; line-height:1.5; }
.cuts-list { margin-top:6px; }
.cut-item { background:#0d1b2a; padding:5px 8px; margin:3px 0; border-radius:4px; display:flex; justify-content:space-between; align-items:center; font-size:11px; }
.cut-item button { background:#f44; color:#fff; border:none; padding:2px 6px; border-radius:3px; cursor:pointer; }
.instructions { background:#1a1a3e; padding:8px; border-radius:6px; margin-bottom:8px; font-size:11px; line-height:1.4; color:#aaa; }
.mode-btn { padding:6px 12px; border:2px solid #444; border-radius:4px; background:#16213e; color:#eee; font-family:monospace; font-size:12px; cursor:pointer; margin:2px; }
.mode-btn.active { border-color:#0ff; background:#0a3a3a; color:#0ff; }
.mode-btn:hover { background:#1a3a4e; }
.brush-controls { display:flex; align-items:center; gap:8px; margin:6px 0; }
.brush-controls input[type=range] { flex:1; accent-color:#0ff; }
.brush-controls span { font-size:11px; color:#888; min-width:30px; }
</style>
</head>
<body>
<div class="header">
    <h1>{{ frame.individuo }} - Frame #{{ frame.frame_idx }} [{{ frame.status }}]</h1>
    <div class="nav-btns">
        <a href="/individuo?ind={{ frame.individuo|urlencode }}">{{ frame.individuo }}</a>
        <a href="/gallery">Individuos</a>
        {% if prev_id %}<a href="/edit/{{ prev_id }}">← Anterior</a>{% endif %}
        {% if next_id %}<a href="/edit/{{ next_id }}">Siguiente →</a>{% endif %}
        <button class="btn-save" onclick="guardar()">Guardar</button>
        <button style="background:#9b59b6 !important;color:#fff !important;border-color:#9b59b6 !important;" onclick="validar()">Validar</button>
        <button class="btn-discard" onclick="descartar()">Descartar</button>
        <button onclick="clearAllCuts()">Limpiar</button>
        <button onclick="undoLastCut()">Deshacer</button>
        <button id="btnOverlay" onclick="toggleOverlay()" style="background:#555 !important;color:#fff !important;">Silueta: OFF</button>
        <button id="btnPred" onclick="togglePred()" style="background:#555 !important;color:#fff !important;">Pred: OFF</button>
        <span style="font-size:11px;color:#888;margin-left:8px;">
            <span style="color:#0c0;">■</span>Barrel
            <span style="color:#f00;">■</span>Descartado
            <span style="color:#35f;">■</span>Modelo
            <span style="color:#ffd700;">✛</span>Cruz
            <span style="color:#ff44ff;">✛</span>Anca
        </span>
    </div>
</div>
<div class="main">
    <div class="canvas-container" id="canvasContainer">
        <canvas id="canvas"></canvas>
    </div>
    <div class="sidebar">
        <div class="instructions">
            <b>Modo Corte:</b> Click y arrastra para dibujar lineas de corte. Las partes desconectadas se descartan.<br>
            <b>Modo Pincel+:</b> Pinta para AGREGAR area a la mascara.<br>
            <b>Modo Pincel-:</b> Pinta para BORRAR area de la mascara.<br>
            <b>Modo Punto Cruz:</b> Click para marcar el punto de la cruz (1 por frame).<br>
            <b>Modo Punto Anca:</b> Click para marcar el punto del anca (1 por frame).<br>
            Scroll o slider para cambiar tamano del pincel.
        </div>
        <h3>Modo</h3>
        <div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:6px;">
            <button class="mode-btn active" id="modeCut" onclick="setMode('cut')">Corte</button>
            <button class="mode-btn" id="modeBrushAdd" onclick="setMode('brush-add')">Pincel +</button>
            <button class="mode-btn" id="modeBrushErase" onclick="setMode('brush-erase')">Pincel -</button>
            <button class="mode-btn" id="modeCruz" onclick="setMode('cruz')" style="border-color:#ffd700;">Punto Cruz</button>
            <button class="mode-btn" id="modeAnca" onclick="setMode('anca')" style="border-color:#ff44ff;">Punto Anca</button>
        </div>
        <div class="brush-controls" id="brushControls" style="display:none;">
            <span>Pincel:</span>
            <input type="range" id="brushSize" min="2" max="60" value="15" oninput="brushRadius=parseInt(this.value);document.getElementById('brushSizeLabel').textContent=this.value+'px';draw();">
            <span id="brushSizeLabel">15px</span>
        </div>
        <div class="info-box">
            <span style="color:#888">Individuo:</span> {{ frame.individuo }}<br>
            <span style="color:#888">Frame:</span> {{ frame.frame_idx }}<br>
            <span style="color:#888">Status:</span> <span id="statusLabel">{{ frame.status }}</span>
        </div>
        <h3>Cortes (<span id="cutCount">0</span>)</h3>
        <div class="cuts-list" id="cutsList"></div>
        <h3 style="color:#ffd700;">Punto de Cruz</h3>
        <div class="info-box" id="cruzInfo" style="border:1px solid #ffd700;">Sin marcar</div>
        <button class="mode-btn" onclick="clearCruz()" style="border-color:#e74c3c;">Borrar punto</button>
        <h3 style="color:#ff44ff;">Punto de Anca</h3>
        <div class="info-box" id="ancaInfo" style="border:1px solid #ff44ff;">Sin marcar</div>
        <button class="mode-btn" onclick="clearAnca()" style="border-color:#e74c3c;">Borrar punto</button>
    </div>
</div>
<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('canvasContainer');
const frameId = "{{ frame.id }}";

let img = new Image();
let maskImg = new Image();
let predImg = new Image();
let originalMask;
let predMask;
let showPred = false;
let showOverlay = false;
let cuts = {{ cuts_json|safe }};
let cruzPoint = {{ cruz_json|safe }};  // {x, y} en coords de imagen original, o null
let ancaPoint = {{ anca_json|safe }};  // {x, y} en coords de imagen original, o null
let drawing = false;
let startX, startY, currentMouseX = 0, currentMouseY = 0;
let scale = 1, offsetX = 0, offsetY = 0;
let editMode = 'cut';  // 'cut', 'brush-add', 'brush-erase'
let brushRadius = 15;
let brushMask = null;  // manual paint edits layer
let brushHistory = []; // for undo of brush strokes

const COLORS = ['#ff4444','#44ff44','#4444ff','#ffff44','#ff44ff','#44ffff','#ff8844','#8844ff'];

// Asignar colores a cortes cargados que no los tengan
cuts.forEach((c, i) => { if (!c.color) c.color = COLORS[i % COLORS.length]; });

img.onload = function() {
    console.log("img loaded:", img.width, "x", img.height);
    maskImg.src = "data:image/png;base64,{{ mask_b64 }}";
};
img.onerror = function() { console.error("ERROR loading img"); };
maskImg.onerror = function() { console.error("ERROR loading mask"); };
maskImg.onload = function() {
    console.log("mask loaded:", maskImg.width, "x", maskImg.height);
    console.log("cuts loaded:", cuts.length);
    resizeCanvas();
    let mc = document.createElement('canvas');
    mc.width = img.width; mc.height = img.height;
    let mctx = mc.getContext('2d');
    mctx.drawImage(maskImg, 0, 0, img.width, img.height);
    let md = mctx.getImageData(0, 0, img.width, img.height);
    originalMask = new Uint8Array(img.width * img.height);
    for (let i = 0; i < originalMask.length; i++) originalMask[i] = md.data[i*4] > 128 ? 1 : 0;
    // Restore brush edits from saved RLE
    let savedBrushRle = {{ brush_rle_json|safe }};
    if (savedBrushRle && savedBrushRle.length > 0) {
        initBrushMask();
        let bi = 0;
        for (let [val, count] of savedBrushRle) {
            for (let j = 0; j < count && bi < brushMask.length; j++, bi++) brushMask[bi] = val;
        }
    }
    // Cargar predicción del modelo
    predImg.onload = function() {
        let pc = document.createElement('canvas');
        pc.width = img.width; pc.height = img.height;
        let pctx = pc.getContext('2d');
        pctx.drawImage(predImg, 0, 0, img.width, img.height);
        let pd = pctx.getImageData(0, 0, img.width, img.height);
        predMask = new Uint8Array(img.width * img.height);
        for (let i = 0; i < predMask.length; i++) predMask[i] = pd.data[i*4] > 128 ? 1 : 0;
        draw();
        updateCutsList();
        updateCruzInfo();
        updateAncaInfo();
    };
    predImg.onerror = function() {
        predMask = null;
        draw();
        updateCutsList();
        updateCruzInfo();
        updateAncaInfo();
    };
    predImg.src = "data:image/png;base64,{{ pred_b64 }}";
};
img.src = "data:image/jpeg;base64,{{ img_b64 }}";

function resizeCanvas() {
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    let sx = canvas.width / img.width, sy = canvas.height / img.height;
    scale = Math.min(sx, sy) * 0.95;
    offsetX = (canvas.width - img.width * scale) / 2;
    offsetY = (canvas.height - img.height * scale) / 2;
}
window.addEventListener('resize', () => { resizeCanvas(); draw(); });

function toImgCoords(cx, cy) { return [(cx-offsetX)/scale, (cy-offsetY)/scale]; }
function toCanvasCoords(ix, iy) { return [ix*scale+offsetX, iy*scale+offsetY]; }

function setMode(mode) {
    editMode = mode;
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    let btnId = mode==='cut'?'modeCut':mode==='brush-add'?'modeBrushAdd':mode==='brush-erase'?'modeBrushErase':mode==='cruz'?'modeCruz':'modeAnca';
    document.getElementById(btnId).classList.add('active');
    document.getElementById('brushControls').style.display = (mode==='brush-add'||mode==='brush-erase')?'flex':'none';
    canvas.style.cursor = (mode==='cut'||mode==='cruz'||mode==='anca')?'crosshair':'none';
    draw();
}

function updateCruzInfo() {
    let el = document.getElementById('cruzInfo');
    if (cruzPoint) el.innerHTML = '<span style="color:#ffd700;">✛</span> x=' + Math.round(cruzPoint.x) + ', y=' + Math.round(cruzPoint.y);
    else el.textContent = 'Sin marcar';
}
function clearCruz() { cruzPoint = null; updateCruzInfo(); draw(); }

function updateAncaInfo() {
    let el = document.getElementById('ancaInfo');
    if (ancaPoint) el.innerHTML = '<span style="color:#ff44ff;">✛</span> x=' + Math.round(ancaPoint.x) + ', y=' + Math.round(ancaPoint.y);
    else el.textContent = 'Sin marcar';
}
function clearAnca() { ancaPoint = null; updateAncaInfo(); draw(); }

function initBrushMask() {
    if (!brushMask || brushMask.length !== img.width * img.height) {
        brushMask = new Int8Array(img.width * img.height); // 0=no edit, 1=add, -1=erase
    }
}

function paintBrush(imgX, imgY, value) {
    initBrushMask();
    let r = brushRadius;
    let painted = [];
    for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
            if (dx*dx + dy*dy > r*r) continue;
            let px = Math.round(imgX + dx), py = Math.round(imgY + dy);
            if (px >= 0 && px < img.width && py >= 0 && py < img.height) {
                let idx = py * img.width + px;
                if (brushMask[idx] !== value) {
                    painted.push({idx: idx, prev: brushMask[idx]});
                    brushMask[idx] = value;
                }
            }
        }
    }
    return painted;
}

let currentStroke = []; // pixels changed in current drag

function computeBarrelMask() {
    let mask = new Uint8Array(originalMask);
    // Apply brush edits on top of original mask
    if (brushMask) {
        for (let i = 0; i < mask.length; i++) {
            if (brushMask[i] === 1) mask[i] = 1;       // brush-add
            else if (brushMask[i] === -1) mask[i] = 0;  // brush-erase
        }
    }
    // Apply cut lines
    let cutThickness = 3;
    cuts.forEach(cut => {
        let dx = cut.x2-cut.x1, dy = cut.y2-cut.y1;
        let len = Math.sqrt(dx*dx+dy*dy);
        if (len < 1) return;
        let steps = Math.ceil(len);
        for (let s = 0; s <= steps; s++) {
            let t = s/steps;
            let cx = Math.round(cut.x1+dx*t), cy = Math.round(cut.y1+dy*t);
            for (let oy = -cutThickness; oy <= cutThickness; oy++)
                for (let ox = -cutThickness; ox <= cutThickness; ox++) {
                    let px=cx+ox, py=cy+oy;
                    if (px>=0 && px<img.width && py>=0 && py<img.height) mask[py*img.width+px]=0;
                }
        }
    });
    if (cuts.length === 0 && !brushMask) return mask;
    if (cuts.length === 0) return mask;  // skip connected-component when only brush edits
    // Connected components - keep largest
    let labels = new Int32Array(img.width*img.height);
    let labelId = 0, labelSizes = {};
    for (let y=0; y<img.height; y++)
        for (let x=0; x<img.width; x++) {
            let idx = y*img.width+x;
            if (!mask[idx] || labels[idx]) continue;
            labelId++;
            let queue=[idx]; labels[idx]=labelId; let size=0;
            while(queue.length) {
                let ci=queue.pop(); size++;
                let cy2=Math.floor(ci/img.width), cx2=ci%img.width;
                for (let ni of [cy2>0?ci-img.width:-1, cy2<img.height-1?ci+img.width:-1, cx2>0?ci-1:-1, cx2<img.width-1?ci+1:-1])
                    if (ni>=0 && mask[ni] && !labels[ni]) { labels[ni]=labelId; queue.push(ni); }
            }
            labelSizes[labelId]=size;
        }
    let maxLabel=0, maxSize=0;
    for (let [l,s] of Object.entries(labelSizes)) if(s>maxSize){maxSize=s;maxLabel=parseInt(l);}
    for (let i=0;i<mask.length;i++) if(mask[i]&&labels[i]!==maxLabel) mask[i]=0;
    return mask;
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, offsetX, offsetY, img.width*scale, img.height*scale);
    let barrel = computeBarrelMask();
    let overlay = ctx.createImageData(img.width, img.height);
    for (let i=0; i<barrel.length; i++) {
        if (showOverlay) {
            // Verde = barrel (tu recorte con cortes)
            if (barrel[i]) { overlay.data[i*4]=0; overlay.data[i*4+1]=220; overlay.data[i*4+2]=0; overlay.data[i*4+3]=100; }
            // Rojo = silueta descartada por cortes
            if (originalMask[i] && !barrel[i]) { overlay.data[i*4]=255; overlay.data[i*4+1]=0; overlay.data[i*4+2]=0; overlay.data[i*4+3]=100; }
        }
        // Azul = predicción del modelo
        if (showPred && predMask && predMask[i] && !barrel[i]) {
            overlay.data[i*4]=50; overlay.data[i*4+1]=100; overlay.data[i*4+2]=255; overlay.data[i*4+3]=120;
        }
    }
    let tc = document.createElement('canvas'); tc.width=img.width; tc.height=img.height;
    tc.getContext('2d').putImageData(overlay,0,0);
    ctx.drawImage(tc, offsetX, offsetY, img.width*scale, img.height*scale);
    cuts.forEach((cut,idx) => {
        let [cx1,cy1]=toCanvasCoords(cut.x1,cut.y1), [cx2,cy2]=toCanvasCoords(cut.x2,cut.y2);
        ctx.beginPath(); ctx.moveTo(cx1,cy1); ctx.lineTo(cx2,cy2);
        ctx.strokeStyle=cut.color; ctx.lineWidth=3; ctx.stroke();
    });
    if (drawing && editMode === 'cut') {
        let [cx1,cy1]=toCanvasCoords(startX,startY);
        ctx.beginPath(); ctx.moveTo(cx1,cy1); ctx.lineTo(currentMouseX,currentMouseY);
        ctx.strokeStyle=COLORS[cuts.length%COLORS.length]; ctx.lineWidth=2;
        ctx.setLineDash([5,5]); ctx.stroke(); ctx.setLineDash([]);
    }
    // Draw cruz point marker (siempre visible si está marcado)
    if (cruzPoint) {
        let [px,py] = toCanvasCoords(cruzPoint.x, cruzPoint.y);
        ctx.strokeStyle='#ffd700'; ctx.lineWidth=2;
        ctx.beginPath(); ctx.moveTo(px-14,py); ctx.lineTo(px+14,py); ctx.moveTo(px,py-14); ctx.lineTo(px,py+14); ctx.stroke();
        ctx.beginPath(); ctx.arc(px,py,6,0,Math.PI*2);
        ctx.fillStyle='rgba(255,215,0,0.85)'; ctx.fill();
        ctx.strokeStyle='#000'; ctx.lineWidth=1.5; ctx.stroke();
        ctx.fillStyle='#ffd700'; ctx.font='bold 12px monospace'; ctx.fillText('Cruz', px+12, py-10);
    }
    // Draw anca point marker (siempre visible si está marcado)
    if (ancaPoint) {
        let [px,py] = toCanvasCoords(ancaPoint.x, ancaPoint.y);
        ctx.strokeStyle='#ff44ff'; ctx.lineWidth=2;
        ctx.beginPath(); ctx.moveTo(px-14,py); ctx.lineTo(px+14,py); ctx.moveTo(px,py-14); ctx.lineTo(px,py+14); ctx.stroke();
        ctx.beginPath(); ctx.arc(px,py,6,0,Math.PI*2);
        ctx.fillStyle='rgba(255,68,255,0.85)'; ctx.fill();
        ctx.strokeStyle='#000'; ctx.lineWidth=1.5; ctx.stroke();
        ctx.fillStyle='#ff44ff'; ctx.font='bold 12px monospace'; ctx.fillText('Anca', px+12, py-10);
    }
    // Draw brush cursor
    if (editMode === 'brush-add' || editMode === 'brush-erase') {
        ctx.beginPath();
        ctx.arc(currentMouseX, currentMouseY, brushRadius * scale, 0, Math.PI * 2);
        ctx.strokeStyle = editMode === 'brush-add' ? '#00ff88' : '#ff4444';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

canvas.addEventListener('mousedown', e => {
    let r=canvas.getBoundingClientRect();
    let mx = e.clientX-r.left, my = e.clientY-r.top;
    [startX,startY]=toImgCoords(mx, my);
    if (editMode === 'cruz') {
        cruzPoint = {x: Math.round(startX), y: Math.round(startY)};
        drawing = false;
        updateCruzInfo();
        draw();
        return;
    }
    if (editMode === 'anca') {
        ancaPoint = {x: Math.round(startX), y: Math.round(startY)};
        drawing = false;
        updateAncaInfo();
        draw();
        return;
    }
    drawing=true;
    if (editMode === 'brush-add' || editMode === 'brush-erase') {
        initBrushMask();
        currentStroke = [];
        let val = editMode === 'brush-add' ? 1 : -1;
        let pixels = paintBrush(startX, startY, val);
        currentStroke.push(...pixels);
        draw();
    }
});
canvas.addEventListener('mousemove', e => {
    let r=canvas.getBoundingClientRect();
    currentMouseX=e.clientX-r.left; currentMouseY=e.clientY-r.top;
    if (drawing && (editMode === 'brush-add' || editMode === 'brush-erase')) {
        let [ix, iy] = toImgCoords(currentMouseX, currentMouseY);
        let val = editMode === 'brush-add' ? 1 : -1;
        let pixels = paintBrush(ix, iy, val);
        currentStroke.push(...pixels);
    }
    if(drawing || editMode === 'brush-add' || editMode === 'brush-erase') draw();
});
canvas.addEventListener('mouseup', e => {
    if(!drawing) return; drawing=false;
    if (editMode === 'cut') {
        let r=canvas.getBoundingClientRect();
        let [ex,ey]=toImgCoords(e.clientX-r.left, e.clientY-r.top);
        if(Math.sqrt((ex-startX)**2+(ey-startY)**2)<10) return;
        cuts.push({x1:startX,y1:startY,x2:ex,y2:ey,color:COLORS[cuts.length%COLORS.length]});
        updateCutsList();
    } else {
        // Save stroke for undo
        if (currentStroke.length > 0) {
            brushHistory.push(currentStroke);
            currentStroke = [];
        }
    }
    draw();
});
// Scroll to change brush size
canvas.addEventListener('wheel', e => {
    if (editMode === 'cut') return;
    e.preventDefault();
    brushRadius = Math.max(2, Math.min(60, brushRadius + (e.deltaY < 0 ? 2 : -2)));
    document.getElementById('brushSize').value = brushRadius;
    document.getElementById('brushSizeLabel').textContent = brushRadius + 'px';
    draw();
}, {passive: false});

function updateCutsList() {
    document.getElementById('cutCount').textContent = cuts.length;
    let html='';
    cuts.forEach((c,i) => {
        html += `<div class="cut-item"><span style="color:${c.color}">Linea ${i+1}</span><button onclick="removeCut(${i})">X</button></div>`;
    });
    document.getElementById('cutsList').innerHTML = html;
}
function removeCut(i) { cuts.splice(i,1); draw(); updateCutsList(); }
function clearAllCuts() { cuts=[]; if(brushMask) brushMask.fill(0); brushHistory=[]; draw(); updateCutsList(); }
function undoLastCut() {
    if (editMode !== 'cut' && brushHistory.length > 0) {
        // Undo last brush stroke
        let stroke = brushHistory.pop();
        for (let p of stroke) brushMask[p.idx] = p.prev;
        draw();
    } else if (cuts.length) { cuts.pop(); draw(); updateCutsList(); }
}
function toggleOverlay() { showOverlay = !showOverlay; let b=document.getElementById('btnOverlay'); b.textContent='Silueta: '+(showOverlay?'ON':'OFF'); b.style.background=(showOverlay?'#2a7':'#555')+'!important'; draw(); }
function togglePred() { showPred = !showPred; let b=document.getElementById('btnPred'); b.textContent='Pred: '+(showPred?'ON':'OFF'); b.style.background=(showPred?'#35f':'#555')+'!important'; draw(); }

function guardar() {
    let barrel = computeBarrelMask();
    let runs=[], count=0, cur=barrel[0];
    for(let i=0;i<barrel.length;i++) {
        if(barrel[i]===cur) count++;
        else { runs.push([cur,count]); cur=barrel[i]; count=1; }
    }
    runs.push([cur,count]);
    // Encode brush edits as RLE for persistence
    let brushRle = [];
    if (brushMask) {
        let bcount=0, bcur=brushMask[0];
        for(let i=0;i<brushMask.length;i++){
            if(brushMask[i]===bcur) bcount++;
            else { brushRle.push([bcur,bcount]); bcur=brushMask[i]; bcount=1; }
        }
        brushRle.push([bcur,bcount]);
    }
    fetch('/api/save', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({id:frameId, cuts:cuts, mask_rle:runs, brush_rle:brushRle, cruz:cruzPoint, anca:ancaPoint, width:img.width, height:img.height})
    }).then(r=>r.json()).then(d => {
        document.getElementById('statusLabel').textContent = 'edited';
        // Auto-avanzar al siguiente
        {% if next_id %}window.location='/edit/{{ next_id }}';{% else %}alert('Guardado! (ultimo frame)');{% endif %}
    });
}

function descartar() {
    if (!confirm('¿Descartar este frame? Se BORRARÁ del índice y se eliminarán sus archivos del disco. Acción IRREVERSIBLE.')) return;
    if (!confirm('Confirmación final: borrar permanentemente "' + frameId + '"?')) return;
    fetch('/api/discard', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({id:frameId})
    }).then(r=>r.json()).then(d => {
        if (!d.ok) { alert('Error: ' + (d.error || 'desconocido')); return; }
        {% if next_id %}window.location='/edit/{{ next_id }}';{% else %}window.location='/gallery';{% endif %}
    });
}

function validar() {
    // Exigir AMBOS puntos (cruz + anca) antes de validar
    if (!cruzPoint) {
        alert('Falta marcar el Punto de Cruz antes de validar.\\nActivá el modo "Punto Cruz" y hacé click en la cruz del individuo.');
        setMode('cruz');
        return;
    }
    if (!ancaPoint) {
        alert('Falta marcar el Punto de Anca antes de validar.\\nActivá el modo "Punto Anca" y hacé click en el anca del individuo.');
        setMode('anca');
        return;
    }
    // Primero guarda los cortes actuales, luego marca como validada
    let barrel = computeBarrelMask();
    let runs=[], count=0, cur=barrel[0];
    for(let i=0;i<barrel.length;i++) {
        if(barrel[i]===cur) count++;
        else { runs.push([cur,count]); cur=barrel[i]; count=1; }
    }
    runs.push([cur,count]);
    let brushRle2 = [];
    if (brushMask) {
        let bcount=0, bcur=brushMask[0];
        for(let i=0;i<brushMask.length;i++){
            if(brushMask[i]===bcur) bcount++;
            else { brushRle2.push([bcur,bcount]); bcur=brushMask[i]; bcount=1; }
        }
        brushRle2.push([bcur,bcount]);
    }
    fetch('/api/validate', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({id:frameId, cuts:cuts, mask_rle:runs, brush_rle:brushRle2, cruz:cruzPoint, anca:ancaPoint, width:img.width, height:img.height})
    }).then(r=>r.json()).then(d => {
        if (!d.ok) { alert('Error al validar: ' + (d.error || 'desconocido')); return; }
        document.getElementById('statusLabel').textContent = 'validated';
        {% if next_id %}window.location='/edit/{{ next_id }}';{% else %}alert('Validado! (ultimo frame)');{% endif %}
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
    if(e.key==='z' && (e.ctrlKey||e.metaKey)) undoLastCut();
    if(e.key==='s' && (e.ctrlKey||e.metaKey)) { e.preventDefault(); guardar(); }
    if(e.key==='v' && (e.ctrlKey||e.metaKey)) { e.preventDefault(); validar(); }
    if(e.key==='ArrowRight') { {% if next_id %}window.location='/edit/{{ next_id }}';{% endif %} }
    if(e.key==='ArrowLeft') { {% if prev_id %}window.location='/edit/{{ prev_id }}';{% endif %} }
});
</script>
</body>
</html>
"""


@app.route('/')
@app.route('/gallery')
def gallery():
    frames = visible_frames()
    groups_map = {}
    for f in frames:
        ind = f['individuo']
        g = groups_map.get(ind)
        if g is None:
            g = {'individuo': ind, 'total': 0, 'pending': 0, 'edited': 0, 'validated': 0, 'cruz': 0, 'anca': 0}
            groups_map[ind] = g
        g['total'] += 1
        g[f['status']] = g.get(f['status'], 0) + 1
        if f.get('cruz'):
            g['cruz'] += 1
        if f.get('anca'):
            g['anca'] += 1
    groups = [groups_map[k] for k in sorted(groups_map)]
    pending = sum(1 for f in frames if f['status'] == 'pending')
    edited = sum(1 for f in frames if f['status'] == 'edited')
    validated = sum(1 for f in frames if f['status'] == 'validated')
    return render_template_string(GALLERY_HTML,
        groups=groups, total=len(frames),
        pending=pending, edited=edited, validated=validated)


@app.route('/individuo')
def individuo():
    ind = request.args.get('ind', '')
    frames = [f for f in load_index() if f['individuo'] == ind]
    if not frames:
        return "Individuo no encontrado", 404
    frames.sort(key=lambda f: f.get('frame_idx', 0))
    pending = sum(1 for f in frames if f['status'] == 'pending')
    edited = sum(1 for f in frames if f['status'] == 'edited')
    validated = sum(1 for f in frames if f['status'] == 'validated')
    cruz = sum(1 for f in frames if f.get('cruz'))
    anca = sum(1 for f in frames if f.get('anca'))
    return render_template_string(INDIVIDUO_HTML,
        individuo=ind, frames=frames,
        pending=pending, edited=edited, validated=validated, cruz=cruz, anca=anca)


@app.route('/frame_img/<frame_id>')
def frame_img(frame_id):
    # El índice guarda la extensión real (.jpg o .png según el origen del frame)
    for f in load_index():
        if f['id'] == frame_id:
            path = DATA_DIR / f['img']
            if path.exists():
                mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
                return send_file(str(path), mimetype=mime)
            break
    return "not found", 404


@app.route('/edit/<frame_id>')
def edit(frame_id):
    frames = load_index()
    frame = None
    idx = -1
    for i, f in enumerate(frames):
        if f['id'] == frame_id:
            frame = f
            idx = i
            break
    if frame is None:
        return "Frame not found", 404

    # Navegación restringida a frames del mismo individuo, ordenados por frame_idx
    siblings = sorted(
        (f for f in frames if f['individuo'] == frame['individuo']),
        key=lambda f: f.get('frame_idx', 0))
    sib_pos = next(i for i, f in enumerate(siblings) if f['id'] == frame_id)
    prev_id = siblings[sib_pos - 1]['id'] if sib_pos > 0 else None
    next_id = siblings[sib_pos + 1]['id'] if sib_pos + 1 < len(siblings) else None

    img_b64 = img_to_b64(DATA_DIR / frame['img'])
    mask_b64 = img_to_b64(DATA_DIR / frame['mask'])
    cuts_json = json.dumps(frame.get('cuts', []))
    brush_rle_json = json.dumps(frame.get('brush_rle', []))
    cruz_json = json.dumps(frame.get('cruz', None))
    anca_json = json.dumps(frame.get('anca', None))

    # Cargar predicción del modelo si existe
    pred_path = DATA_DIR / f"{frame['id']}_pred.png"
    if pred_path.exists():
        pred_b64 = img_to_b64(pred_path)
    else:
        # Imagen transparente 1x1 como fallback
        pred_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAA0lEQVQI12P4z8BQDwAEgAF/QualzQAAAABJRU5ErkJggg=="

    return render_template_string(EDITOR_HTML,
        frame=frame, prev_id=prev_id, next_id=next_id,
        img_b64=img_b64, mask_b64=mask_b64, pred_b64=pred_b64, cuts_json=cuts_json, brush_rle_json=brush_rle_json, cruz_json=cruz_json, anca_json=anca_json)


@app.route('/api/save', methods=['POST'])
def api_save():
    req = request.json
    frame_id = req['id']
    cuts = req['cuts']
    mask_rle = req['mask_rle']
    brush_rle = req.get('brush_rle', [])
    cruz = req.get('cruz', None)
    anca = req.get('anca', None)
    width = req['width']
    height = req['height']

    with _index_lock:
        frames = load_index()
        for f in frames:
            if f['id'] == frame_id:
                f['status'] = 'edited'
                f['cuts'] = cuts
                f['brush_rle'] = brush_rle
                f['cruz'] = cruz
                f['anca'] = anca
                break
        save_index(frames)

    # Guardar máscara del barril
    barrel = np.zeros(width * height, dtype=np.uint8)
    idx = 0
    for val, count in mask_rle:
        if val:
            barrel[idx:idx+count] = 255
        idx += count
    barrel = barrel.reshape(height, width)
    cv2.imwrite(str(DATA_DIR / f"{frame_id}_barrel.png"), barrel)

    return jsonify({'ok': True})


@app.route('/api/discard', methods=['POST'])
def api_discard():
    """Borra el frame del índice y elimina sus archivos del disco."""
    req = request.json
    frame_id = req['id']
    with _index_lock:
        frames = load_index()
        new_frames = [f for f in frames if f['id'] != frame_id]
        if len(new_frames) == len(frames):
            return jsonify({'ok': False, 'error': 'frame no encontrado'}), 404

        for suffix in ('_img.jpg', '_img.png', '_mask.png', '_pred.png', '_barrel.png'):
            p = DATA_DIR / f"{frame_id}{suffix}"
            if p.exists():
                p.unlink()

        save_index(new_frames)
    return jsonify({'ok': True, 'remaining': len(new_frames)})


@app.route('/api/validate', methods=['POST'])
def api_validate():
    req = request.json
    frame_id = req['id']
    cuts = req['cuts']
    mask_rle = req['mask_rle']
    brush_rle = req.get('brush_rle', [])
    cruz = req.get('cruz', None)
    anca = req.get('anca', None)
    width = req['width']
    height = req['height']

    # Solo se puede validar con AMBOS puntos marcados (cruz + anca).
    if not cruz or not anca:
        falta = []
        if not cruz:
            falta.append('cruz')
        if not anca:
            falta.append('anca')
        return jsonify({'ok': False, 'error': 'falta punto: ' + ', '.join(falta)}), 400

    with _index_lock:
        frames = load_index()
        for f in frames:
            if f['id'] == frame_id:
                f['status'] = 'validated'
                f['cuts'] = cuts
                f['brush_rle'] = brush_rle
                f['cruz'] = cruz
                f['anca'] = anca
                break
        save_index(frames)

    # Guardar máscara del barril
    barrel = np.zeros(width * height, dtype=np.uint8)
    idx = 0
    for val, count in mask_rle:
        if val:
            barrel[idx:idx+count] = 255
        idx += count
    barrel = barrel.reshape(height, width)
    cv2.imwrite(str(DATA_DIR / f"{frame_id}_barrel.png"), barrel)

    return jsonify({'ok': True})


if __name__ == '__main__':
    print(f"\n  Barrel Training Editor")
    print(f"  {len(load_index())} frames disponibles")
    print(f"  Abrir en: http://localhost:5055\n")
    app.run(host='0.0.0.0', port=5055, debug=False, threaded=True)
