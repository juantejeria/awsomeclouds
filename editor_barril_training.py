"""
Editor de Barril - Training Mode.
UI para anotar el barril en 108 frames extraídos de videos.
Funciones: ver galería, editar (recortar con líneas), guardar, descartar.
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

app = Flask(__name__)


@app.after_request
def add_ngrok_header(response):
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response


PROJECT = Path(__file__).parent
DATA_DIR = PROJECT / 'output_modelos3d_grandes' / '_barril_training'
INDEX_FILE = DATA_DIR / 'frames_index.json'


def load_index():
    with open(INDEX_FILE) as f:
        return json.load(f)


def save_index(data):
    with open(INDEX_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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
.stat-discarded { background:#e74c3c; color:#fff; }
.filters { background:#0d1b2a; padding:10px 20px; display:flex; gap:10px; align-items:center; }
.filters select, .filters button { padding:6px 12px; border-radius:4px; border:1px solid #444; background:#16213e; color:#eee; font-family:monospace; cursor:pointer; }
.filters button:hover { background:#0ff; color:#000; }
.gallery { display:grid; grid-template-columns:repeat(auto-fill, minmax(200px, 1fr)); gap:8px; padding:15px; }
.card { background:#16213e; border-radius:8px; overflow:hidden; cursor:pointer; position:relative; transition:transform 0.2s; }
.card:hover { transform:scale(1.03); }
.card img { width:100%; height:150px; object-fit:cover; }
.card .info { padding:6px 8px; font-size:11px; }
.card .status { position:absolute; top:5px; right:5px; padding:2px 6px; border-radius:3px; font-size:10px; font-weight:bold; }
.status-pending { background:#e67e22; color:#000; }
.status-edited { background:#27ae60; color:#fff; }
.status-validated { background:#9b59b6; color:#fff; }
.status-discarded { background:#e74c3c; color:#fff; opacity:0.8; }
.card.discarded { opacity:0.4; }
.card.discarded img { filter:grayscale(100%); }
</style>
</head>
<body>
<div class="header">
    <h1>Barrel Training - {{ total }} frames</h1>
    <div class="stats">
        <span class="stat stat-pending">Pendientes: {{ pending }}</span>
        <span class="stat stat-edited">Editados: {{ edited }}</span>
        <span class="stat stat-validated">Validados: {{ validated }}</span>
        <span class="stat stat-discarded">Descartados: {{ discarded }}</span>
    </div>
</div>
<div class="filters">
    <label>Filtrar:</label>
    <select id="filterIndividuo" onchange="applyFilter()">
        <option value="all">Todos</option>
        {% for ind in individuos %}
        <option value="{{ ind }}">{{ ind }}</option>
        {% endfor %}
    </select>
    <select id="filterStatus" onchange="applyFilter()">
        <option value="all">Todos</option>
        <option value="pending">Pendientes</option>
        <option value="edited">Editados</option>
        <option value="validated">Validados</option>
        <option value="discarded">Descartados</option>
    </select>
    <button onclick="window.location.reload()">Refrescar</button>
</div>
<div class="gallery" id="gallery">
    {% for f in frames %}
    <div class="card {{ 'discarded' if f.status == 'discarded' else '' }}"
         data-individuo="{{ f.individuo }}" data-status="{{ f.status }}"
         onclick="window.location='/edit/{{ f.id }}'">
        <img src="/frame_img/{{ f.id }}" loading="lazy">
        <span class="status status-{{ f.status }}">{{ f.status }}</span>
        <div class="info">{{ f.individuo.replace('vaca_','').replace('_36','') }} #{{ f.frame_idx }}</div>
    </div>
    {% endfor %}
</div>
<script>
function applyFilter() {
    let ind = document.getElementById('filterIndividuo').value;
    let status = document.getElementById('filterStatus').value;
    document.querySelectorAll('.card').forEach(c => {
        let show = true;
        if (ind !== 'all' && c.dataset.individuo !== ind) show = false;
        if (status !== 'all' && c.dataset.status !== status) show = false;
        c.style.display = show ? '' : 'none';
    });
}
</script>
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
        <a href="/gallery">Galeria</a>
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
            Scroll o slider para cambiar tamano del pincel.
        </div>
        <h3>Modo</h3>
        <div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:6px;">
            <button class="mode-btn active" id="modeCut" onclick="setMode('cut')">Corte</button>
            <button class="mode-btn" id="modeBrushAdd" onclick="setMode('brush-add')">Pincel +</button>
            <button class="mode-btn" id="modeBrushErase" onclick="setMode('brush-erase')">Pincel -</button>
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
    };
    predImg.onerror = function() {
        predMask = null;
        draw();
        updateCutsList();
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
    document.getElementById(mode==='cut'?'modeCut':mode==='brush-add'?'modeBrushAdd':'modeBrushErase').classList.add('active');
    document.getElementById('brushControls').style.display = mode==='cut'?'none':'flex';
    canvas.style.cursor = mode==='cut'?'crosshair':'none';
    draw();
}

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
    // Draw brush cursor
    if (editMode !== 'cut') {
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
    drawing=true;
    if (editMode !== 'cut') {
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
    if (drawing && editMode !== 'cut') {
        let [ix, iy] = toImgCoords(currentMouseX, currentMouseY);
        let val = editMode === 'brush-add' ? 1 : -1;
        let pixels = paintBrush(ix, iy, val);
        currentStroke.push(...pixels);
    }
    if(drawing || editMode !== 'cut') draw();
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
        body:JSON.stringify({id:frameId, cuts:cuts, mask_rle:runs, brush_rle:brushRle, width:img.width, height:img.height})
    }).then(r=>r.json()).then(d => {
        document.getElementById('statusLabel').textContent = 'edited';
        // Auto-avanzar al siguiente
        {% if next_id %}window.location='/edit/{{ next_id }}';{% else %}alert('Guardado! (ultimo frame)');{% endif %}
    });
}

function descartar() {
    fetch('/api/discard', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({id:frameId})
    }).then(r=>r.json()).then(d => {
        {% if next_id %}window.location='/edit/{{ next_id }}';{% else %}alert('Descartado! (ultimo frame)');{% endif %}
    });
}

function validar() {
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
        body:JSON.stringify({id:frameId, cuts:cuts, mask_rle:runs, brush_rle:brushRle2, width:img.width, height:img.height})
    }).then(r=>r.json()).then(d => {
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
    frames = load_index()
    individuos = sorted(set(f['individuo'] for f in frames))
    pending = sum(1 for f in frames if f['status'] == 'pending')
    edited = sum(1 for f in frames if f['status'] == 'edited')
    validated = sum(1 for f in frames if f['status'] == 'validated')
    discarded = sum(1 for f in frames if f['status'] == 'discarded')
    return render_template_string(GALLERY_HTML,
        frames=frames, individuos=individuos, total=len(frames),
        pending=pending, edited=edited, validated=validated, discarded=discarded)


@app.route('/frame_img/<frame_id>')
def frame_img(frame_id):
    path = DATA_DIR / f"{frame_id}_img.jpg"
    if path.exists():
        return send_file(str(path), mimetype='image/jpeg')
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

    # Navigation: find prev/next non-discarded
    prev_id = None
    next_id = None
    for i in range(idx - 1, -1, -1):
        if frames[i]['status'] != 'discarded':
            prev_id = frames[i]['id']
            break
    for i in range(idx + 1, len(frames)):
        if frames[i]['status'] != 'discarded':
            next_id = frames[i]['id']
            break

    img_b64 = img_to_b64(DATA_DIR / frame['img'])
    mask_b64 = img_to_b64(DATA_DIR / frame['mask'])
    cuts_json = json.dumps(frame.get('cuts', []))
    brush_rle_json = json.dumps(frame.get('brush_rle', []))

    # Cargar predicción del modelo si existe
    pred_path = DATA_DIR / f"{frame['id']}_pred.png"
    if pred_path.exists():
        pred_b64 = img_to_b64(pred_path)
    else:
        # Imagen transparente 1x1 como fallback
        pred_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQABNjN9GQAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAA0lEQVQI12P4z8BQDwAEgAF/QualzQAAAABJRU5ErkJggg=="

    return render_template_string(EDITOR_HTML,
        frame=frame, prev_id=prev_id, next_id=next_id,
        img_b64=img_b64, mask_b64=mask_b64, pred_b64=pred_b64, cuts_json=cuts_json, brush_rle_json=brush_rle_json)


@app.route('/api/save', methods=['POST'])
def api_save():
    req = request.json
    frame_id = req['id']
    cuts = req['cuts']
    mask_rle = req['mask_rle']
    brush_rle = req.get('brush_rle', [])
    width = req['width']
    height = req['height']

    frames = load_index()
    for f in frames:
        if f['id'] == frame_id:
            f['status'] = 'edited'
            f['cuts'] = cuts
            f['brush_rle'] = brush_rle
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
    req = request.json
    frame_id = req['id']
    frames = load_index()
    for f in frames:
        if f['id'] == frame_id:
            f['status'] = 'discarded'
            break
    save_index(frames)
    return jsonify({'ok': True})


@app.route('/api/validate', methods=['POST'])
def api_validate():
    req = request.json
    frame_id = req['id']
    cuts = req['cuts']
    mask_rle = req['mask_rle']
    brush_rle = req.get('brush_rle', [])
    width = req['width']
    height = req['height']

    frames = load_index()
    for f in frames:
        if f['id'] == frame_id:
            f['status'] = 'validated'
            f['cuts'] = cuts
            f['brush_rle'] = brush_rle
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
    app.run(host='0.0.0.0', port=5055, debug=False)
