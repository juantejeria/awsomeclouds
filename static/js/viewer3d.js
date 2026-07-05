// ====================================================================
// viewer3d.js - Interactive 3D PLY Model Viewer (ES Module)
// ====================================================================

import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── State ──

let scene, camera, renderer, controls;
// Shell / contorno (superficie exterior)
let currentMesh = null;
let currentPoints = null;
let currentWireframe = null;
let modelosData = [];
let currentModelId = null; // ID of currently loaded model (e.g. 'vaca1')
let renderMode = 'solid'; // solid | wireframe | points
let animFrameId = null;
let _pendingLoads = 0;  // contador de cargas en vuelo → pausa render durante carga

// ── Medición del diámetro torácico ("detrás de las manos") ──
let diameterGroup = null;        // THREE.Group con la línea + anillo de la sección
let currentGeometry = null;      // geometría centrada del modelo cargado (cm)
let currentModelMeta = null;     // metadatos del modelo (incluye barril_dir)
let girthFrac = 0.20;            // fracción del largo desde el frente (manos) al plano
let lastGirthSection = null;     // última sección medida (para guardar la etiqueta)
let verijaFrac = 0.25;           // fracción del largo desde el FONDO (atrás) al plano verija
let verijaRaiseFrac = 0.5;       // piso de la verija (0–1): 0=abajo afuera del barril,
                                 // 0.5=fondo de la panza, 1=tope del barril
let lastVerijaSection = null;    // última sección verija medida
let cruzFrac = 0.20;             // fracción del largo desde el FRENTE al plano de la cruz
let lastCruzSection = null;      // última sección cruz medida (cruz_pose.pt)
let ancaFrac = 0.25;             // fracción del largo desde el FONDO al plano de la anca
let lastAncaSection = null;      // última sección anca medida (anotación manual)

// ── Recorte cruz↔verija (caja con piso en el mínimo de la verija) ──
let cutEnabled = false;          // mostrar el modelo recortado a la caja
const clipPlanes = [new THREE.Plane(), new THREE.Plane(), new THREE.Plane()];
let lastCutBox = null;           // {xLo, xHi, yFloor} de la última caja calculada
let lastCutVolumeLiters = null;  // volumen de la región recortada (L)
let lastSecVolumes = null;       // [v1,v2,v3,v4] de las 4 sub-secciones (cruz→verija)
let lastVolVerija = null;        // vol cruz↔anca con piso en mínimo verija (L)
let lastVolMid = null;           // vol cruz↔anca con piso a 0.5*dist desde topline (L)
let lastVolCentral = null;       // vol central sin piso (punto medio ±CENTRAL_HALF_FRAC*dist) (L)
const CENTRAL_HALF_FRAC = 1 / 8;  // franja central: 1/8 de la distancia cruz↔anca a cada lado
let midFloorFrac = 0.5;          // fracción de la distancia cruz↔anca para el piso de corte (UI editable)
// Recorte de "cresta" (no destructivo, localizado): ignora lo que esté por encima
// del techo (ceil) SOLO dentro del rango X [xLo,xHi] (fracciones desde el FRENTE).
// No toca el PLY; solo limpia yMax/diámetros/min-max de las secciones en esa zona.
let crestTrim = { on: false, xLo: 0.0, xHi: 0.5, ceil: 0.10 };
let crestTrimGroup = null;       // caja visual de la zona recortada
let lastDist = null;             // distancia cruz↔anca (cm)
let floorGroup = null;           // grupo del piso (mín verija) — su check decide con/sin piso del volumen
let floorMidGroup = null;        // grupo del piso a 0.5*dist (vol_piso_meddist) — toggle propio
let sectionsGroup = null;        // grupo del tramo central (¼ a cada lado del punto medio) — toggle propio
let lastFloorOn = true;          // si el volumen usa el piso (mín verija) o es sección completa

// ── Init ──

function initViewer3D() {
    const container = document.getElementById('viewer3dContainer');
    const canvas = document.getElementById('viewer3dCanvas');
    if (!container || !canvas) return;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f5f5);

    // Camera
    const aspect = container.clientWidth / container.clientHeight;
    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 10000);
    camera.position.set(0, 100, 300);

    // Renderer — alpha:false (opaco), antialias:false y pixelRatio=1 para rendimiento
    renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: false,       // antes true, baja bastante el costo
        alpha: false,
        powerPreference: 'high-performance',
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(1);  // antes window.devicePixelRatio → pesado en Retina
    renderer.setClearColor(0xf5f5f5, 1);
    renderer.localClippingEnabled = true;  // recorte cruz↔verija (clipping planes por material)

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(200, 300, 200);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    dirLight2.position.set(-200, 100, -200);
    scene.add(dirLight2);

    // Grid helper
    const grid = new THREE.GridHelper(500, 20, 0xcccccc, 0xe0e0e0);
    scene.add(grid);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 10;
    controls.maxDistance = 2000;

    // Resize debounced con check de dimensiones (evita setSize excesivo)
    let _lastW = 0, _lastH = 0;
    let _resizeTimeout = null;
    function _doResize() {
        const w = container.clientWidth;
        const h = container.clientHeight;
        if (w === _lastW && h === _lastH) return;
        if (w <= 0 || h <= 0) return;
        _lastW = w;
        _lastH = h;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
    }
    const resizeObserver = new ResizeObserver(() => {
        if (_resizeTimeout) clearTimeout(_resizeTimeout);
        _resizeTimeout = setTimeout(_doResize, 120);
    });
    resizeObserver.observe(container);

    function animate() {
        animFrameId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // Load available models
    loadModelosDisponibles();
}

// ── Load available models list ──

async function loadModelosDisponibles(selectModelId) {
    try {
        const resp = await fetch('/api/modelos_disponibles');
        modelosData = await resp.json();
        const sel = document.getElementById('modelSelector');
        if (!sel) return;
        sel.innerHTML = '';

        // Opción placeholder default (no muestra ningún modelo)
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = modelosData.length === 0
            ? 'No hay modelos disponibles'
            : '-- Seleccioná un modelo --';
        placeholder.disabled = modelosData.length === 0;
        sel.appendChild(placeholder);

        if (modelosData.length === 0) return;

        let targetIdx = -1;
        // Agrupar por dataset (optgroup) para navegar varios datasets a la vez.
        const groups = {};
        modelosData.forEach((m, i) => {
            const g = m.dataset_label || 'otros';
            (groups[g] = groups[g] || []).push(i);
            if (selectModelId && (m.id === selectModelId || m.individuo === selectModelId)) {
                targetIdx = i;
            }
        });
        Object.keys(groups).forEach(g => {
            const og = document.createElement('optgroup');
            og.label = g;
            groups[g].forEach(i => {
                const m = modelosData[i];
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = (m.nombre || m.id) + (m.peso_kg ? ` (${m.peso_kg} kg)` : '');
                og.appendChild(opt);
            });
            sel.appendChild(og);
        });

        // Solo auto-cargar si VINO un modelo específico por parámetro
        // (ej. recién generado desde "Generar modelo 3D"). Sino, queda en placeholder.
        if (selectModelId && targetIdx >= 0) {
            sel.value = targetIdx;
            loadModelByIndex(targetIdx);
        } else {
            sel.value = '';
            clearModel();  // limpia cualquier modelo previo
        }
    } catch (e) {
        console.error('Error loading modelos disponibles:', e);
    }
}

// ── Load a 3D model by index ──

function loadModelByIndex(idx) {
    const m = modelosData[idx];
    if (!m) return;

    const shellFile = m.ply_3d || m.ply_lateral;
    if (!shellFile) return;

    currentModelId = m.id;
    currentModelMeta = m;
    // Posición del girth: usar el punto guardado del modelo si existe
    // (manual > auto > 20% por defecto). El slider se sincroniza.
    const _savedFrac = (m.girth_frac_manual != null) ? m.girth_frac_manual
                      : (m.girth_frac != null ? m.girth_frac : 0.20);
    girthFrac = Math.max(0, Math.min(0.5, parseFloat(_savedFrac) || 0.20));
    const _gs = document.getElementById('girthPosSlider');
    if (_gs) _gs.value = Math.round(girthFrac * 100);
    const _gl = document.getElementById('girthPosLabel');
    if (_gl) _gl.textContent = Math.round(girthFrac * 100) + '%';
    // Verija: posición guardada desde el fondo (manual > 25% por defecto).
    const _savedVer = (m.verija_frac_manual != null) ? m.verija_frac_manual : 0.25;
    verijaFrac = Math.max(0, Math.min(0.5, parseFloat(_savedVer) || 0.25));
    const _vs = document.getElementById('verijaPosSlider');
    if (_vs) _vs.value = Math.round(verijaFrac * 100);
    const _vl = document.getElementById('verijaPosLabel');
    if (_vl) _vl.textContent = Math.round(verijaFrac * 100) + '%';
    // Verija — subida del piso hacia adentro del barril (manual > 0% por defecto).
    const _vr = parseFloat(m.verija_raise_manual);
    verijaRaiseFrac = Number.isFinite(_vr) ? Math.max(0, Math.min(1, _vr)) : 0.5;
    const _vrs = document.getElementById('verijaRaiseSlider');
    if (_vrs) _vrs.value = Math.round(verijaRaiseFrac * 100);
    const _vrl = document.getElementById('verijaRaiseLabel');
    if (_vrl) _vrl.textContent = Math.round(verijaRaiseFrac * 100) + '%';
    // Cruz: posición detectada por cruz_pose.pt desde el frente
    // (corrección manual > auto cruz_frac > 20% por defecto).
    const _savedCruz = (m.cruz_frac_manual != null) ? m.cruz_frac_manual
                      : (m.cruz_frac != null ? m.cruz_frac : 0.20);
    cruzFrac = Math.max(0, Math.min(0.5, parseFloat(_savedCruz) || 0.20));
    const _cs = document.getElementById('cruzPosSlider');
    if (_cs) _cs.value = Math.round(cruzFrac * 100);
    const _cl = document.getElementById('cruzPosLabel');
    if (_cl) _cl.textContent = Math.round(cruzFrac * 100) + '%';
    // Anca: posición anotada a mano desde el FONDO (corrección manual > auto anca_frac > 25%).
    const _savedAnca = (m.anca_frac_manual != null) ? m.anca_frac_manual
                      : (m.anca_frac != null ? m.anca_frac : 0.25);
    ancaFrac = Math.max(0, Math.min(0.5, parseFloat(_savedAnca) || 0.25));
    const _as = document.getElementById('ancaPosSlider');
    if (_as) _as.value = Math.round(ancaFrac * 100);
    const _al = document.getElementById('ancaPosLabel');
    if (_al) _al.textContent = Math.round(ancaFrac * 100) + '%';
    // Piso de corte: % guardado de la distancia cruz↔anca (manual > 50% por defecto).
    const _savedCorte = (m.corte_frac_manual != null) ? m.corte_frac_manual : 0.5;
    midFloorFrac = Math.max(0, Math.min(1, parseFloat(_savedCorte) || 0.5));
    const _ci = document.getElementById('midFloorPctInput');
    if (_ci) _ci.value = Math.round(midFloorFrac * 100);
    // Recorte de cresta guardado por individuo (no destructivo, localizado).
    const _ct = m.crest_trim || {};
    crestTrim = {
        on: !!_ct.on,
        xLo: Math.max(0, Math.min(1, parseFloat(_ct.x_lo != null ? _ct.x_lo : 0.0) || 0.0)),
        xHi: Math.max(0, Math.min(1, parseFloat(_ct.x_hi != null ? _ct.x_hi : 0.5) || 0.5)),
        ceil: Math.max(0, Math.min(0.5, parseFloat(_ct.ceil != null ? _ct.ceil : 0.10) || 0.10)),
    };
    _syncCrestTrimUI();
    _pendingLoads++;
    clearModel();
    // Cache-bust: asegura recarga del PLY cuando se regenera (el PLYLoader
    // cachea por URL — sin esto el viewer mostraría el modelo viejo aunque
    // el archivo en disco sea nuevo).
    const _t = Date.now();
    loadShellPLY(`/api/modelo3d/${m.id}/${shellFile}?t=${_t}`, function() {
        _pendingLoads--;
    });
    updateInfoPanel(m);

    const heightInput = document.getElementById('inputAlturaCm');
    if (heightInput) heightInput.value = '';
}

// ── Load PLY model ──

function loadModelo(url) {
    // Back-compat: single-arg loader used by external code (loads as shell)
    clearModel();
    loadShellPLY(url);
}

function loadShellPLY(url, onDone) {
    const loader3d = document.getElementById('viewer3dLoader');
    if (loader3d) loader3d.style.display = 'flex';

    const loader = new PLYLoader();
    loader.load(url, function(geometry) {
        try {
        geometry.computeVertexNormals();

        // Center geometry
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);

        // Determine if geometry has vertex colors
        const hasColors = geometry.hasAttribute('color');

        // Solid mesh material
        let material;
        if (hasColors) {
            material = new THREE.MeshPhongMaterial({
                vertexColors: true,
                side: THREE.DoubleSide,
                shininess: 30,
                flatShading: false,
            });
        } else {
            material = new THREE.MeshPhongMaterial({
                color: 0x8B6914,
                side: THREE.DoubleSide,
                shininess: 30,
                flatShading: false,
            });
        }

        // Create mesh
        currentMesh = new THREE.Mesh(geometry, material);
        scene.add(currentMesh);

        // Create wireframe (hidden initially)
        const wireMat = new THREE.MeshBasicMaterial({
            color: hasColors ? undefined : 0x44aa44,
            vertexColors: hasColors,
            wireframe: true,
        });
        currentWireframe = new THREE.Mesh(geometry, wireMat);
        currentWireframe.visible = false;
        scene.add(currentWireframe);

        // Create points (hidden initially)
        const pointsMat = new THREE.PointsMaterial({
            size: 1.5,
            vertexColors: hasColors,
            color: hasColors ? undefined : 0x66cc66,
        });
        currentPoints = new THREE.Points(geometry, pointsMat);
        currentPoints.visible = false;
        scene.add(currentPoints);

        // Apply current render mode
        applyRenderMode();

        // Guardar geometría centrada para medir la sección del diámetro
        currentGeometry = geometry;
        updateGirthMeasurement();

        // Fit camera to model
        fitCameraToModel(geometry);

        // Hide loader
        if (loader3d) loader3d.style.display = 'none';
        } finally {
            if (typeof onDone === 'function') onDone();
        }
    },
    undefined,
    function(error) {
        console.error('Error loading PLY:', error);
        if (loader3d) loader3d.style.display = 'none';
        if (typeof onDone === 'function') onDone();
    });
}

// ── Clear current model ──

function clearModel() {
    if (currentMesh) { scene.remove(currentMesh); currentMesh = null; }
    if (currentWireframe) { scene.remove(currentWireframe); currentWireframe = null; }
    if (currentPoints) { scene.remove(currentPoints); currentPoints = null; }
    _clearDiameterGroup();
    _clearCrestTrimGroup();
    currentGeometry = null;
}

// ── Medición del diámetro torácico "detrás de las manos" ──

function _clearDiameterGroup() {
    if (!diameterGroup) return;
    scene.remove(diameterGroup);
    diameterGroup.traverse(function(o) {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
    });
    diameterGroup = null;
}

function _clearFloorGroup() {
    if (!floorGroup) return;
    scene.remove(floorGroup);
    floorGroup.traverse(function(o) {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
    });
    floorGroup = null;
}

function _clearSectionsGroup() {
    if (!sectionsGroup) return;
    scene.remove(sectionsGroup);
    sectionsGroup.traverse(function(o) {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
    });
    sectionsGroup = null;
}

function _clearFloorMidGroup() {
    if (!floorMidGroup) return;
    scene.remove(floorMidGroup);
    floorMidGroup.traverse(function(o) {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
    });
    floorMidGroup = null;
}

function _clearCrestTrimGroup() {
    if (!crestTrimGroup) return;
    scene.remove(crestTrimGroup);
    crestTrimGroup.traverse(function(o) {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
    });
    crestTrimGroup = null;
}

// Corta la malla CERRADA con el plano X = xPlane (intersección plano–triángulos).
// Cada triángulo que cruza el plano aporta un segmento; el conjunto de segmentos
// es el CONTORNO REAL de la sección transversal del modelo en ese X. Devuelve los
// segmentos (pares de puntos), los extremos Y/Z y el perímetro real (suma de
// longitudes de los segmentos). NO asume elipse — sigue la superficie real.
function _sliceMeshAtX(geometry, xPlane) {
    const pos = geometry.getAttribute('position');
    const index = geometry.getIndex();
    const triCount = index ? (index.count / 3) : (pos.count / 3);
    const gi = (t, k) => index ? index.getX(t * 3 + k) : (t * 3 + k);

    const segs = [];  // [p0,q0, p1,q1, ...] para THREE.LineSegments
    let yMin = Infinity, yMax = -Infinity, zMin = Infinity, zMax = -Infinity;
    const V = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];

    for (let t = 0; t < triCount; t++) {
        V[0].fromBufferAttribute(pos, gi(t, 0));
        V[1].fromBufferAttribute(pos, gi(t, 1));
        V[2].fromBufferAttribute(pos, gi(t, 2));
        const hits = [];
        for (let e = 0; e < 3; e++) {
            const p = V[e], q = V[(e + 1) % 3];
            const dp = p.x - xPlane, dq = q.x - xPlane;
            // El borde cruza el plano si un extremo queda a cada lado.
            if ((dp < 0 && dq >= 0) || (dp >= 0 && dq < 0)) {
                const f = dp / (dp - dq);
                hits.push(new THREE.Vector3(
                    xPlane, p.y + (q.y - p.y) * f, p.z + (q.z - p.z) * f));
            }
        }
        if (hits.length === 2) {
            segs.push(hits[0], hits[1]);
            for (const h of hits) {
                if (h.y < yMin) yMin = h.y;
                if (h.y > yMax) yMax = h.y;
                if (h.z < zMin) zMin = h.z;
                if (h.z > zMax) zMax = h.z;
            }
        }
    }
    if (!segs.length) return null;
    let perim = 0;
    for (let i = 0; i < segs.length; i += 2) perim += segs[i].distanceTo(segs[i + 1]);
    return { segs, yMin, yMax, zMin, zMax,
             yc: (yMin + yMax) / 2, zc: (zMin + zMax) / 2,
             vert: yMax - yMin, depth: zMax - zMin, perim };
}

// ── Volumen de la región recortada cruz↔verija con piso en y=yFloor ──
// Polígono (z,y) de una sección: puntos únicos del contorno ordenados angularmente
// alrededor del centroide (las secciones de barril son estrelladas → orden válido).
function _polyFromSection(sec) {
    const pts = [];
    const seen = new Set();
    for (const p of sec.segs) {
        const key = Math.round(p.z * 50) + ',' + Math.round(p.y * 50);
        if (seen.has(key)) continue;
        seen.add(key);
        pts.push({ z: p.z, y: p.y });
    }
    if (pts.length < 3) return null;
    pts.sort((a, b) => Math.atan2(a.y - sec.yc, a.z - sec.zc) -
                       Math.atan2(b.y - sec.yc, b.z - sec.zc));
    return pts;
}

// Sutherland-Hodgman: recorta el polígono al semiplano y >= yFloor.
function _clipPolyAboveFloor(poly, yFloor) {
    const out = [];
    const n = poly.length;
    for (let i = 0; i < n; i++) {
        const cur = poly[i], prev = poly[(i + n - 1) % n];
        const curIn = cur.y >= yFloor, prevIn = prev.y >= yFloor;
        if (curIn) {
            if (!prevIn) {
                const t = (yFloor - prev.y) / (cur.y - prev.y);
                out.push({ z: prev.z + (cur.z - prev.z) * t, y: yFloor });
            }
            out.push(cur);
        } else if (prevIn) {
            const t = (yFloor - prev.y) / (cur.y - prev.y);
            out.push({ z: prev.z + (cur.z - prev.z) * t, y: yFloor });
        }
    }
    return out;
}

function _polyArea(poly) {
    let a = 0;
    for (let i = 0; i < poly.length; i++) {
        const p = poly[i], q = poly[(i + 1) % poly.length];
        a += p.z * q.y - q.z * p.y;
    }
    return Math.abs(a) / 2;
}

// Integra el área (recortada al piso) sobre X en [xLo,xHi] → litros (geom en cm).
function _clippedVolumeLiters(geometry, xLo, xHi, yFloor, nSteps) {
    if (xHi - xLo <= 0) return 0;
    nSteps = nSteps || 36;
    const dx = (xHi - xLo) / nSteps;
    let vol = 0;  // cm³
    for (let i = 0; i < nSteps; i++) {
        const sec = _sliceMeshAtX(geometry, xLo + (i + 0.5) * dx);
        if (!sec) continue;
        const poly = _polyFromSection(sec);
        if (!poly) continue;
        const clipped = _clipPolyAboveFloor(poly, yFloor);
        if (clipped.length >= 3) vol += _polyArea(clipped) * dx;
    }
    return vol / 1000.0;
}

// Sutherland-Hodgman: recorta el polígono al semiplano y <= yCeil (techo).
function _clipPolyBelowCeil(poly, yCeil) {
    const out = [];
    const n = poly.length;
    for (let i = 0; i < n; i++) {
        const cur = poly[i], prev = poly[(i + n - 1) % n];
        const curIn = cur.y <= yCeil, prevIn = prev.y <= yCeil;
        if (curIn) {
            if (!prevIn) {
                const t = (yCeil - prev.y) / (cur.y - prev.y);
                out.push({ z: prev.z + (cur.z - prev.z) * t, y: yCeil });
            }
            out.push(cur);
        } else if (prevIn) {
            const t = (yCeil - prev.y) / (cur.y - prev.y);
            out.push({ z: prev.z + (cur.z - prev.z) * t, y: yCeil });
        }
    }
    return out;
}

// Recorte de cresta: si el plano X cae dentro de la zona [xLoW,xHiW] y la sección
// asoma por encima del techo yCeil, devuelve una sección NUEVA con el contorno
// recortado plano en yCeil (limpia yMax/vert/perim). Fuera de la zona, sin cambios.
function _applyCrestTrim(sec, xPlane, xLoW, xHiW, yCeil) {
    if (!sec || xLoW == null) return sec;
    if (xPlane < xLoW || xPlane > xHiW) return sec;
    if (sec.yMax <= yCeil) return sec;  // nada por encima del techo
    const poly = _polyFromSection(sec);
    if (!poly) return sec;
    const clipped = _clipPolyBelowCeil(poly, yCeil);
    if (clipped.length < 3) return sec;
    let yMin = Infinity, yMax = -Infinity, zMin = Infinity, zMax = -Infinity;
    const segs = [];
    const n = clipped.length;
    for (let i = 0; i < n; i++) {
        const p = clipped[i], q = clipped[(i + 1) % n];
        segs.push(new THREE.Vector3(xPlane, p.y, p.z), new THREE.Vector3(xPlane, q.y, q.z));
        if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y;
        if (p.z < zMin) zMin = p.z; if (p.z > zMax) zMax = p.z;
    }
    let perim = 0;
    for (let i = 0; i < segs.length; i += 2) perim += segs[i].distanceTo(segs[i + 1]);
    return { segs, yMin, yMax, zMin, zMax,
             yc: (yMin + yMax) / 2, zc: (zMin + zMax) / 2,
             vert: yMax - yMin, depth: zMax - zMin, perim };
}

// Aplica/quita los 3 clipping planes (x>=xLo, x<=xHi, y>=yFloor) a las mallas.
function updateCutClip() {
    if (cutEnabled && lastCutBox) {
        clipPlanes[0].set(new THREE.Vector3(1, 0, 0), -lastCutBox.xLo);
        clipPlanes[1].set(new THREE.Vector3(-1, 0, 0), lastCutBox.xHi);
        clipPlanes[2].set(new THREE.Vector3(0, 1, 0), -lastCutBox.yFloor);
    }
    const planes = (cutEnabled && lastCutBox) ? clipPlanes : null;
    for (const mesh of [currentMesh, currentWireframe, currentPoints]) {
        if (!mesh || !mesh.material) continue;
        mesh.material.clippingPlanes = planes;
        mesh.material.clipIntersection = false;  // intersección de los semiespacios = caja
        mesh.material.needsUpdate = true;
    }
}

function _updateCutReadout() {
    const set = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
    const suf = lastFloorOn ? ' (c/piso)' : ' (s/piso)';
    set('info3dCutVol', (lastCutVolumeLiters != null) ? lastCutVolumeLiters.toFixed(1) + ' L' + suf : '-');
    const v = lastSecVolumes;
    for (let i = 1; i <= 4; i++) {
        set('info3dSec' + i, (v && v[i - 1] != null) ? v[i - 1].toFixed(1) + ' L' : '-');
    }
}

// Construye/actualiza el contorno real de la sección + las líneas de diámetro.
function updateGirthMeasurement() {
    _clearDiameterGroup();
    _clearFloorGroup();
    _clearFloorMidGroup();
    _clearSectionsGroup();
    _clearCrestTrimGroup();
    if (!currentGeometry) return;

    currentGeometry.computeBoundingBox();
    const bb = currentGeometry.boundingBox;
    const xMin = bb.min.x, xMax = bb.max.x;
    const L = xMax - xMin;
    if (L <= 0) return;

    // Dirección de marcha = hacia dónde mira el barril (manos al frente).
    // Si mira a la derecha, el frente está en xMax; si a la izquierda, en xMin.
    // Sin dato guardado → asumimos 'right' (se puede ajustar con el slider).
    const dir = (currentModelMeta && currentModelMeta.barril_dir) || 'unknown';
    const facingRight = (dir !== 'left');  // 'right' o 'unknown' → frente en xMax
    const xFront = facingRight ? xMax : xMin;
    const _dl = document.getElementById('girthDirLabel');
    if (_dl) _dl.textContent = (dir === 'left' ? '◄ izq' : (dir === 'right' ? 'der ►' : '? (asumido der)'));
    let xPlane = facingRight ? (xFront - girthFrac * L) : (xFront + girthFrac * L);

    // Recorte de cresta (no destructivo, localizado). Zona X [xTrimLoW,xTrimHiW]
    // (fracciones desde el FRENTE) y techo yCeilW (fracción desde lo más alto).
    let xTrimLoW = null, xTrimHiW = null, yCeilW = null;
    if (crestTrim.on) {
        const xa = facingRight ? (xFront - crestTrim.xLo * L) : (xFront + crestTrim.xLo * L);
        const xb = facingRight ? (xFront - crestTrim.xHi * L) : (xFront + crestTrim.xHi * L);
        xTrimLoW = Math.min(xa, xb);
        xTrimHiW = Math.max(xa, xb);
        yCeilW = bb.max.y - crestTrim.ceil * (bb.max.y - bb.min.y);
    }

    // Corte real de la malla en el plano. Si cae justo en el borde (sección
    // degenerada/vacía) lo empujamos un poco hacia el cuerpo y reintentamos.
    let sec = _sliceMeshAtX(currentGeometry, xPlane);
    for (let k = 0; k < 4 && !sec; k++) {
        xPlane += (facingRight ? -1 : 1) * Math.max(0.5, L * 0.01);
        sec = _sliceMeshAtX(currentGeometry, xPlane);
    }
    if (!sec) return;

    const group = new THREE.Group();

    // Torácico: se calcula (compat. con el botón guardar) pero NO se dibuja:
    // los diámetros visibles son CRUZ y ANCA.
    lastGirthSection = sec;

    // ── VERIJA: plano desde el FONDO. Se calcula SOLO para el piso del volumen
    // (yMin de la verija); no se dibuja su anillo.
    const xRear = facingRight ? xMin : xMax;
    let xVer = facingRight ? (xRear + verijaFrac * L) : (xRear - verijaFrac * L);
    let secV = _sliceMeshAtX(currentGeometry, xVer);
    for (let k = 0; k < 4 && !secV; k++) {
        xVer += (facingRight ? 1 : -1) * Math.max(0.5, L * 0.01);
        secV = _sliceMeshAtX(currentGeometry, xVer);
    }
    secV = _applyCrestTrim(secV, xVer, xTrimLoW, xTrimHiW, yCeilW);
    lastVerijaSection = secV || null;

    // ── Diámetro ANCA: plano desde el FONDO (cadera), por anotación manual.
    // Colores propios (anillo magenta, vertical magenta claro).
    let xAnca = facingRight ? (xRear + ancaFrac * L) : (xRear - ancaFrac * L);
    let secAn = _sliceMeshAtX(currentGeometry, xAnca);
    for (let k = 0; k < 4 && !secAn; k++) {
        xAnca += (facingRight ? 1 : -1) * Math.max(0.5, L * 0.01);
        secAn = _sliceMeshAtX(currentGeometry, xAnca);
    }
    secAn = _applyCrestTrim(secAn, xAnca, xTrimLoW, xTrimHiW, yCeilW);
    if (secAn) {
        group.add(new THREE.LineSegments(
            new THREE.BufferGeometry().setFromPoints(secAn.segs),
            new THREE.LineBasicMaterial({ color: 0xff44ff })));         // anillo anca (magenta)
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(xAnca, secAn.yMin, secAn.zc),
            new THREE.Vector3(xAnca, secAn.yMax, secAn.zc),
        ]), new THREE.LineBasicMaterial({ color: 0xff88ff })));         // vertical anca
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(xAnca, secAn.yc, secAn.zMin),
            new THREE.Vector3(xAnca, secAn.yc, secAn.zMax),
        ]), new THREE.LineBasicMaterial({ color: 0xffbbff })));         // profundidad anca
        lastAncaSection = secAn;
    } else {
        lastAncaSection = null;
    }

    // ── Diámetro CRUZ: plano desde el FRENTE, por anotación manual.
    // Colores propios (anillo cian, vertical naranja).
    let xCruz = facingRight ? (xFront - cruzFrac * L) : (xFront + cruzFrac * L);
    let secC = _sliceMeshAtX(currentGeometry, xCruz);
    for (let k = 0; k < 4 && !secC; k++) {
        xCruz += (facingRight ? -1 : 1) * Math.max(0.5, L * 0.01);
        secC = _sliceMeshAtX(currentGeometry, xCruz);
    }
    secC = _applyCrestTrim(secC, xCruz, xTrimLoW, xTrimHiW, yCeilW);
    if (secC) {
        group.add(new THREE.LineSegments(
            new THREE.BufferGeometry().setFromPoints(secC.segs),
            new THREE.LineBasicMaterial({ color: 0x00e5ff })));         // anillo cruz (cian)
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(xCruz, secC.yMin, secC.zc),
            new THREE.Vector3(xCruz, secC.yMax, secC.zc),
        ]), new THREE.LineBasicMaterial({ color: 0xff9100 })));         // vertical cruz (naranja)
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(xCruz, secC.yc, secC.zMin),
            new THREE.Vector3(xCruz, secC.yc, secC.zMax),
        ]), new THREE.LineBasicMaterial({ color: 0xffd180 })));         // profundidad cruz
        lastCruzSection = secC;
    } else {
        lastCruzSection = null;
    }

    diameterGroup = group;
    const tgl = document.getElementById('girthToggle');
    group.visible = tgl ? tgl.checked : true;
    scene.add(group);

    // Caja visual de la zona recortada (cresta): rojo translúcido sobre el techo.
    if (crestTrim.on && xTrimLoW != null && yCeilW < bb.max.y) {
        const cg = new THREE.Group();
        const cx = (xTrimLoW + xTrimHiW) / 2;
        const cy = (yCeilW + bb.max.y) / 2;
        const cz = (bb.min.z + bb.max.z) / 2;
        const boxGeo = new THREE.BoxGeometry(
            Math.max(0.1, xTrimHiW - xTrimLoW),
            Math.max(0.1, bb.max.y - yCeilW),
            Math.max(0.1, bb.max.z - bb.min.z));
        const boxMesh = new THREE.Mesh(boxGeo, new THREE.MeshBasicMaterial({
            color: 0xff3030, transparent: true, opacity: 0.16, depthWrite: false }));
        boxMesh.position.set(cx, cy, cz);
        cg.add(boxMesh);
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeo),
            new THREE.LineBasicMaterial({ color: 0xff3030 }));
        edges.position.set(cx, cy, cz);
        cg.add(edges);
        crestTrimGroup = cg;
        scene.add(cg);
    }

    _updateCruzReadout(lastCruzSection);
    _updateAncaReadout(lastAncaSection);

    // ── Volúmenes entre CRUZ y ANCA (3 variantes de piso) ──
    lastSecVolumes = null;
    if (lastCruzSection && lastAncaSection) {
        const xLo = Math.min(xCruz, xAnca), xHi = Math.max(xCruz, xAnca);
        const dist = xHi - xLo;
        const yBottom = bb.min.y - 1;
        lastDist = dist;

        // (1) piso en el mínimo de la verija
        // Piso de la verija: arranca en el fondo de la panza (yMin) y puede SUBIR
        // hacia adentro del barril hasta el tope del propio barril (yMax de la verija).
        let yFloorVer = yBottom;
        if (lastVerijaSection) {
            const yVerBot = lastVerijaSection.yMin;          // fondo de la panza
            const yVerCap = lastVerijaSection.yMax;          // tope del barril
            const rng = yVerCap - yVerBot;
            // Mapeo del slider (0–1):
            //   0%  = una altura de barril POR DEBAJO de la panza (afuera, abajo)
            //   50% = fondo de la panza (borde del barril, sin recortar)
            //   100%= tope del barril (totalmente adentro)
            const yLow = yVerBot - rng;
            yFloorVer = yLow + verijaRaiseFrac * (yVerCap - yLow);
        }
        lastVolVerija = _clippedVolumeLiters(currentGeometry, xLo, xHi, yFloorVer, 36);

        // (2) piso de corte a midFloorFrac*dist hacia abajo. La medición SIEMPRE
        // empieza en el punto más alto del diámetro de la CRUZ (no el lomo general).
        const yTop = lastCruzSection.yMax;
        const yFloorMid = yTop - midFloorFrac * dist;
        lastVolMid = _clippedVolumeLiters(currentGeometry, xLo, xHi, yFloorMid, 36);

        // (3) sin piso, tramo central (punto medio ± CENTRAL_HALF_FRAC*dist)
        const xMid = (xCruz + xAnca) / 2;
        lastVolCentral = _clippedVolumeLiters(currentGeometry, xMid - CENTRAL_HALF_FRAC * dist, xMid + CENTRAL_HALF_FRAC * dist, yBottom, 18);

        lastCutBox = { xLo, xHi, yFloor: yFloorVer };
        lastCutVolumeLiters = lastVolVerija;

        // Piso visual (verija): rectángulo horizontal en y=yFloorVer entre cruz y anca.
        const fgroup = new THREE.Group();
        const zA = lastVerijaSection ? lastVerijaSection.zMin : Math.min(lastCruzSection.zMin, lastAncaSection.zMin);
        const zB = lastVerijaSection ? lastVerijaSection.zMax : Math.max(lastCruzSection.zMax, lastAncaSection.zMax);
        const corners = [
            new THREE.Vector3(xLo, yFloorVer, zA), new THREE.Vector3(xHi, yFloorVer, zA),
            new THREE.Vector3(xHi, yFloorVer, zB), new THREE.Vector3(xLo, yFloorVer, zB),
        ];
        fgroup.add(new THREE.Mesh(
            new THREE.BufferGeometry().setFromPoints([corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]]),
            new THREE.MeshBasicMaterial({ color: 0xff7043, transparent: true, opacity: 0.28,
                side: THREE.DoubleSide, depthWrite: false })));
        fgroup.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(corners),
            new THREE.LineBasicMaterial({ color: 0xff7043 })));
        floorGroup = fgroup;
        fgroup.visible = true;  // el piso de la verija está SIEMPRE visible
        scene.add(fgroup);

        // Piso visual a 0.5*dist (vol_piso_meddist): rectángulo horizontal en y=yFloorMid.
        const zC1 = Math.min(lastCruzSection.zMin, lastAncaSection.zMin);
        const zC2 = Math.max(lastCruzSection.zMax, lastAncaSection.zMax);
        const mg = new THREE.Group();
        const mcorners = [
            new THREE.Vector3(xLo, yFloorMid, zC1), new THREE.Vector3(xHi, yFloorMid, zC1),
            new THREE.Vector3(xHi, yFloorMid, zC2), new THREE.Vector3(xLo, yFloorMid, zC2),
        ];
        mg.add(new THREE.Mesh(
            new THREE.BufferGeometry().setFromPoints([mcorners[0], mcorners[1], mcorners[2], mcorners[0], mcorners[2], mcorners[3]]),
            new THREE.MeshBasicMaterial({ color: 0xffd180, transparent: true, opacity: 0.28,
                side: THREE.DoubleSide, depthWrite: false })));
        mg.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(mcorners),
            new THREE.LineBasicMaterial({ color: 0xffd180 })));
        floorMidGroup = mg;
        const fmtgl = document.getElementById('floorMidToggle');
        mg.visible = fmtgl ? fmtgl.checked : false;
        scene.add(mg);

        // Tramo central visual (¼ a cada lado del punto medio): 2 planos verticales límite.
        const cg = new THREE.Group();
        [xMid - CENTRAL_HALF_FRAC * dist, xMid + CENTRAL_HALF_FRAC * dist].forEach(function(xb) {
            const sd = _sliceMeshAtX(currentGeometry, xb);
            if (!sd) return;
            const c = [
                new THREE.Vector3(xb, sd.yMin, sd.zMin), new THREE.Vector3(xb, sd.yMax, sd.zMin),
                new THREE.Vector3(xb, sd.yMax, sd.zMax), new THREE.Vector3(xb, sd.yMin, sd.zMax),
            ];
            cg.add(new THREE.Mesh(
                new THREE.BufferGeometry().setFromPoints([c[0], c[1], c[2], c[0], c[2], c[3]]),
                new THREE.MeshBasicMaterial({ color: 0x99ff66, transparent: true, opacity: 0.18,
                    side: THREE.DoubleSide, depthWrite: false })));
            cg.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(c),
                new THREE.LineBasicMaterial({ color: 0x99ff66 })));
        });
        sectionsGroup = cg;
        const ctgl = document.getElementById('centralToggle');
        cg.visible = ctgl ? ctgl.checked : false;
        scene.add(cg);
    } else {
        lastCutBox = null;
        lastCutVolumeLiters = null;
        lastVolVerija = lastVolMid = lastVolCentral = lastDist = null;
    }
    updateCutClip();
    _updateAncaCruzVolReadout();
}

function _updateAncaReadout(sec) {
    const set = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
    set('info3dAncaVert', sec ? sec.vert.toFixed(1) + ' cm' : '-');
    set('info3dAncaDepth', sec ? sec.depth.toFixed(1) + ' cm' : '-');
    set('info3dAncaPerim', sec ? sec.perim.toFixed(1) + ' cm' : '-');
    const posPct = document.getElementById('ancaPosLabel');
    if (posPct) posPct.textContent = Math.round(ancaFrac * 100) + '%';
}

function _updateAncaCruzVolReadout() {
    const set = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
    const cm = (v) => (v != null ? v.toFixed(1) + ' cm' : '-');
    const L = (v) => (v != null ? v.toFixed(1) + ' L' : '-');
    set('info3dDiamCruz', lastCruzSection ? cm(lastCruzSection.perim) : '-');
    set('info3dDiamAnca', lastAncaSection ? cm(lastAncaSection.perim) : '-');
    set('info3dDist', cm(lastDist));
    set('info3dDist15', cm(lastDist != null ? lastDist * 1.5 : null));
    set('info3dVolVerija', L(lastVolVerija));
    set('info3dVolMid', L(lastVolMid));
    set('info3dVolCentral', L(lastVolCentral));
    set('info3dCutVol', L(lastVolVerija));
}

function _updateGirthReadout(sec, dir) {
    const set = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
    set('info3dGirthVert', sec.vert.toFixed(1) + ' cm');
    set('info3dGirthDepth', sec.depth.toFixed(1) + ' cm');
    // Perímetro REAL = suma de los segmentos del contorno de la sección.
    set('info3dGirthPerim', sec.perim.toFixed(1) + ' cm');
    const posPct = document.getElementById('girthPosLabel');
    if (posPct) posPct.textContent = Math.round(girthFrac * 100) + '%';
    const dirLabel = document.getElementById('girthDirLabel');
    if (dirLabel) dirLabel.textContent =
        dir === 'left' ? '◄ izq' : (dir === 'right' ? 'der ►' : '? (asumido der)');
}

function _updateVerijaReadout(sec) {
    const set = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
    set('info3dVerijaVert', sec ? sec.vert.toFixed(1) + ' cm' : '-');
    set('info3dVerijaDepth', sec ? sec.depth.toFixed(1) + ' cm' : '-');
    set('info3dVerijaPerim', sec ? sec.perim.toFixed(1) + ' cm' : '-');
    const posPct = document.getElementById('verijaPosLabel');
    if (posPct) posPct.textContent = Math.round(verijaFrac * 100) + '%';
}

function _updateCruzReadout(sec) {
    const set = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
    set('info3dCruzVert', sec ? sec.vert.toFixed(1) + ' cm' : '-');
    set('info3dCruzDepth', sec ? sec.depth.toFixed(1) + ' cm' : '-');
    set('info3dCruzPerim', sec ? sec.perim.toFixed(1) + ' cm' : '-');
    const posPct = document.getElementById('cruzPosLabel');
    if (posPct) posPct.textContent = Math.round(cruzFrac * 100) + '%';
}

// Sincroniza los controles del recorte de cresta con el estado crestTrim.
function _syncCrestTrimUI() {
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v; };
    const txt = (id, t) => { const el = document.getElementById(id); if (el) el.textContent = t; };
    const tg = document.getElementById('crestTrimToggle');
    if (tg) tg.checked = crestTrim.on;
    set('crestCeilSlider', Math.round(crestTrim.ceil * 100));
    set('crestXLoSlider', Math.round(crestTrim.xLo * 100));
    set('crestXHiSlider', Math.round(crestTrim.xHi * 100));
    txt('crestCeilLabel', Math.round(crestTrim.ceil * 100) + '%');
    txt('crestXLoLabel', Math.round(crestTrim.xLo * 100) + '%');
    txt('crestXHiLabel', Math.round(crestTrim.xHi * 100) + '%');
}

// ── Fit camera to bounding box ──

function fitCameraToModel(geometry) {
    const bbox = geometry.boundingBox;
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let dist = maxDim / (2 * Math.tan(fov / 2));
    dist *= 1.5;

    camera.position.set(dist * 0.7, dist * 0.5, dist);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    controls.update();
}

// ── Render mode switching ──

function applyRenderMode() {
    // La malla 3D (_3d.ply) es el único modelo: se muestra según renderMode.
    if (currentMesh)      currentMesh.visible      = (renderMode === 'solid');
    if (currentWireframe) currentWireframe.visible = (renderMode === 'wireframe');
    if (currentPoints)    currentPoints.visible    = (renderMode === 'points');
}

// ── Set camera view ──

function setVista(nombre) {
    if (!currentMesh) return;
    const geometry = currentMesh.geometry;
    const bbox = geometry.boundingBox;
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim * 2;

    controls.target.set(0, 0, 0);

    switch (nombre) {
        case 'lateral':
            camera.position.set(dist, 0, 0);
            break;
        case 'frontal':
            camera.position.set(0, 0, dist);
            break;
        case 'superior':
            camera.position.set(0, dist, 0);
            break;
        case '3d':
        default:
            camera.position.set(dist * 0.5, dist * 0.35, dist * 0.7);
            break;
    }
    camera.lookAt(0, 0, 0);
    controls.update();
}

// ── Update info panel ──

function updateInfoPanel(m) {
    const panel = document.getElementById('viewer3dInfo');
    if (!panel) return;

    const show = m.peso_kg || m.largo_cm || m.alto_cm;
    panel.style.display = show ? 'block' : 'none';

    const set = (id, val, unit) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val != null ? `${val} ${unit}` : '-';
    };
    set('info3dPeso', m.peso_kg, 'kg');
    set('info3dLargo', m.largo_cm, 'cm');
    set('info3dAlto', m.alto_cm, 'cm');
    set('info3dVolumen', m.volumen_litros, 'L');
    set('info3dSuperficie', m.superficie_cm2, 'cm²');
}

// ── Public API: load model from external code ──

// ── Recalculate with real height ──

async function recalcularConAltura() {
    const input = document.getElementById('inputAlturaCm');
    if (!input) return;
    const altura = parseFloat(input.value);
    if (!altura || altura <= 0 || altura > 250) {
        alert('Ingresa una altura valida (30-250 cm).');
        return;
    }

    if (!currentModelId) {
        alert('No hay modelo seleccionado.');
        return;
    }

    try {
        const resp = await fetch(`/api/modelo3d/${currentModelId}/recalcular`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ altura_cm: altura }),
        });
        if (!resp.ok) {
            const err = await resp.json();
            alert('Error: ' + (err.error || resp.statusText));
            return;
        }
        const data = await resp.json();
        updateInfoPanel(data);
    } catch (e) {
        console.error('Error recalculando:', e);
        alert('Error al recalcular: ' + e.message);
    }
}

// Expuesta para que engine.js pueda refrescar la lista tras generar modelos en vivo
window.loadModelosDisponibles = loadModelosDisponibles;

window.viewer3dLoadModel = function(url, infoData, modelId) {
    // Ensure viewer is initialized
    if (!renderer) {
        const container = document.getElementById('viewer3dContainer');
        if (container) initViewer3D();
    }

    currentModelId = modelId || null;
    loadModelo(url);

    if (infoData) {
        updateInfoPanel({
            peso_kg: infoData.peso_kg,
            largo_cm: infoData.largo_cm,
            alto_cm: infoData.alto_cm,
            volumen_litros: infoData.volumen_litros,
            superficie_cm2: infoData.superficie_cm2,
        });
    }

    // Refresh model selector, selecting the new model (without auto-loading index 0)
    loadModelosDisponibles(modelId || null);
};

// ── Event bindings ──

document.addEventListener('DOMContentLoaded', function() {
    // Only init if the viewer container exists
    if (!document.getElementById('viewer3dContainer')) return;

    initViewer3D();

    // Model selector
    const sel = document.getElementById('modelSelector');
    if (sel) {
        sel.addEventListener('change', function() {
            const val = this.value;
            if (val === '' || val == null) {
                clearModel();  // placeholder seleccionado → vaciar el viewer
                return;
            }
            const idx = parseInt(val);
            if (!isNaN(idx)) loadModelByIndex(idx);
        });
    }


    // View buttons
    document.querySelectorAll('.btn-vista').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.btn-vista').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            setVista(this.dataset.vista);
        });
    });

    // Render mode buttons
    document.querySelectorAll('.btn-render').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.btn-render').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            renderMode = this.dataset.mode;
            applyRenderMode();
        });
    });

    // Reset view
    const btnReset = document.getElementById('btnResetVista');
    if (btnReset) {
        btnReset.addEventListener('click', function() {
            if (currentMesh && currentMesh.geometry) {
                fitCameraToModel(currentMesh.geometry);
            }
        });
    }

    // Recalculate with real height
    const btnRecalc = document.getElementById('btnRecalcularPeso');
    if (btnRecalc) {
        btnRecalc.addEventListener('click', recalcularConAltura);
    }
    // Also allow Enter key in height input
    const inputAltura = document.getElementById('inputAlturaCm');
    if (inputAltura) {
        inputAltura.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') recalcularConAltura();
        });
    }

    // Slider de posición del plano del diámetro ("detrás de las manos")
    const girthSlider = document.getElementById('girthPosSlider');
    if (girthSlider) {
        girthSlider.addEventListener('input', function() {
            girthFrac = Math.max(0, Math.min(0.5, parseFloat(this.value) / 100));
            updateGirthMeasurement();
        });
    }
    // Toggle de visibilidad de la medición
    const girthToggle = document.getElementById('girthToggle');
    if (girthToggle) {
        girthToggle.addEventListener('change', function() {
            if (diameterGroup) diameterGroup.visible = this.checked;
        });
    }

    // Guardar el punto del diámetro torácico elegido (etiqueta + persistencia)
    const girthSaveBtn = document.getElementById('girthSaveBtn');
    if (girthSaveBtn) {
        const _label = girthSaveBtn.innerHTML;
        girthSaveBtn.addEventListener('click', async function() {
            if (!currentModelId) { alert('Cargá un modelo primero.'); return; }
            const sec = lastGirthSection || {};
            const payload = {
                girth_frac: girthFrac,
                vert_cm: (sec.vert != null) ? +sec.vert.toFixed(2) : null,
                depth_cm: (sec.depth != null) ? +sec.depth.toFixed(2) : null,
                perim_cm: (sec.perim != null) ? +sec.perim.toFixed(2) : null,
            };
            girthSaveBtn.disabled = true;
            try {
                const r = await fetch(`/api/modelo3d/${currentModelId}/girth`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const j = await r.json();
                if (j.success) {
                    if (currentModelMeta) currentModelMeta.girth_frac_manual = girthFrac;
                    girthSaveBtn.innerHTML = '<i class="fas fa-check"></i> Guardado ' +
                        Math.round(girthFrac * 100) + '%';
                    setTimeout(() => { girthSaveBtn.innerHTML = _label; }, 1600);
                } else {
                    alert('Error al guardar: ' + (j.error || '?'));
                }
            } catch (e) {
                alert('Error: ' + e);
            } finally {
                girthSaveBtn.disabled = false;
            }
        });
    }

    // ── Piso VERIJA: un solo slider vertical (entra/sale del barril) + guardar ──
    // La posición a lo largo del cuerpo (verijaFrac) queda fija en el valor guardado;
    // solo se mueve el piso verticalmente con verijaRaiseSlider.
    // Slider de SUBIDA del piso de la verija (hacia adentro del barril).
    const verijaRaiseSlider = document.getElementById('verijaRaiseSlider');
    if (verijaRaiseSlider) {
        verijaRaiseSlider.addEventListener('input', function() {
            verijaRaiseFrac = Math.max(0, Math.min(1, parseFloat(this.value) / 100));
            const lbl = document.getElementById('verijaRaiseLabel');
            if (lbl) lbl.textContent = Math.round(verijaRaiseFrac * 100) + '%';
            updateGirthMeasurement();
        });
    }
    const verijaSaveBtn = document.getElementById('verijaSaveBtn');
    if (verijaSaveBtn) {
        const _vlabel = verijaSaveBtn.innerHTML;
        verijaSaveBtn.addEventListener('click', async function() {
            if (!currentModelId) { alert('Cargá un modelo primero.'); return; }
            const sec = lastVerijaSection || {};
            const payload = {
                verija_frac: verijaFrac,
                verija_raise: verijaRaiseFrac,
                vert_cm: (sec.vert != null) ? +sec.vert.toFixed(2) : null,
                depth_cm: (sec.depth != null) ? +sec.depth.toFixed(2) : null,
                perim_cm: (sec.perim != null) ? +sec.perim.toFixed(2) : null,
            };
            verijaSaveBtn.disabled = true;
            try {
                const r = await fetch(`/api/modelo3d/${currentModelId}/verija`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const j = await r.json();
                if (j.success) {
                    if (currentModelMeta) {
                        currentModelMeta.verija_frac_manual = verijaFrac;
                        currentModelMeta.verija_raise_manual = verijaRaiseFrac;
                    }
                    verijaSaveBtn.innerHTML = '<i class="fas fa-check"></i> Guardado ' +
                        Math.round(verijaFrac * 100) + '%';
                    setTimeout(() => { verijaSaveBtn.innerHTML = _vlabel; }, 1600);
                } else {
                    alert('Error al guardar verija: ' + (j.error || '?'));
                }
            } catch (e) {
                alert('Error: ' + e);
            } finally {
                verijaSaveBtn.disabled = false;
            }
        });
    }

    // ── Diámetro CRUZ: slider (desde el frente, auto cruz_pose) + guardar ──
    const cruzSlider = document.getElementById('cruzPosSlider');
    if (cruzSlider) {
        cruzSlider.addEventListener('input', function() {
            cruzFrac = Math.max(0, Math.min(0.5, parseFloat(this.value) / 100));
            updateGirthMeasurement();
        });
    }
    const cruzSaveBtn = document.getElementById('cruzSaveBtn');
    if (cruzSaveBtn) {
        const _clabel = cruzSaveBtn.innerHTML;
        cruzSaveBtn.addEventListener('click', async function() {
            if (!currentModelId) { alert('Cargá un modelo primero.'); return; }
            const sec = lastCruzSection || {};
            const payload = {
                cruz_frac: cruzFrac,
                vert_cm: (sec.vert != null) ? +sec.vert.toFixed(2) : null,
                depth_cm: (sec.depth != null) ? +sec.depth.toFixed(2) : null,
                perim_cm: (sec.perim != null) ? +sec.perim.toFixed(2) : null,
            };
            cruzSaveBtn.disabled = true;
            try {
                const r = await fetch(`/api/modelo3d/${currentModelId}/cruz`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const j = await r.json();
                if (j.success) {
                    if (currentModelMeta) currentModelMeta.cruz_frac_manual = cruzFrac;
                    cruzSaveBtn.innerHTML = '<i class="fas fa-check"></i> Guardado ' +
                        Math.round(cruzFrac * 100) + '%';
                    setTimeout(() => { cruzSaveBtn.innerHTML = _clabel; }, 1600);
                } else {
                    alert('Error al guardar cruz: ' + (j.error || '?'));
                }
            } catch (e) {
                alert('Error: ' + e);
            } finally {
                cruzSaveBtn.disabled = false;
            }
        });
    }

    // ── Diámetro ANCA: slider (desde el fondo) + guardar ──
    const ancaSlider = document.getElementById('ancaPosSlider');
    if (ancaSlider) {
        ancaSlider.addEventListener('input', function() {
            ancaFrac = Math.max(0, Math.min(0.5, parseFloat(this.value) / 100));
            updateGirthMeasurement();
        });
    }
    const ancaSaveBtn = document.getElementById('ancaSaveBtn');
    if (ancaSaveBtn) {
        const _alabel = ancaSaveBtn.innerHTML;
        ancaSaveBtn.addEventListener('click', async function() {
            if (!currentModelId) { alert('Cargá un modelo primero.'); return; }
            const sec = lastAncaSection || {};
            const payload = {
                anca_frac: ancaFrac,
                vert_cm: (sec.vert != null) ? +sec.vert.toFixed(2) : null,
                depth_cm: (sec.depth != null) ? +sec.depth.toFixed(2) : null,
                perim_cm: (sec.perim != null) ? +sec.perim.toFixed(2) : null,
            };
            ancaSaveBtn.disabled = true;
            try {
                const r = await fetch(`/api/modelo3d/${currentModelId}/anca`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const j = await r.json();
                if (j.success) {
                    if (currentModelMeta) currentModelMeta.anca_frac_manual = ancaFrac;
                    ancaSaveBtn.innerHTML = '<i class="fas fa-check"></i> Guardado ' +
                        Math.round(ancaFrac * 100) + '%';
                    setTimeout(() => { ancaSaveBtn.innerHTML = _alabel; }, 1600);
                } else {
                    alert('Error al guardar anca: ' + (j.error || '?'));
                }
            } catch (e) {
                alert('Error: ' + e);
            } finally {
                ancaSaveBtn.disabled = false;
            }
        });
    }

    // Toggle del PISO (mínimo verija). Recalcula el volumen: marcado = con piso
    // (recorta panza), desmarcado = sección completa.
    const floorToggle = document.getElementById('floorToggle');
    if (floorToggle) {
        floorToggle.addEventListener('change', function() {
            updateGirthMeasurement();
        });
    }
    // Toggle del PISO de corte (vol_piso_meddist): solo visual.
    const floorMidToggle = document.getElementById('floorMidToggle');
    if (floorMidToggle) {
        floorMidToggle.addEventListener('change', function() {
            if (floorMidGroup) floorMidGroup.visible = this.checked;
        });
    }
    // Input del % de corte: el usuario pone el %, recalculamos volumen y la línea
    // se reposiciona (medido hacia abajo desde lo más alto del diámetro de la cruz).
    const midFloorInput = document.getElementById('midFloorPctInput');
    if (midFloorInput) {
        const _applyMidFloor = function() {
            let p = parseFloat(midFloorInput.value);
            if (isNaN(p)) return;
            p = Math.max(0, Math.min(100, p));
            midFloorFrac = p / 100;
            // Aseguramos que la línea de corte esté visible al ajustar el %.
            const ftgl = document.getElementById('floorMidToggle');
            if (ftgl && !ftgl.checked) { ftgl.checked = true; }
            updateGirthMeasurement();
        };
        midFloorInput.addEventListener('input', _applyMidFloor);
        midFloorInput.addEventListener('change', _applyMidFloor);
    }
    // Guardar el % del piso de corte elegido (persistencia + etiqueta).
    const corteSaveBtn = document.getElementById('corteSaveBtn');
    if (corteSaveBtn) {
        const _ctlabel = corteSaveBtn.innerHTML;
        corteSaveBtn.addEventListener('click', async function() {
            if (!currentModelId) { alert('Cargá un modelo primero.'); return; }
            const payload = {
                corte_frac: midFloorFrac,
                vol_corte_litros: (lastVolMid != null) ? +lastVolMid.toFixed(2) : null,
            };
            corteSaveBtn.disabled = true;
            try {
                const r = await fetch(`/api/modelo3d/${currentModelId}/corte`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const j = await r.json();
                if (j.success) {
                    if (currentModelMeta) currentModelMeta.corte_frac_manual = midFloorFrac;
                    corteSaveBtn.innerHTML = '<i class="fas fa-check"></i> Guardado ' +
                        Math.round(midFloorFrac * 100) + '%';
                    setTimeout(() => { corteSaveBtn.innerHTML = _ctlabel; }, 1600);
                } else {
                    alert('Error al guardar corte: ' + (j.error || '?'));
                }
            } catch (e) {
                alert('Error: ' + e);
            } finally {
                corteSaveBtn.disabled = false;
            }
        });
    }
    // Toggle del TRAMO CENTRAL (¼ a cada lado del punto medio): solo visual.
    const centralToggle = document.getElementById('centralToggle');
    if (centralToggle) {
        centralToggle.addEventListener('change', function() {
            if (sectionsGroup) sectionsGroup.visible = this.checked;
        });
    }

    // ── RECORTE DE CRESTA (no destructivo, localizado) ──
    const crestTrimToggle = document.getElementById('crestTrimToggle');
    if (crestTrimToggle) {
        crestTrimToggle.addEventListener('change', function() {
            crestTrim.on = this.checked;
            updateGirthMeasurement();
        });
    }
    const _bindCrest = (id, key, max, lblId) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('input', function() {
            crestTrim[key] = Math.max(0, Math.min(max, parseFloat(this.value) / 100));
            const lbl = document.getElementById(lblId);
            if (lbl) lbl.textContent = Math.round(crestTrim[key] * 100) + '%';
            if (crestTrim.on) updateGirthMeasurement();
        });
    };
    _bindCrest('crestCeilSlider', 'ceil', 0.5, 'crestCeilLabel');
    _bindCrest('crestXLoSlider', 'xLo', 1.0, 'crestXLoLabel');
    _bindCrest('crestXHiSlider', 'xHi', 1.0, 'crestXHiLabel');

    const crestSaveBtn = document.getElementById('crestTrimSaveBtn');
    if (crestSaveBtn) {
        const _crlabel = crestSaveBtn.innerHTML;
        crestSaveBtn.addEventListener('click', async function() {
            if (!currentModelId) { alert('Cargá un modelo primero.'); return; }
            const payload = {
                on: crestTrim.on,
                x_lo: +crestTrim.xLo.toFixed(4),
                x_hi: +crestTrim.xHi.toFixed(4),
                ceil: +crestTrim.ceil.toFixed(4),
            };
            crestSaveBtn.disabled = true;
            try {
                const r = await fetch(`/api/modelo3d/${currentModelId}/crest_trim`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const j = await r.json();
                if (j.success) {
                    if (currentModelMeta) currentModelMeta.crest_trim = { ...payload };
                    crestSaveBtn.innerHTML = '<i class="fas fa-check"></i> Guardado';
                    setTimeout(() => { crestSaveBtn.innerHTML = _crlabel; }, 1600);
                } else {
                    alert('Error al guardar recorte: ' + (j.error || '?'));
                }
            } catch (e) {
                alert('Error: ' + e);
            } finally {
                crestSaveBtn.disabled = false;
            }
        });
    }

});
