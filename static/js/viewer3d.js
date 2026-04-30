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
// Volumen / rebanadas
let volumenMesh = null;
let volumenWireframe = null;
let modelosData = [];
let currentModelId = null; // ID of currently loaded model (e.g. 'vaca1')
let renderMode = 'solid'; // solid | wireframe | points
let layerMode = 'contorno'; // contorno | volumen | ambos
let animFrameId = null;
let _pendingLoads = 0;  // contador de cargas en vuelo → pausa render durante carga

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
        modelosData.forEach((m, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = m.nombre + (m.peso_kg ? ` (${m.peso_kg} kg)` : '');
            sel.appendChild(opt);
            if (selectModelId && m.id === selectModelId) {
                targetIdx = i;
            }
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
    _pendingLoads++;
    clearModel();
    // Cache-bust: asegura recarga del PLY cuando se regenera (el PLYLoader
    // cachea por URL — sin esto el viewer mostraría el modelo viejo aunque
    // el archivo en disco sea nuevo).
    const _t = Date.now();
    loadShellPLY(`/api/modelo3d/${m.id}/${shellFile}?t=${_t}`, function() {
        _pendingLoads--;
    });
    if (m.ply_volumen) {
        _pendingLoads++;
        loadVolumenPLY(`/api/modelo3d/${m.id}/${m.ply_volumen}?t=${_t}`, function() {
            _pendingLoads--;
        });
    }
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

function loadVolumenPLY(url, onDone) {
    const loader = new PLYLoader();
    loader.load(url, function(geometry) {
        try {
        geometry.computeVertexNormals();
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);

        const hasColors = geometry.hasAttribute('color');

        const solidMat = new THREE.MeshPhongMaterial({
            color: hasColors ? 0xffffff : 0xe68a2e,
            vertexColors: hasColors,
            side: THREE.DoubleSide,
            shininess: 15,
            transparent: true,
            opacity: 0.92,
        });
        volumenMesh = new THREE.Mesh(geometry, solidMat);
        scene.add(volumenMesh);

        const wireMat = new THREE.MeshBasicMaterial({
            color: 0xcc6a16,
            wireframe: true,
            transparent: true,
            opacity: 0.85,
        });
        volumenWireframe = new THREE.Mesh(geometry, wireMat);
        scene.add(volumenWireframe);

        applyLayerMode();
        } finally {
            if (typeof onDone === 'function') onDone();
        }
    },
    undefined,
    function(error) {
        console.error('Error loading volumen PLY:', error);
        if (typeof onDone === 'function') onDone();
    });
}

// ── Clear current model ──

function clearModel() {
    if (currentMesh) { scene.remove(currentMesh); currentMesh = null; }
    if (currentWireframe) { scene.remove(currentWireframe); currentWireframe = null; }
    if (currentPoints) { scene.remove(currentPoints); currentPoints = null; }
    if (volumenMesh) { scene.remove(volumenMesh); volumenMesh = null; }
    if (volumenWireframe) { scene.remove(volumenWireframe); volumenWireframe = null; }
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
    applyLayerMode();
}

function applyLayerMode() {
    const showShell = (layerMode === 'contorno' || layerMode === 'ambos');
    const showVol   = (layerMode === 'volumen'  || layerMode === 'ambos');

    // Shell: respeta renderMode
    if (currentMesh)      currentMesh.visible      = showShell && (renderMode === 'solid');
    if (currentWireframe) currentWireframe.visible = showShell && (renderMode === 'wireframe');
    if (currentPoints)    currentPoints.visible    = showShell && (renderMode === 'points');

    // En "ambos": shell semi-transparente para ver las rebanadas adentro
    if (currentMesh && currentMesh.material) {
        const mat = currentMesh.material;
        if (layerMode === 'ambos') {
            mat.transparent = true;
            mat.opacity = 0.25;
        } else {
            mat.transparent = false;
            mat.opacity = 1.0;
        }
        mat.needsUpdate = true;
    }

    // Volumen: solid o wireframe (no soporta points)
    if (volumenMesh) volumenMesh.visible = showVol && (renderMode !== 'wireframe');
    if (volumenWireframe) volumenWireframe.visible = showVol && (renderMode === 'wireframe');
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

    // Layer buttons (Contorno / Volumen / Ambos)
    document.querySelectorAll('.btn-layer').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.btn-layer').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            layerMode = this.dataset.layer;
            applyLayerMode();
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

});
