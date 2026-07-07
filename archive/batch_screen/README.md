# Batch screening de video — REMOVIDO 2026-07-06

Flujo que subía el video completo a `/batch_screen`, muestreaba frames cada
`frame_interval`, corría `estimate_weight` por frame (con la escala cm/px de la
calibración de 2 postes) y devolvía resultados por SSE. El front renderizaba
progreso, resumen estadístico (promedio/mediana/desvío), tabla ordenable por
frame, galería de thumbnails anotados con modal, y exportación a CSV.

Estaba **muerto**: su botón de arranque `#btnScreenVideo` (y toda la sección
`#batchScreeningSection`, `#screenInterval`, `#screenMinScore`, `#screeningHint`)
ya no existían en `templates/index.html`, así que `startBatchScreening()` era
inalcanzable. Solo quedaba el botón `#btnCancelScreening` en el HTML, sin flujo
que lo mostrara. Fue reemplazado en la práctica por el flujo de **"Detectar
pasadas"** (`startDetectPassings` / `processPassingLoop`), que hace el barrido
del lado del cliente frame a frame contra `/analyze_frame`.

## Qué se removió y de dónde

| Archivo original | Qué | Copia acá |
|---|---|---|
| `static/js/engine.js` | Campos AppState (`screeningResults`, `screeningActive`, `screeningAbortController`, `screeningSortCol`, `screeningSortAsc`, `screeningTableResults`); funciones `updateScreeningSection`, `startBatchScreening`, `handleScreeningEvent`, `updateScreeningProgress`, `finishScreening`, `renderScreeningResults`, `buildStatCard`, `buildSortableTh`, `buildScreeningTableBody`, `sortScreeningTable`, `buildFrameCard`, `showScreeningFullImage`, `exportScreeningCSV`, `cancelScreening`; llamadas a `updateScreeningSection()` desde `updateSelectionSummary` y `finishModelo3D`; resets en `initVideoPlayer`; `$('#btnCancelScreening').hide()` en `startDetectPassings`; bindings de `#btnScreenVideo` y `#btnCancelScreening` | `engine_batch_screen.js` |
| `templates/index.html` | Botón `#btnCancelScreening` dentro del header de `#screeningCard` | `index_btn_cancel_screening.html` |
| `app.py` | Ruta `/batch_screen` (streaming SSE), `import statistics`, `get_weight_range` del import de `core.breed_coefficients` | `app_batch_screen_route.py` |

## Qué se conservó (COMPARTIDO con "Detectar pasadas")

- `#screeningCard`, `#screeningProgress`, `#screeningProgressBar`,
  `#screeningProgressText`, `#screeningProgressCount`, `#screeningSummary` y
  `#screeningGallery` en `index.html`: son los contenedores donde
  `processPassingLoop` / `finalizePassingResults` pintan el progreso, el
  resumen y la galería del flujo activo. Solo se quitó el botón Cancelar.
- Los estilos CSS `screening-*` que siguen usándose desde el flujo activo
  (p. ej. `screening-error-banner` también lo usa `finishModelo3D`).
- `get_weight_range` sigue existiendo en `core/breed_coefficients.py`; solo se
  quitó su import en `app.py` porque nadie más lo usaba ahí.

Nota: `updateScreeningSection` también tocaba `#btnModelo3DHibrido`,
`#btnModelo3DSfm` y `#modelo3dOptions` — esos IDs tampoco existen en
`index.html` (restos del flujo 3D archivado en `archive/ui_3d_consensus/`),
así que quitarla no afecta nada vivo. Los bindings/usos de esos botones que
quedan en `engine.js` (fuera de este flujo) no se tocaron.

`docs/RESUMEN_PESO.md` todavía menciona `/batch_screen` como endpoint; no se
actualizó la documentación en esta pasada.

Para restaurar: `git log --follow` sobre los archivos originales, o partir de
las copias de esta carpeta.
