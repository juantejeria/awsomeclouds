# Consenso 3D dentro de la UI — REMOVIDO 2026-07-06

Camino paralelo de generación 3D que vivía en la UI y que el flujo real no
usa (confirmado por Juan: el flujo es **Guardar 21 frames → pipeline offline
(`pipeline/procesar_21_frames_filtrado.py`, barril_seg_v8) → corrección de
cresta**).

## Qué se removió

| Pieza | Dónde vivía | Copia acá |
|---|---|---|
| `generate3DFromResult()` — botón "Generar modelo 3D" del resumen del barrido | engine.js | `engine_ui_3d_consensus.js` |
| `processFolder()` + `renderProcessedFrameThumb()` — botón "Procesar carpeta → 3D" | engine.js | `engine_ui_3d_consensus.js` |
| Variantes B (p75), C (maxW_p75) y E (envelope) del consenso en vivo — se calculaban en cada barrido solo para experimentos | engine.js | `engine_ui_3d_consensus.js` |
| Botón "Procesar carpeta → 3D" | templates/index.html | `index_btn_procesar_carpeta.html` |
| Rutas `/generate_3d_consensus` y `/generate_3d_from_frame` | app.py | `app_rutas_3d_consensus.py` |

## Qué se conservó (importante)

- **`_computeBarrilConsensus(..., 'median')`**: el "Barril: X L" del resumen
  del barrido ES el consenso mediana (A_median) con filtrado de outliers —
  validado el 2026-07-06 contra peso real (ternero 125 kg → 127.3 kg estimado
  con altura real). Es el mejor estimador en vivo del sistema.
- `barril_contour_norm` y `barril_volumen_litros` por frame (backend).
- "Guardar 21 frames" y "Ver frames guardados" — intactos.
- `downloadResultCard` ("Descargar resultado") — intacto.

## Pendiente relacionado

El flujo **batch_screen** (`startBatchScreening`, `handleScreeningEvent`,
`finishScreening`, `renderScreeningResults`, `updateScreeningSection` en
engine.js + ruta `/batch_screen` en app.py) también está muerto — su botón
`#btnScreenVideo` no existe en el HTML. No se removió en esta pasada porque
comparte el `#screeningCard` con la galería de pasadas y requiere un corte
más cuidadoso.
