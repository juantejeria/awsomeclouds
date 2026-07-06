# Trayectoria de pezuñas (feet map) + Regla de medición — REMOVIDO 2026-07-06

Feature experimental que acumulaba los picos del borde inferior de la silueta
(`find_peaks` sobre la máscara de silueta_seg) en cada frame del barrido de
"Detectar pasadas", pintaba la trayectoria temporal de las pezuñas en un
canvas, derivaba una "línea de mínimos" (pezuñas plantadas) y de su cruce con
la línea de piso obtenía una escala cm/px para la **Regla de medición** (Fase
C). Se guardaba como `feet_map.png`/`feet_map.json` junto a los 21 frames.

No alimentaba ningún cálculo de altura/volumen/peso (la detección de pezuñas
por frame que ajusta el borde inferior del bbox en `detect_cow_fast` **sigue
activa** — solo se removió la acumulación de trayectoria y su UI). Se quitó
para no computar `find_peaks` + canvas en cada frame del barrido sin uso.

**La Regla de medición se removió junto con esto** porque su única fuente de
escala era el cruce piso × línea de mínimos (`rulerScale.source =
'feet-floor-cross'`). Si se quiere resucitar, conviene rehacerla con la escala
de los postes (locked_reference), que es más robusta.

## Qué se removió y de dónde

| Archivo original | Qué | Copia acá |
|---|---|---|
| `static/js/engine.js` | `_segIntersect`, `_filterFeetByMode`, `renderFeetTrajectoryMap`, `renderRulerCard`, `rulerCaptureVideoFrame`, `rulerLoadThumb`, `rulerDraw`, `rulerResetPoints`, campos AppState, handlers, dibujo de puntos verdes, paso 8 de processFolder | `engine_feet_trajectory.js` |
| `templates/index.html` | `#feetTrajectoryCard`, `#rulerCard`, `#savedFeetMapWrap` | `index_feet_ruler.html` |
| `app.py` | rutas `/saved_feet_map/<folder>` y `/save_feet_map`, bloque `find_peaks` en `detect_cow_fast`, campo `feet_points` de la respuesta, `feet_map_exists` del listado | `app_feet_routes_y_deteccion.py` |

Se conservó `_detectMovementDir` (engine.js) porque el consenso 3D lo usa para
determinar `barril_dir`.

Los `feet_map.png`/`feet_map.json` ya guardados en `checkpoints/<dataset>/<individuo>/`
no se tocaron; simplemente ya no se muestran ni se generan nuevos.

Para restaurar: `git log --follow` sobre los archivos originales, o partir de
las copias de esta carpeta.
