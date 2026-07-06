# Galería "frames del barril usados para el 3D" — REMOVIDA 2026-07-06

Sección visual (`#barril3dHeader` + `#barril3dGallery`, `renderBarril3DGallery()`)
que mostraba los thumbnails de los frames de barril elegibles tras cada barrido
de "Detectar pasadas". Display puro: no alimentaba ningún cálculo.

Se conservaron intactos:
- `barril_volumen_litros` por frame y el promedio "Barril: X L" del resumen
  (validado 2026-07-06 contra peso real como el mejor estimador en vivo).
- `barril_contour_norm` y el consenso multi-frame (`_computeBarrilConsensus`,
  botón "Generar modelo 3D", `/generate_3d_consensus`) — rama aparte.
