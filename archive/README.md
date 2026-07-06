# archive/

Código y resultados que **no forman parte del pipeline activo**. Nada fue borrado — todo se movió acá con `git mv` (historia preservada) para dejar la raíz del repo solo con lo que se usa.

Si algo de acá vuelve a ser necesario, moverlo de vuelta a la raíz (los scripts asumen ejecutarse desde la raíz del proyecto; la mayoría usa rutas relativas).

| Carpeta | Contenido |
|---|---|
| `legacy_face/` | Proyecto original de reconocimiento facial (entrenamiento, Grad-CAM, SSIM, visualizaciones). **Ojo:** la *inferencia* facial sigue activa — `testing.py` y `keras_vggface/` quedan en la raíz porque `app.py` los importa. |
| `old_pipelines/` | Generadores 3D superados por el flujo v8 (`modelo_*.py`, backups de `generar_modelos3d_grandes`, `procesar_21_frames` sin filtrar, reparación de mallas, carcasa). |
| `one_offs/` | Scripts de mantenimiento/backfill atados a un dataset o fecha puntual (22abril, 26marz, mayo…): regenerar frames, escribir cruz_frac retroactivo, recortes, limpiezas. |
| `analysis/` | Estudios y tablas exploratorias (verija, girth, perímetros, diámetros, constantes de peso, validaciones). Generaban los CSV/XLSX de `results/`. |
| `debug/` | Scripts de depuración visual y experimentos (inpainting, franjas, muestras de keypoints). |
| `batch_scripts/` | Corridas batch `.sh` fechadas (13/14/20 mayo, 6mayo, v7…). |
| `results/` | Salidas generadas: RESULTADOS_COMPLETOS.*, tablas CSV/XLSX, análisis de constantes. Todo regenerable con el pipeline actual. |
| `models_old/` | Pesos YOLO viejos (`barril_seg` v3–v7 y `_prev_`). Activos en raíz: `barril_seg.pt`, `barril_seg_v8.pt`, `silueta_seg.pt`, `cruz_pose.pt`. |
| `backups/` | `backups_ply/`, `backups_resumen/`, `.bak` de alturas. |
