"""Aplica la reparación de hueco a un modelo 3D existente:
- _lateral.ply  → reemplaza con malla 2D reparada (vértices nuevos pintados naranja)
- _3d.ply       → regenera shell simétrico (z+/z-) con depth=0 en el rim
                  (el volumen reportado es su volumen encerrado)

Uso:
    python aplicar_reparar_test2barril.py [NOMBRE_MODELO]

Default: Test2BarrilV1. El modelo debe existir en
    output_modelos3d_live/<nombre>/<nombre>_{lateral,3d}.ply

Backups quedan en archivo *_orig.ply (solo si no existían ya).
"""
import os
import sys
import shutil
import numpy as np

from reparar_hueco_malla import (
    leer_ply_plano, escribir_ply_plano, reparar,
)
from generar_modelos3d_grandes import guardar_ply, volumen_malla_cerrada

COLOR_REPARADO = np.array([255, 140, 0], dtype=np.uint8)  # naranja

NAME = sys.argv[1] if len(sys.argv) > 1 else 'Test2BarrilV1'
MODEL_DIR = f'output_modelos3d_live/{NAME}'
if not os.path.isdir(MODEL_DIR):
    print(f"[error] no existe la carpeta {MODEL_DIR}")
    sys.exit(1)

src_lateral = f'{MODEL_DIR}/{NAME}_lateral.ply'
src_3d      = f'{MODEL_DIR}/{NAME}_3d.ply'

# Backups (solo si no existen)
for src in (src_lateral, src_3d):
    bak = src.replace('.ply', '_orig.ply')
    if not os.path.exists(bak):
        shutil.copy(src, bak)
        print(f"backup: {bak}")

# 1. Leer lateral original
verts, colors, faces, comments = leer_ply_plano(src_lateral)
n_orig = len(verts)
print(f"[lateral] in: {n_orig} verts, {len(faces)} faces")

# 2. Reparar
new_v, new_f, new_c = reparar(verts, faces, colors)
n_total = len(new_v)
n_reparados = n_total - n_orig
print(f"[lateral] out: {n_total} verts, {len(new_f)} faces ({n_reparados} nuevos)")

# 3. Pintar vértices reparados de naranja (índices >= n_orig)
if new_c is None:
    new_c = np.full((n_total, 3), [139, 90, 43], dtype=np.uint8)
if n_reparados > 0:
    new_c[n_orig:] = COLOR_REPARADO
    print(f"[lateral] {n_reparados} vértices pintados de naranja {COLOR_REPARADO.tolist()}")

# 4. Escribir lateral reparado (z=0)
escribir_ply_plano(src_lateral, new_v, new_c, new_f, comments)
print(f"[lateral] guardado en {src_lateral}")

# 5. Regenerar _3d.ply con simetría espejo en Z
# guardar_ply espera pts_cm en (x, y) shape (N, 2 o más). Usa profundidad_eliptica
# internamente cuando simetrico=True.
pts2d = new_v[:, :2]
escala_info_3d = next(
    (c.replace('comment ', '').strip()
     for c in comments if c.startswith('comment Escala')),
    'Escala desconocida'
)
pts_3d, tris_3d = guardar_ply(src_3d, pts2d, new_f, new_c, simetrico=True,
                              escala_info=escala_info_3d)
print(f"[3d] regenerado con simetría → {src_3d}")

# 6. Volumen reportado = volumen encerrado de la malla _3d.ply reparada.
vol_barril = volumen_malla_cerrada(pts_3d, tris_3d)
print(f"[volumen] malla cerrada _3d.ply: {vol_barril} L")
print("OK")
