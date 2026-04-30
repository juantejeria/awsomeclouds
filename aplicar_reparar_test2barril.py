"""Aplica la reparación de hueco a un modelo 3D existente:
- _lateral.ply  → reemplaza con malla 2D reparada (vértices nuevos pintados naranja)
- _3d.ply       → regenera shell simétrico (z+/z-) con depth=0 en el rim
- _volumen.ply  → regenera malla elipsoidal de rebanadas con la silueta nueva

Uso:
    python aplicar_reparar_test2barril.py [NOMBRE_MODELO]

Default: Test2BarrilV1. El modelo debe existir en
    output_modelos3d_live/<nombre>/<nombre>_{lateral,3d,volumen}.ply

Backups quedan en archivo *_orig.ply (solo si no existían ya).
"""
import os
import sys
import shutil
import numpy as np

from reparar_hueco_malla import (
    leer_ply_plano, escribir_ply_plano, reparar,
)
from generar_modelos3d_grandes import guardar_ply
from generar_ply_volumen import rebanadas_desde_contorno, malla_elipsoidal, escribir_ply

COLOR_REPARADO = np.array([255, 140, 0], dtype=np.uint8)  # naranja

NAME = sys.argv[1] if len(sys.argv) > 1 else 'Test2BarrilV1'
MODEL_DIR = f'output_modelos3d_live/{NAME}'
if not os.path.isdir(MODEL_DIR):
    print(f"[error] no existe la carpeta {MODEL_DIR}")
    sys.exit(1)

src_lateral = f'{MODEL_DIR}/{NAME}_lateral.ply'
src_3d      = f'{MODEL_DIR}/{NAME}_3d.ply'
src_vol     = f'{MODEL_DIR}/{NAME}_volumen.ply'

# Backups (solo si no existen)
for src in (src_lateral, src_3d, src_vol):
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
guardar_ply(src_3d, pts2d, new_f, new_c, simetrico=True, escala_info=escala_info_3d)
print(f"[3d] regenerado con simetría → {src_3d}")

# 5. Regenerar _volumen.ply usando contorno reparado como nueva silueta
# Reconstruir contorno externo desde la triangulación reparada.
import cv2
from collections import Counter, defaultdict

cnt = Counter()
for a, b, c in new_f:
    cnt[(min(a, b), max(a, b))] += 1
    cnt[(min(b, c), max(b, c))] += 1
    cnt[(min(a, c), max(a, c))] += 1
boundary = [e for e, k in cnt.items() if k == 1]
adj = defaultdict(list)
for a, b in boundary:
    adj[a].append(b); adj[b].append(a)
loops = []
seen = set()
for v0 in list(adj.keys()):
    if v0 in seen: continue
    loop=[v0]; seen.add(v0); prev=-1; cur=v0
    while True:
        nxt=[w for w in adj[cur] if w!=prev]
        if not nxt: break
        nv = nxt[0]
        if nv in seen: break
        seen.add(nv); loop.append(nv); prev=cur; cur=nv
    loops.append(loop)
def loop_area(lp):
    pts = pts2d[lp]
    x, y = pts[:,0], pts[:,1]; n=len(lp)
    return 0.5 * abs(sum(x[i]*y[(i+1)%n] - x[(i+1)%n]*y[i] for i in range(n)))
outer = max(loops, key=loop_area)
print(f"[volumen] contorno externo: {len(outer)} verts")
contorno_cm = pts2d[outer]

rebanadas = rebanadas_desde_contorno(contorno_cm, n_slices=80)
if len(rebanadas) >= 3:
    verts_v, tris_v = malla_elipsoidal(rebanadas, n_vert=32)
    escribir_ply(src_vol, verts_v, tris_v,
                 comentario=f'{NAME} volumen reparado (hueco rellenado)')
    print(f"[volumen] regenerado → {src_vol}")
else:
    print("[volumen] no se pudo regenerar (rebanadas insuficientes)")
print("OK")
