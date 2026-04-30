"""
Genera un PLY de VOLUMEN (rebanadas elipticas apiladas) a partir del
contorno lateral 2D ya calculado en *_lateral.ply.

Cada rebanada es una elipse en el plano (Y, Z):
    semi-eje vertical   = h/2
    semi-eje profundidad = h * K_DEPTH
donde h = altura del animal en esa columna X (en cm).

Esto reconstruye visualmente el MISMO volumen que integra
volumen_por_rebanadas() en generar_modelos3d_grandes.py.

Uso:
    python3 generar_ply_volumen.py              # procesa toda la carpeta servida por la UI
    python3 generar_ply_volumen.py --dir DIR    # una carpeta especifica de vaca
"""
import sys
import argparse
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).parent

# Mismo K_DEPTH que volumen_por_rebanadas (generar_modelos3d_grandes.py:820)
K_DEPTH = 0.25

# Resolucion del mallado
N_SLICES = 80      # numero de rebanadas a lo largo del eje X
N_VERTICES = 40    # vertices por elipse
COLOR = (230, 140, 40)  # naranja calido


def load_lateral_ply(path: Path):
    """Lee un PLY ASCII generado por guardar_ply (simetrico=False)
    y retorna los puntos XY del contorno (Z=0).
    """
    lines = path.read_text().splitlines()
    n_vertex = 0
    header_end = 0
    for i, ln in enumerate(lines):
        if ln.startswith('element vertex'):
            n_vertex = int(ln.split()[-1])
        if ln.strip() == 'end_header':
            header_end = i + 1
            break

    pts = []
    for ln in lines[header_end:header_end + n_vertex]:
        parts = ln.split()
        if len(parts) >= 3:
            pts.append((float(parts[0]), float(parts[1])))
    return np.array(pts, dtype=float)


def rebanadas_desde_contorno(pts_xy, n_slices=N_SLICES):
    """Para cada una de n_slices posiciones X equiespaciadas, calcula el
    rango [y_min, y_max] del contorno -> altura h y centro y_c.

    Retorna lista de (x, y_c, h) validos (h > 0).
    """
    x_min, x_max = pts_xy[:, 0].min(), pts_xy[:, 0].max()
    xs = np.linspace(x_min, x_max, n_slices + 2)[1:-1]  # evitar extremos

    dx = (x_max - x_min) / n_slices
    rebanadas = []
    for x in xs:
        # Ventana alrededor de x
        mask = (pts_xy[:, 0] >= x - dx) & (pts_xy[:, 0] <= x + dx)
        ys = pts_xy[mask, 1]
        if len(ys) < 2:
            continue
        y_min, y_max = ys.min(), ys.max()
        h = y_max - y_min
        if h < 1.0:
            continue
        y_c = (y_min + y_max) / 2.0
        rebanadas.append((x, y_c, h))
    return rebanadas


def malla_elipsoidal(rebanadas, n_vert=N_VERTICES):
    """Construye vertices y triangulos de un tubo elipsoidal cuyas
    secciones transversales son las rebanadas [(x, y_c, h), ...].

    Elipse en (Z, Y) con semi-ejes b=h*K_DEPTH (profundidad Z) y a=h/2 (vertical Y).
    """
    if len(rebanadas) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    theta = np.linspace(0, 2 * np.pi, n_vert, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    vertices = []
    for (x, y_c, h) in rebanadas:
        a = h / 2.0         # semi-eje vertical
        b = h * K_DEPTH     # semi-eje profundidad
        for i in range(n_vert):
            vy = y_c + a * sin_t[i]
            vz = b * cos_t[i]
            vertices.append((x, vy, vz))

    vertices = np.array(vertices, dtype=float)

    # Triangulos que conectan rebanadas consecutivas
    tris = []
    n_ring = n_vert
    n_rings = len(rebanadas)
    for r in range(n_rings - 1):
        base_a = r * n_ring
        base_b = (r + 1) * n_ring
        for i in range(n_ring):
            i_next = (i + 1) % n_ring
            v0 = base_a + i
            v1 = base_a + i_next
            v2 = base_b + i
            v3 = base_b + i_next
            tris.append((v0, v2, v1))
            tris.append((v1, v2, v3))

    # Tapas en los extremos (triangle fan hacia el centro de cada rebanada)
    # Extremo inicial
    x0, y0_c, _ = rebanadas[0]
    idx_center_start = len(vertices)
    vertices = np.vstack([vertices, [[x0, y0_c, 0.0]]])
    for i in range(n_ring):
        i_next = (i + 1) % n_ring
        tris.append((idx_center_start, i_next, i))

    # Extremo final
    xf, yf_c, _ = rebanadas[-1]
    idx_center_end = len(vertices)
    vertices = np.vstack([vertices, [[xf, yf_c, 0.0]]])
    base_last = (n_rings - 1) * n_ring
    for i in range(n_ring):
        i_next = (i + 1) % n_ring
        tris.append((idx_center_end, base_last + i, base_last + i_next))

    return vertices, np.array(tris, dtype=int)


def escribir_ply(path: Path, vertices, tris, color=COLOR, comentario=""):
    r, g, b = color
    nv, nf = len(vertices), len(tris)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("comment Unidades: centimetros\n")
        f.write("comment Malla de volumen por rebanadas elipticas (K_DEPTH=%.2f)\n" % K_DEPTH)
        if comentario:
            f.write(f"comment {comentario}\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in vertices:
            f.write(f"{v[0]:.2f} {v[1]:.2f} {v[2]:.2f} {r} {g} {b}\n")
        for t in tris:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def procesar_vaca(vaca_dir: Path):
    """Genera _volumen.ply en una carpeta de vaca."""
    lateral_candidates = list(vaca_dir.glob('*_lateral.ply'))
    if not lateral_candidates:
        return False, "no hay _lateral.ply"
    lateral = lateral_candidates[0]

    pts = load_lateral_ply(lateral)
    if len(pts) < 10:
        return False, f"contorno con solo {len(pts)} puntos"

    rebanadas = rebanadas_desde_contorno(pts)
    if len(rebanadas) < 3:
        return False, f"solo {len(rebanadas)} rebanadas validas"

    vertices, tris = malla_elipsoidal(rebanadas)

    # Volumen estimado (sanity check, debe coincidir aprox con vol_total)
    vol_cm3 = 0.0
    for i in range(len(rebanadas) - 1):
        x0, _, h0 = rebanadas[i]
        x1, _, h1 = rebanadas[i + 1]
        dx = x1 - x0
        a0 = h0 / 2.0
        b0 = h0 * K_DEPTH
        a1 = h1 / 2.0
        b1 = h1 * K_DEPTH
        area_avg = (np.pi * a0 * b0 + np.pi * a1 * b1) / 2.0
        vol_cm3 += area_avg * dx
    vol_l = vol_cm3 / 1000.0

    out = lateral.with_name(lateral.name.replace('_lateral.ply', '_volumen.ply'))
    escribir_ply(out, vertices, tris,
                 comentario=f"Volumen aprox: {vol_l:.1f} L en {len(rebanadas)} rebanadas")
    return True, f"{out.name} ({len(rebanadas)} rebanadas, {vol_l:.1f} L)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None,
                        help='Carpeta de una vaca especifica')
    parser.add_argument('--root', type=str,
                        default='output_modelos3d_Recorte26marz_altdiag',
                        help='Carpeta raiz de modelos')
    args = parser.parse_args()

    if args.dir:
        d = Path(args.dir)
        ok, msg = procesar_vaca(d)
        print(f"{d.name}: {'OK' if ok else 'FAIL'} - {msg}")
        return

    root = PROJECT / args.root
    if not root.is_dir():
        print(f"ERROR: {root} no existe")
        sys.exit(1)

    count_ok = count_fail = 0
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith('_'):
            continue
        ok, msg = procesar_vaca(entry)
        status = 'OK' if ok else 'FAIL'
        print(f"  {entry.name}: {status} - {msg}")
        if ok:
            count_ok += 1
        else:
            count_fail += 1

    print(f"\nResultado: {count_ok} OK, {count_fail} FAIL")


if __name__ == '__main__':
    main()
