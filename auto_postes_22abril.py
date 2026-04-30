"""Auto-detecta los 2 postes (franjas rojas verticales de 50cm) en el frame
central de cada carpeta checkpoints/22abril/v*/ y guarda el resultado en su
context.json (campo locked_reference).

Genera un grid PNG en grids_postes/ con la detección marcada para validación
visual antes de re-procesar volúmenes.

Uso:
    python auto_postes_22abril.py [--solo v1,v2] [--apply]
    --apply: escribe locked_reference en context.json (sin esto, solo genera el grid)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).parent
ALL = ['v1','v2','v3','v4','v5','v7','v8','v9','v10','v12','v13','v14','v15']


def detectar_rojo(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 50, 30]), np.array([15, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 50, 30]), np.array([180, 255, 255]))
    mask = m1 | m2
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def franjas(mask, min_h=20, min_aspect=1.5, min_fill=0.4):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if h < min_h or w == 0:
            continue
        aspect = h / w
        if aspect < min_aspect:
            continue
        roi = mask[y:y+h, x:x+w]
        fill = roi.sum() / 255.0 / (w*h)
        if fill < min_fill:
            continue
        candidatos.append({'x': x, 'y': y, 'w': w, 'h': h, 'aspect': aspect, 'fill': fill, 'cx': x + w/2})
    return candidatos


def elegir_dos_postes(cands, video_w):
    """De los candidatos, elegir 2 'mejores' postes:
    - Verticales y altos
    - Cerca del centro horizontal (a izquierda y derecha del centro o a la
      misma altura vertical, lo que sugiere postes paralelos)
    Estrategia: agarrar las 2 más altas, descartando duplicados muy cercanos.
    """
    if len(cands) < 2:
        return None
    cands_sorted = sorted(cands, key=lambda c: -c['h'])
    elegidos = [cands_sorted[0]]
    for c in cands_sorted[1:]:
        # No elegir uno que esté pegado al primero (mismo poste detectado dos veces)
        too_close = any(abs(c['cx'] - e['cx']) < 30 for e in elegidos)
        if too_close:
            continue
        elegidos.append(c)
        if len(elegidos) == 2:
            break
    if len(elegidos) < 2:
        return None
    elegidos.sort(key=lambda c: c['cx'])  # post1 = izquierda, post2 = derecha
    return elegidos


def post_to_lockformat(p):
    """Convierte un candidato en formato {cx, top_tape, floor, tape_px}."""
    return {
        'cx': float(p['cx']),
        'top_tape': float(p['y']),
        'floor': float(p['y'] + p['h']),
        'tape_px': float(p['h']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solo', default='', help='IDs separados por coma (ej v1,v2)')
    ap.add_argument('--apply', action='store_true',
                    help='Escribir locked_reference en context.json')
    args = ap.parse_args()

    ids = [v for v in ALL if v in (set(args.solo.split(',')) if args.solo else ALL)]

    out_grid_dir = PROJ / 'grids_postes'
    out_grid_dir.mkdir(exist_ok=True)

    tiles = []
    resultados = {}

    for v in ids:
        d = PROJ / 'checkpoints' / '22abril' / v
        if not d.is_dir():
            print(f"[{v}] sin carpeta"); continue
        # Frame central: el que tiene 'frame_000_*' o el primero alfabéticamente
        cents = sorted(d.glob('frame_000_*.jpg'))
        fp = cents[0] if cents else (sorted(d.glob('frame_*.jpg'))[0] if any(d.glob('frame_*.jpg')) else None)
        if fp is None:
            print(f"[{v}] sin frames"); continue
        img = cv2.imread(str(fp))
        H, W = img.shape[:2]
        mask = detectar_rojo(img)
        cands = franjas(mask)
        elegidos = elegir_dos_postes(cands, W)

        # Construir locked_reference (las coords son las del frame original)
        if elegidos:
            p1, p2 = elegidos
            lock = {
                'post1': post_to_lockformat(p1),
                'post2': post_to_lockformat(p2),
                'original_coords': {
                    'post1': post_to_lockformat(p1),
                    'post2': post_to_lockformat(p2),
                    'video_w': W, 'video_h': H,
                },
            }
            resultados[v] = lock
            tape_avg = (p1['h'] + p2['h']) / 2
            cm_per_px_post = 50.0 / tape_avg  # tape = 50 cm
            print(f"[{v}] OK  postes en cx={p1['cx']:.0f},{p2['cx']:.0f}  tape_avg={tape_avg:.0f}px  cm/px={cm_per_px_post:.4f}  ({len(cands)} candidatos)")
        else:
            print(f"[{v}] FALLA: solo {len(cands)} candidatos válidos")

        # Visualización: dibujar todos los candidatos + marcar los elegidos
        vis = img.copy()
        for c in cands:
            cv2.rectangle(vis, (c['x'], c['y']), (c['x']+c['w'], c['y']+c['h']),
                          (0, 0, 255), 1)
        if elegidos:
            for idx, p in enumerate(elegidos):
                color = (0, 255, 0) if idx == 0 else (255, 255, 0)
                cv2.rectangle(vis, (p['x'], p['y']), (p['x']+p['w'], p['y']+p['h']), color, 3)
                label = f"P{idx+1} h={int(p['h'])}px"
                cv2.putText(vis, label, (p['x'], max(20, p['y']-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Tile 480 wide
        tw = 480
        scale = tw / W
        th = int(H * scale)
        tile = cv2.resize(vis, (tw, th))
        bar = np.zeros((30, tw, 3), dtype=np.uint8)
        cm_str = f"cm/px={50.0/((p1['h']+p2['h'])/2):.4f}" if elegidos else "sin postes"
        txt = f"{v} | {cm_str} | {len(cands)} cand."
        cv2.putText(bar, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile]))

    # Grid
    if tiles:
        cols = 4
        rows = (len(tiles)+cols-1)//cols
        th, tw = tiles[0].shape[:2]
        rows_imgs = []
        for r in range(rows):
            row = tiles[r*cols:(r+1)*cols]
            if len(row) < cols:
                blank = np.zeros((th, tw, 3), dtype=np.uint8)
                row = row + [blank]*(cols-len(row))
            rows_imgs.append(np.hstack([cv2.resize(t,(tw,th)) for t in row]))
        grid = np.vstack(rows_imgs)
        out = out_grid_dir / 'auto_postes_grid.png'
        cv2.imwrite(str(out), grid)
        print(f"\n[ok] grid → {out}")

    if args.apply:
        for v, lock in resultados.items():
            ctx_path = PROJ / 'checkpoints' / '22abril' / v / 'context.json'
            if ctx_path.exists():
                ctx = json.load(open(ctx_path))
            else:
                ctx = {}
            ctx['locked_reference'] = lock
            ctx['locked_reference_source'] = 'auto_detect_franjas'
            json.dump(ctx, open(ctx_path, 'w'), indent=2)
        print(f"[apply] {len(resultados)} context.json actualizados")
    else:
        print(f"\n[dry-run] {len(resultados)} detecciones listas. Re-correr con --apply para guardar.")


if __name__ == '__main__':
    main()
