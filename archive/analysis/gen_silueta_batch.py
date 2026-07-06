"""Genera modelos 3D de los 28 individuos (14mayo + 20mayo) usando la SILUETA
completa (silueta_seg.pt) en vez del barril, sobre los 21 frames.
Reusa procesar_21_frames_filtrado.py por individuo (--barril-model silueta_seg.pt).
Altura real tomada del _resumen.json del modelo de barril existente.
Salida: output_modelos3d_live_<ds>_silueta/<ind>/
"""
import json, subprocess, sys
from pathlib import Path

PROJ = Path(__file__).parent
PY = sys.executable
SCRIPT = PROJ / 'procesar_21_frames_filtrado.py'
DS = [('14mayo', PROJ / 'checkpoints' / '14mayo', PROJ / 'output_modelos3d_live_14mayo'),
      ('20mayo', PROJ / 'checkpoints' / '20mayo', PROJ / 'output_modelos3d_live_20mayo')]


def altura_de(ind, resbase):
    rj = resbase / ind / f'{ind}_resumen.json'
    if rj.is_file():
        try:
            a = json.loads(rj.read_text()).get('altura_real_cm')
            if a:
                return float(a)
        except Exception:
            pass
    try:
        return float(ind.split('_')[0])
    except Exception:
        return None


ok = fail = 0
for ds, frames_base, resbase in DS:
    if not frames_base.is_dir():
        print(f"[skip] {frames_base}"); continue
    for folder in sorted(frames_base.iterdir()):
        if not folder.is_dir():
            continue
        ind = folder.name
        alt = altura_de(ind, resbase)
        if alt is None:
            print(f"[warn] {ds}/{ind}: sin altura, salto"); fail += 1; continue
        print(f"\n===== {ds}/{ind}  altura={alt}cm =====", flush=True)
        r = subprocess.run(
            [PY, str(SCRIPT), str(folder), str(alt), ind,
             '--barril-model', 'silueta_seg.pt', '--out-tag', f'{ds}_silueta'],
            capture_output=True, text=True)
        tail = '\n'.join(l for l in r.stdout.splitlines() if 'volumen' in l.lower()
                         or 'Done' in l or '[ply] 3d' in l)
        if r.returncode == 0:
            print(tail or '[ok]'); ok += 1
        else:
            print(f"[ERROR rc={r.returncode}] {r.stderr.strip().splitlines()[-3:]}"); fail += 1

print(f"\n[BATCH DONE] ok={ok} fail={fail}")
