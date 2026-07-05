"""Diagnóstico de barril con el modelo v8 para 14mayo + 20mayo.
Reusa diagnostico_21frames_barril.py (--barril-model barril_seg_v8.pt).
Salida: checkpoints/<ds>/<ind>/diagnostico_barril_v8_grid.png  (no pisa los previos)
"""
import subprocess, sys
from pathlib import Path

PROJ = Path(__file__).parent
PY = sys.executable
SCRIPT = PROJ / 'diagnostico_21frames_barril.py'
MODEL = 'barril_seg_v8.pt'

ok = fail = 0
for ds in ['14mayo', '20mayo']:
    base = PROJ / 'checkpoints' / ds
    if not base.is_dir():
        print(f"[skip] {base}"); continue
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        out = folder / 'diagnostico_barril_v8_grid.png'
        r = subprocess.run(
            [PY, str(SCRIPT), str(folder), '--barril-model', MODEL, '--out', str(out)],
            capture_output=True, text=True)
        if r.returncode == 0 and out.is_file():
            print(f"[ok] {ds}/{folder.name}", flush=True); ok += 1
        else:
            print(f"[FAIL] {ds}/{folder.name}: {r.stderr.strip().splitlines()[-2:]}", flush=True); fail += 1

print(f"\n[DONE] ok={ok} fail={fail}")
