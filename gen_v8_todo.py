"""Genera modelos 3D v8 + diagnóstico v8 para 6mayo, 14mayo, 20mayo y 12 junio.
- Modelo 3D: procesar_21_frames_filtrado.py con barril_seg_v8.pt, salida en
  output_modelos3d_live_<tag>/<individuo>/
- Diagnóstico: diagnostico_21frames_barril.py con v8, en la carpeta de frames
  como diagnostico_barril_v8_grid.png (skip si ya existe).
Alturas calc: 14/20mayo y 6mayo desde sus resúmenes; 12 junio desde dict.
"""
import json, subprocess, sys
from pathlib import Path

PROJ = Path(__file__).parent
PY = sys.executable
GEN = PROJ / 'procesar_21_frames_filtrado.py'
DIAG = PROJ / 'diagnostico_21frames_barril.py'
MODEL = 'barril_seg_v8.pt'

JUNIO = {'000_306':102.8,'000_391':111.7,'000_392':109.3,'000_392A':120.7,
 '000_392B':111.1,'000_392C':110.9,'000_395':118.5,'000_435':122.7,
 '000_435A':121.3,'000_436':122.1,'000_446':109.4,'000_447':121.2,
 '000_448':112.9,'000_448A':114.0,'000_459':122.7,'000_499':116.6,
 '000_524':109.8,'000_560':124.1}

# (dataset_label, frames_base, calc_dir|None, out_tag, heights_dict|None)
DATASETS = [
    ('14mayo', PROJ/'checkpoints'/'14mayo', PROJ/'output_modelos3d_live_14mayo', '14mayo_v8', None),
    ('20mayo', PROJ/'checkpoints'/'20mayo', PROJ/'output_modelos3d_live_20mayo', '20mayo_v8', None),
    ('6mayo',  PROJ/'checkpoints'/'6mayo',  PROJ/'output_modelos3d_live_6mayo_v7', '6mayo_v8', None),
    ('12junio', PROJ/'checkpoints'/'12 junio', None, '12junio_v8', JUNIO),
]


def altura_de(ind, calc_dir, hdict):
    if hdict is not None:
        return hdict.get(ind)
    rj = calc_dir / ind / f'{ind}_resumen.json'
    if rj.is_file():
        try:
            return float(json.loads(rj.read_text()).get('altura_real_cm') or 0) or None
        except Exception:
            return None
    return None


okM = failM = okD = failD = skipD = 0
for label, fbase, calc_dir, tag, hdict in DATASETS:
    if not fbase.is_dir():
        print(f"[skip dataset] {fbase}"); continue
    print(f"\n########## {label}  ({fbase}) ##########", flush=True)
    for folder in sorted(fbase.iterdir()):
        if not folder.is_dir():
            continue
        ind = folder.name
        if not list(folder.glob('frame_*.jpg')):
            continue
        alt = altura_de(ind, calc_dir, hdict)
        if not alt:
            print(f"[NO-ALT] {label}/{ind} -> sin altura, salto"); failM += 1; continue
        # 1) modelo 3D v8
        r = subprocess.run([PY, str(GEN), str(folder), str(alt), ind,
                            '--out-tag', tag, '--barril-model', MODEL],
                           capture_output=True, text=True)
        vol = ''
        for ln in r.stdout.splitlines():
            if 'volumen' in ln.lower():
                vol = ln.strip()
        if r.returncode == 0:
            print(f"[MODEL ok] {label}/{ind} alt={alt}  {vol}", flush=True); okM += 1
        else:
            print(f"[MODEL FAIL] {label}/{ind}: {r.stderr.strip().splitlines()[-2:]}", flush=True); failM += 1
        # 2) diagnóstico v8 (skip si ya existe)
        out = folder / 'diagnostico_barril_v8_grid.png'
        if out.is_file():
            skipD += 1
        else:
            rd = subprocess.run([PY, str(DIAG), str(folder), '--out', str(out),
                                 '--barril-model', MODEL], capture_output=True, text=True)
            if rd.returncode == 0 and out.is_file():
                okD += 1
            else:
                print(f"[DIAG FAIL] {label}/{ind}: {rd.stderr.strip().splitlines()[-1:]}", flush=True); failD += 1

print(f"\n[DONE] modelos ok={okM} fail={failM} | diag ok={okD} skip(existían)={skipD} fail={failD}")
