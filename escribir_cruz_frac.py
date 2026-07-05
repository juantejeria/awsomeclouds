"""
Escribe `cruz_frac` en el _resumen.json de cada modelo 3D (dirs v8 que sirve la UI)
a partir de output_cruz_modelos/cruz_resultados.csv.

`cruz_frac` es la fracción del largo del barril DESDE EL FRENTE hasta la cruz
(misma convencion que girth_frac, para reutilizar la logica del visor):
    cruz_frac = (1 - cruz_xn)   si barril_dir != 'left'  (frente en xMax)
    cruz_frac =      cruz_xn     si barril_dir == 'left'  (frente en xMin)
donde cruz_xn es la x de la cruz normalizada dentro del bbox del barril (0=izq, 1=der).

Uso:  python escribir_cruz_frac.py
"""
import csv
import json
import os
from pathlib import Path

PROJ = Path(__file__).parent
CSV = PROJ / 'output_cruz_modelos' / 'cruz_resultados.csv'
V8 = {
    '6mayo':  'output_modelos3d_live_6mayo_v8',
    '14mayo': 'output_modelos3d_live_14mayo_v8',
    '20mayo': 'output_modelos3d_live_20mayo_v8',
    '12junio': 'output_modelos3d_live_12junio_v8',
}


def resumen_path(dir_path):
    for f in sorted(os.listdir(dir_path)):
        if f.endswith('_resumen.json') or f == 'resumen.json':
            return os.path.join(dir_path, f)
    return None


def main():
    rows = list(csv.DictReader(open(CSV)))
    ok = skip = 0
    for r in rows:
        if r.get('cruz_xn') in (None, ''):
            print(f"[skip] {r['dataset']}/{r['individuo']}: sin cruz_xn"); skip += 1; continue
        dir_path = PROJ / V8[r['dataset']] / r['individuo']
        rp = resumen_path(dir_path) if dir_path.is_dir() else None
        if not rp:
            print(f"[skip] {r['dataset']}/{r['individuo']}: sin resumen"); skip += 1; continue
        meta = json.load(open(rp))
        cruz_xn = float(r['cruz_xn'])
        facing_right = (meta.get('barril_dir') != 'left')
        cruz_frac = (1.0 - cruz_xn) if facing_right else cruz_xn
        cruz_frac = max(0.0, min(0.5, cruz_frac))  # rango del visor (desde el frente)

        meta['cruz_frac'] = round(cruz_frac, 4)
        meta['cruz_xn'] = round(cruz_xn, 4)
        if r.get('cruz_yn') not in (None, ''):
            meta['cruz_yn'] = round(float(r['cruz_yn']), 4)
        meta['cruz_conf'] = float(r.get('conf_med') or 0.0)

        tmp = rp + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp, rp)
        ok += 1
        print(f"  {r['dataset']}/{r['individuo']}: barril_dir={meta.get('barril_dir')} "
              f"cruz_xn={cruz_xn:.3f} -> cruz_frac={cruz_frac:.3f}")

    print(f"\n[done] escritos {ok}, saltados {skip}")


if __name__ == '__main__':
    main()
