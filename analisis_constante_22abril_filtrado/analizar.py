"""Análisis comparativo: original (output_modelos3d_live) vs filtrado
(output_modelos3d_live_filtrado, frames con covX<55% descartados).

Salidas en analisis_constante_22abril_filtrado/:
  - resultados.md            (tablas comparadas)
  - resultados.csv           (tabla cruda con ambas variantes)
  - resumen.json             (estadísticas para cada caso/escenario)
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

ALT = json.load(open(ROOT / 'alturas_individuos.json'))
H = ALT['alturas_22abril_cm']
W = ALT['pesos_22abril_kg']

ALL_IDS = ['v1','v2','v3','v4','v5','v7','v8','v9','v10','v12','v13','v14','v15']

def load(base, vid):
    return json.load(open(ROOT / base / vid / f'{vid}_resumen.json'))

rows = []
for v in ALL_IDS:
    o = load('output_modelos3d_live', v)
    f = load('output_modelos3d_live_filtrado', v)
    rows.append({
        'id': v,
        'altura_cm': H[v], 'peso_kg': W[v],
        # original
        'orig_vol_A': o['vol_consenso_A_mediana'],
        'orig_vol_B': o['vol_consenso_B_p75'],
        'orig_vol_E': o['vol_consenso_E_envelope'],
        'orig_n_frames': o['frames_usados'],
        # filtrado
        'filt_vol_A': f['vol_consenso_A_mediana'],
        'filt_vol_B': f['vol_consenso_B_p75'],
        'filt_vol_E': f['vol_consenso_E_envelope'],
        'filt_n_frames': f['frames_usados'],
        'filt_n_descartados': f.get('frames_descartados', 0),
    })

# CSV
csv_lines = [
    'id,altura_cm,peso_kg,'
    'orig_vol_A,orig_vol_B,orig_vol_E,orig_n,'
    'filt_vol_A,filt_vol_B,filt_vol_E,filt_n,filt_descartados,'
    'orig_kE,filt_kE,delta_volE'
]
for r in rows:
    csv_lines.append(','.join([
        r['id'], f"{r['altura_cm']}", f"{r['peso_kg']}",
        f"{r['orig_vol_A']:.1f}", f"{r['orig_vol_B']:.1f}", f"{r['orig_vol_E']:.1f}", f"{r['orig_n_frames']}",
        f"{r['filt_vol_A']:.1f}", f"{r['filt_vol_B']:.1f}", f"{r['filt_vol_E']:.1f}", f"{r['filt_n_frames']}",
        f"{r['filt_n_descartados']}",
        f"{r['peso_kg']/r['orig_vol_E']:.3f}",
        f"{r['peso_kg']/r['filt_vol_E']:.3f}",
        f"{r['filt_vol_E']-r['orig_vol_E']:+.1f}",
    ]))
(OUT / 'resultados.csv').write_text('\n'.join(csv_lines) + '\n')


def stats(ids, key_A, key_B, key_E):
    sub = [r for r in rows if r['id'] in ids]
    ws = np.array([r['peso_kg'] for r in sub])
    hs = np.array([r['altura_cm'] for r in sub])
    out = {'n': len(sub), 'ids': list(ids)}
    for label, key in [('A_mediana', key_A), ('B_p75', key_B), ('E_envelope', key_E)]:
        V = np.array([r[key] for r in sub])
        k = ws / V
        rho = float(np.corrcoef(V, ws)[0, 1])
        a_prop = float((V*ws).sum() / (V*V).sum())
        err_prop = ws - a_prop * V
        a_lin, b_lin = np.polyfit(V, ws, 1)
        err_lin = ws - (a_lin*V + b_lin)
        out[label] = {
            'k_media': float(k.mean()), 'k_sd': float(k.std(ddof=1)),
            'k_cv_pct': float(k.std(ddof=1)/k.mean()*100),
            'k_min': float(k.min()), 'k_max': float(k.max()),
            'corr_vol_peso': rho,
            'prop_a': a_prop, 'prop_MAE': float(np.abs(err_prop).mean()),
            'lin_a': float(a_lin), 'lin_b': float(b_lin),
            'lin_MAE': float(np.abs(err_lin).mean()),
        }
    a, b = np.polyfit(hs, ws, 1)
    err = ws - (a*hs + b)
    out['solo_altura'] = {
        'a': float(a), 'b': float(b),
        'MAE': float(np.abs(err).mean()),
        'corr_h_peso': float(np.corrcoef(hs, ws)[0, 1]),
    }
    return out


CASOS = {
    'todos_13': ALL_IDS,
    'sin_v14_v15': [v for v in ALL_IDS if v not in ('v14','v15')],
    'sin_v12_v14_v15': [v for v in ALL_IDS if v not in ('v12','v14','v15')],
}
resumen = {}
for casen, ids in CASOS.items():
    resumen[casen] = {
        'original': stats(ids, 'orig_vol_A', 'orig_vol_B', 'orig_vol_E'),
        'filtrado': stats(ids, 'filt_vol_A', 'filt_vol_B', 'filt_vol_E'),
    }
(OUT / 'resumen.json').write_text(json.dumps(resumen, indent=2))


# Markdown
md = []
md.append('# Comparación original vs filtrado (covX ≥ 55%)\n')
md.append('Generado por `analisis_constante_22abril_filtrado/analizar.py`. Crudo en `resultados.csv`, estadísticas en `resumen.json`.\n')
md.append('## Tabla por individuo\n')
md.append('| ID | h cm | peso kg | n_orig | vol_A_o | vol_B_o | vol_E_o | n_filt | desc | vol_A_f | vol_B_f | vol_E_f | ΔE |')
md.append('|----|------|---------|--------|---------|---------|---------|--------|------|---------|---------|---------|------|')
for r in rows:
    dE = r['filt_vol_E'] - r['orig_vol_E']
    md.append(
        f"| {r['id']} | {r['altura_cm']:.1f} | {r['peso_kg']:.1f} | "
        f"{r['orig_n_frames']} | {r['orig_vol_A']:.1f} | {r['orig_vol_B']:.1f} | {r['orig_vol_E']:.1f} | "
        f"{r['filt_n_frames']} | {r['filt_n_descartados']} | "
        f"{r['filt_vol_A']:.1f} | {r['filt_vol_B']:.1f} | {r['filt_vol_E']:.1f} | "
        f"{dE:+.1f} |"
    )

for casen, ids in CASOS.items():
    s = resumen[casen]
    md.append(f"\n## Caso `{casen}` (n={s['original']['n']})")
    md.append(f"IDs: {', '.join(s['original']['ids'])}\n")
    md.append('### Original')
    md.append('| Variante | k medio | CV % | corr | lineal MAE | prop MAE |')
    md.append('|----------|---------|------|------|------------|----------|')
    for var in ['A_mediana','B_p75','E_envelope']:
        d = s['original'][var]
        md.append(f"| {var} | {d['k_media']:.3f} ± {d['k_sd']:.3f} | {d['k_cv_pct']:.1f} | {d['corr_vol_peso']:+.3f} | {d['lin_MAE']:.1f} kg | {d['prop_MAE']:.1f} kg |")
    md.append(f"_solo altura_: peso = {s['original']['solo_altura']['a']:.3f}·h + {s['original']['solo_altura']['b']:.1f} → MAE {s['original']['solo_altura']['MAE']:.1f} kg, corr {s['original']['solo_altura']['corr_h_peso']:+.3f}\n")

    md.append('### Filtrado')
    md.append('| Variante | k medio | CV % | corr | lineal MAE | prop MAE |')
    md.append('|----------|---------|------|------|------------|----------|')
    for var in ['A_mediana','B_p75','E_envelope']:
        d = s['filtrado'][var]
        md.append(f"| {var} | {d['k_media']:.3f} ± {d['k_sd']:.3f} | {d['k_cv_pct']:.1f} | {d['corr_vol_peso']:+.3f} | {d['lin_MAE']:.1f} kg | {d['prop_MAE']:.1f} kg |")

(OUT / 'resultados.md').write_text('\n'.join(md) + '\n')
print(f'[ok] generado en {OUT}/')
for f in OUT.iterdir():
    if f.is_file():
        print(f'  - {f.name} ({f.stat().st_size} bytes)')
