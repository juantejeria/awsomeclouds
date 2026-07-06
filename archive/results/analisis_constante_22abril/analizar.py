"""Análisis: ¿se puede generar una constante k = peso/vol_barril a partir
de los 13 individuos del dataset 22abril?

Salidas en analisis_constante_22abril/:
  - resultados.md        (tablas formateadas)
  - resultados.csv       (tabla cruda con todas las variantes)
  - resumen.json         (constantes y estadísticas finales)
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

rows = []
for v in ALL_IDS:
    r = json.load(open(ROOT / 'output_modelos3d_live' / v / f'{v}_resumen.json'))
    rows.append({
        'id': v,
        'altura_cm': H[v],
        'peso_kg': W[v],
        'vol_A_mediana': r['vol_consenso_A_mediana'],
        'vol_B_p75': r['vol_consenso_B_p75'],
        'vol_E_envelope': r['vol_consenso_E_envelope'],
        'width_cm': r['width_consenso_cm'],
        'alto_max_cm': r['alto_max_consenso_cm'],
    })

# CSV crudo
csv_lines = ['id,altura_cm,peso_kg,vol_A_mediana_L,vol_B_p75_L,vol_E_envelope_L,width_cm,alto_max_cm,k_A,k_B,k_E']
for r in rows:
    csv_lines.append(','.join([
        r['id'], f"{r['altura_cm']}", f"{r['peso_kg']}",
        f"{r['vol_A_mediana']:.2f}", f"{r['vol_B_p75']:.2f}", f"{r['vol_E_envelope']:.2f}",
        f"{r['width_cm']:.2f}", f"{r['alto_max_cm']:.2f}",
        f"{r['peso_kg']/r['vol_A_mediana']:.3f}",
        f"{r['peso_kg']/r['vol_B_p75']:.3f}",
        f"{r['peso_kg']/r['vol_E_envelope']:.3f}",
    ]))
(OUT / 'resultados.csv').write_text('\n'.join(csv_lines) + '\n')


def stats(ids, label):
    sub = [r for r in rows if r['id'] in ids]
    ws = np.array([r['peso_kg'] for r in sub])
    hs = np.array([r['altura_cm'] for r in sub])
    out = {'n': len(sub), 'ids': ids}
    for key, vk in [('A_mediana','vol_A_mediana'), ('B_p75','vol_B_p75'), ('E_envelope','vol_E_envelope')]:
        V = np.array([r[vk] for r in sub])
        k = ws / V
        rho = float(np.corrcoef(V, ws)[0, 1])
        a_prop = float((V*ws).sum() / (V*V).sum())
        err_prop = ws - a_prop * V
        a_lin, b_lin = np.polyfit(V, ws, 1)
        pred_lin = a_lin*V + b_lin
        err_lin = ws - pred_lin
        out[key] = {
            'k_media': float(k.mean()),
            'k_sd': float(k.std(ddof=1)),
            'k_cv_pct': float(k.std(ddof=1)/k.mean()*100),
            'k_min': float(k.min()),
            'k_max': float(k.max()),
            'corr_vol_peso': rho,
            'prop_a': a_prop,
            'prop_MAE': float(np.abs(err_prop).mean()),
            'prop_max': float(np.abs(err_prop).max()),
            'lin_a': float(a_lin),
            'lin_b': float(b_lin),
            'lin_MAE': float(np.abs(err_lin).mean()),
            'lin_max': float(np.abs(err_lin).max()),
        }
    # solo altura
    a, b = np.polyfit(hs, ws, 1)
    pred = a*hs + b; err = ws - pred
    out['solo_altura'] = {
        'a': float(a), 'b': float(b), 'MAE': float(np.abs(err).mean()),
        'max': float(np.abs(err).max()),
        'corr_h_peso': float(np.corrcoef(hs, ws)[0, 1]),
    }
    # multivar (vol_E + altura)
    V = np.array([r['vol_E_envelope'] for r in sub])
    X = np.column_stack([V, hs, np.ones_like(V)])
    coef, *_ = np.linalg.lstsq(X, ws, rcond=None)
    err = ws - X @ coef
    out['multivar_E_h'] = {
        'a_vol': float(coef[0]), 'a_h': float(coef[1]), 'b': float(coef[2]),
        'MAE': float(np.abs(err).mean()), 'max': float(np.abs(err).max()),
    }
    return out


CASOS = {
    'todos_13': ALL_IDS,
    'sin_v14_v15': [v for v in ALL_IDS if v not in ('v14','v15')],
    'sin_v12_v14_v15': [v for v in ALL_IDS if v not in ('v12','v14','v15')],
}
resumen = {label: stats(ids, label) for label, ids in CASOS.items()}
(OUT / 'resumen.json').write_text(json.dumps(resumen, indent=2))


# Markdown
md = []
md.append('# Análisis constante peso ↔ volumen barril (dataset 22abril)\n')
md.append('Generado por `analisis_constante_22abril/analizar.py`. Datos crudos en `resultados.csv`, estadísticas en `resumen.json`.\n')
md.append('## Datos por individuo\n')
md.append('| ID | Altura cm | Peso kg | vol A (med) | vol B (p75) | vol E (env) | k_A | k_B | k_E |')
md.append('|----|-----------|---------|-------------|-------------|-------------|-----|-----|-----|')
for r in rows:
    md.append(f"| {r['id']} | {r['altura_cm']:.1f} | {r['peso_kg']:.1f} | {r['vol_A_mediana']:.1f} | {r['vol_B_p75']:.1f} | {r['vol_E_envelope']:.1f} | {r['peso_kg']/r['vol_A_mediana']:.2f} | {r['peso_kg']/r['vol_B_p75']:.2f} | {r['peso_kg']/r['vol_E_envelope']:.2f} |")

for label in CASOS:
    s = resumen[label]
    md.append(f'\n## Caso: `{label}` (n={s["n"]})\n')
    md.append(f'IDs incluidos: {", ".join(s["ids"])}\n')
    md.append('| Variante | k medio kg/L | CV % | corr(vol,peso) | proporcional a*vol | MAE prop | lineal a*vol+b | MAE lin |')
    md.append('|----------|--------------|------|----------------|--------------------|----------|----------------|---------|')
    for key in ['A_mediana','B_p75','E_envelope']:
        d = s[key]
        md.append(f"| {key} | {d['k_media']:.3f} ± {d['k_sd']:.3f} | {d['k_cv_pct']:.1f} | {d['corr_vol_peso']:+.3f} | {d['prop_a']:.4f}·vol | {d['prop_MAE']:.1f} kg | {d['lin_a']:.4f}·vol + {d['lin_b']:.1f} | {d['lin_MAE']:.1f} kg |")
    sa = s['solo_altura']
    md.append(f"\n**Sólo altura:** peso = {sa['a']:.3f}·h + {sa['b']:.1f} → MAE {sa['MAE']:.1f} kg, max {sa['max']:.1f} kg, corr(h,peso)={sa['corr_h_peso']:+.3f}")
    sm = s['multivar_E_h']
    md.append(f"\n**Multivar (vol_E + altura):** peso = {sm['a_vol']:+.4f}·vol + {sm['a_h']:+.4f}·h + {sm['b']:+.1f} → MAE {sm['MAE']:.1f} kg")

md.append('\n## Conclusión\n')
md.append("- Con los 13 individuos la correlación vol↔peso es **negativa** en las 3 variantes → el volumen del barril (esta pipeline) no es predictor confiable de peso.")
md.append("- Quitando v14, v15 y v12 la correlación se vuelve **positiva débil** para A y B (+0.16, +0.21), pero CV sigue ~20%.")
md.append("- La menos mala como constante única: **k ≈ 2.30 kg/L** sobre `vol_B_p75` (sin v12,v14,v15). Implica error ~±20% en peso.")
md.append("- La altura sola predice mejor que cualquier volumen (corr +0.58, MAE ~14 kg).")
md.append("- Antes de fijar constante, conviene revisar máscaras infladas (v14,v15) y/o filtrar frames de baja cobertura.")

(OUT / 'resultados.md').write_text('\n'.join(md) + '\n')
print(f'[ok] generado en {OUT}/')
for f in OUT.iterdir():
    if f.is_file():
        print(f'  - {f.name} ({f.stat().st_size} bytes)')
