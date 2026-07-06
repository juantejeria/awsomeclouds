# Análisis constante peso ↔ volumen barril (dataset 22abril)

Generado por `analisis_constante_22abril/analizar.py`. Datos crudos en `resultados.csv`, estadísticas en `resumen.json`.

## Datos por individuo

| ID | Altura cm | Peso kg | vol A (med) | vol B (p75) | vol E (env) | k_A | k_B | k_E |
|----|-----------|---------|-------------|-------------|-------------|-----|-----|-----|
| v1 | 92.5 | 165.0 | 69.3 | 79.8 | 124.5 | 2.38 | 2.07 | 1.33 |
| v2 | 100.1 | 230.0 | 89.8 | 107.1 | 152.9 | 2.56 | 2.15 | 1.50 |
| v3 | 92.3 | 180.5 | 56.3 | 67.0 | 106.8 | 3.21 | 2.69 | 1.69 |
| v4 | 100.1 | 202.0 | 75.1 | 91.3 | 167.0 | 2.69 | 2.21 | 1.21 |
| v5 | 92.3 | 168.5 | 63.4 | 72.2 | 181.1 | 2.66 | 2.33 | 0.93 |
| v7 | 96.2 | 169.0 | 85.9 | 96.4 | 153.0 | 1.97 | 1.75 | 1.10 |
| v8 | 92.3 | 180.5 | 64.4 | 78.0 | 119.5 | 2.80 | 2.31 | 1.51 |
| v9 | 94.1 | 188.5 | 62.3 | 73.8 | 123.9 | 3.03 | 2.55 | 1.52 |
| v10 | 95.1 | 210.0 | 53.2 | 65.0 | 113.8 | 3.95 | 3.23 | 1.85 |
| v12 | 100.0 | 183.5 | 90.3 | 115.9 | 200.2 | 2.03 | 1.58 | 0.92 |
| v13 | 97.9 | 167.0 | 76.7 | 97.9 | 179.5 | 2.18 | 1.71 | 0.93 |
| v14 | 98.6 | 166.0 | 91.1 | 117.9 | 261.7 | 1.82 | 1.41 | 0.63 |
| v15 | 96.5 | 160.0 | 95.1 | 132.4 | 238.9 | 1.68 | 1.21 | 0.67 |

## Caso: `todos_13` (n=13)

IDs incluidos: v1, v2, v3, v4, v5, v7, v8, v9, v10, v12, v13, v14, v15

| Variante | k medio kg/L | CV % | corr(vol,peso) | proporcional a*vol | MAE prop | lineal a*vol+b | MAE lin |
|----------|--------------|------|----------------|--------------------|----------|----------------|---------|
| A_mediana | 2.535 ± 0.630 | 24.8 | -0.147 | 2.3486·vol | 35.2 kg | -0.2103·vol + 198.1 | 15.6 kg |
| B_p75 | 2.093 ± 0.561 | 26.8 | -0.195 | 1.8792·vol | 41.6 kg | -0.1869·vol + 199.5 | 15.5 kg |
| E_envelope | 1.215 ± 0.387 | 31.9 | -0.398 | 1.0199·vol | 52.3 kg | -0.1699·vol + 210.1 | 14.3 kg |

**Sólo altura:** peso = 2.407·h + -48.7 → MAE 15.7 kg, max 37.8 kg, corr(h,peso)=+0.365

**Multivar (vol_E + altura):** peso = -0.3663·vol + +5.5205·h + -287.8 → MAE 9.5 kg

## Caso: `sin_v14_v15` (n=11)

IDs incluidos: v1, v2, v3, v4, v5, v7, v8, v9, v10, v12, v13

| Variante | k medio kg/L | CV % | corr(vol,peso) | proporcional a*vol | MAE prop | lineal a*vol+b | MAE lin |
|----------|--------------|------|----------------|--------------------|----------|----------------|---------|
| A_mediana | 2.677 ± 0.575 | 21.5 | +0.119 | 2.5278·vol | 28.7 kg | 0.1878·vol + 172.4 | 16.0 kg |
| B_p75 | 2.236 ± 0.479 | 21.4 | +0.150 | 2.0973·vol | 30.1 kg | 0.1821·vol + 170.2 | 16.0 kg |
| E_envelope | 1.317 ± 0.324 | 24.6 | -0.135 | 1.2061·vol | 39.4 kg | -0.0876·vol + 198.8 | 15.7 kg |

**Sólo altura:** peso = 3.166·h + -117.2 → MAE 14.1 kg, max 30.3 kg, corr(h,peso)=+0.510

**Multivar (vol_E + altura):** peso = -0.4888·vol + +6.0960·h + -325.6 → MAE 9.9 kg

## Caso: `sin_v12_v14_v15` (n=10)

IDs incluidos: v1, v2, v3, v4, v5, v7, v8, v9, v10, v13

| Variante | k medio kg/L | CV % | corr(vol,peso) | proporcional a*vol | MAE prop | lineal a*vol+b | MAE lin |
|----------|--------------|------|----------------|--------------------|----------|----------------|---------|
| A_mediana | 2.742 ± 0.562 | 20.5 | +0.156 | 2.6090·vol | 26.1 kg | 0.2801·vol + 166.6 | 17.1 kg |
| B_p75 | 2.301 ± 0.450 | 19.6 | +0.213 | 2.1952·vol | 25.4 kg | 0.3208·vol + 159.5 | 16.8 kg |
| E_envelope | 1.357 ± 0.311 | 22.9 | -0.137 | 1.2615·vol | 37.1 kg | -0.1063·vol + 201.2 | 17.0 kg |

**Sólo altura:** peso = 3.999·h + -195.0 → MAE 13.8 kg, max 29.5 kg, corr(h,peso)=+0.583

**Multivar (vol_E + altura):** peso = -0.4695·vol + +6.1487·h + -333.1 → MAE 10.7 kg

## Conclusión

- Con los 13 individuos la correlación vol↔peso es **negativa** en las 3 variantes → el volumen del barril (esta pipeline) no es predictor confiable de peso.
- Quitando v14, v15 y v12 la correlación se vuelve **positiva débil** para A y B (+0.16, +0.21), pero CV sigue ~20%.
- La menos mala como constante única: **k ≈ 2.30 kg/L** sobre `vol_B_p75` (sin v12,v14,v15). Implica error ~±20% en peso.
- La altura sola predice mejor que cualquier volumen (corr +0.58, MAE ~14 kg).
- Antes de fijar constante, conviene revisar máscaras infladas (v14,v15) y/o filtrar frames de baja cobertura.
