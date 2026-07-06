# Variables de calibración — única fuente de verdad

Las tres constantes que gobiernan todas las mediciones del sistema viven en
**`config.ini`, sección `[calibracion]`**. No se hardcodean en ningún otro
archivo activo.

| Variable | Valor actual | Qué controla | Sensibilidad |
|---|---|---|---|
| `vara_cm` | 110 | Altura real de la vara/cinta roja de referencia. Define la escala px→cm de **todas** las mediciones (altura del animal, largo, volúmenes). | Lineal en altura, **cúbica en peso**: +1% de vara → +3% de peso. |
| `k_depth` | 0.25 | Ratio profundidad/altura del corte elíptico del barril en el modelo 3D. Semi-eje de profundidad = alto de la rebanada × k_depth. | Lineal en volumen/peso. |
| `densidad_kg_l` | 1.03 | Conversión volumen→peso: `peso_kg = volumen_L × densidad_kg_l`. | Lineal en peso. |

## Cómo fluyen los valores

```
config.ini [calibracion]
    └── core/calibracion.py          ← los lee UNA vez (con defaults si faltan)
          ├── app.py                  (escala por postes, K_DEPTH del contorno vivo, peso)
          ├── core/weight_estimation.py   (escala por postes, labels de overlay)
          ├── core/generar_modelos3d_grandes.py  (volumen elíptico, peso)
          ├── core/reconstruccion_3d.py           (peso)
          └── templates/base.html → window.VARA_CM → static/js/engine.js
                                     (labels del overlay, regla, distancias en cm)
```

Para cambiar un valor: editar `config.ini` y reiniciar la app. Nada más.

## Reglas

1. **Nunca** escribir `110`, `0.25` o `1.03` como literal en código nuevo.
   Backend: `from core.calibracion import VARA_CM, K_DEPTH, DENSIDAD_KG_L`.
   Frontend: usar la global `VARA_CM` (inyectada por `base.html`).
2. Si se cambia la vara física, cambiar `vara_cm` — todas las mediciones
   nuevas quedan consistentes. Las históricas conservan la escala con la que
   se generaron (los `resumen.json` guardan sus cm ya convertidos).
3. Los scripts de `archive/` conservan valores hardcodeados de su época;
   son históricos y no se corrigen.

## Historia

- Hasta 2026-07-06 el frontend usaba **112** cm y el backend **110** — las
  mediciones de la UI (regla, distancias) quedaban +1.8% infladas. Unificado
  a 110 (valor real de la vara).
- `k_depth = 0.25` y `densidad = 1.03` validados el 2026-07-06 contra un
  ternero de peso real conocido (125 kg): consenso mediana + altura real
  → 127.3 kg (+1.8%). Ver docs/README_ANALISIS_VOLUMEN.md para el análisis
  histórico envelope vs mediana.
