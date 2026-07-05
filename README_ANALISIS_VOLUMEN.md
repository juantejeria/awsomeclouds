# Análisis del cálculo de volumen del barril (2026-05-19)

Documento de razonamiento generado durante la sesión de análisis de los datasets
`6mayo` (20 individuos) y `14mayo` (14 individuos). 34 individuos en total con
peso real conocido (codificado en el nombre `<altura>_<peso>`).

---

## 1. ¿Qué volumen usa el pipeline y qué representa cada mesh?

### Las 3 mallas que se exportan por individuo

| Mesh | Cómo se construye | Volumen geométrico (ejemplo `116_316`) |
|---|---|---|
| `_lateral.obj` | Triangulación 2D del contorno del barril (Z=0) | — (es 2D) |
| `_3d.obj` | Envelope simétrico: lateral + `profundidad_eliptica(x,y)` espejado en Z | 360.4 L |
| `_volumen.obj` | Tubo de 80 anillos elípticos apilados, `K_DEPTH=0.25` constante | 339.3 L |

### El volumen que va al peso

El pipeline reporta `vol_barril_litros` en el JSON, igual a `vol_consenso_E_envelope`,
y multiplica por `1.03` para estimar peso (`app.py:3131`, `generar_modelos3d_grandes.py:1947`).

**Fórmula efectiva** (`procesar_21_frames.py:186` y `app.py:2188-2201`):

```
para cada columna X del barril:
    h_cm = altura silueta en esa columna
    a = h_cm / 2          (semi-eje vertical)
    b = h_cm * 0.25       (semi-eje profundidad)
    vol_cm3 += π * a * b * dx_cm

vol_total_litros = vol_cm3 / 1000
peso_kg = vol_total_litros * 1.03
```

El `K_DEPTH = 0.25` es una **constante hardcodeada para todos los individuos**.
Significa: la profundidad del barril es 0.25 × la altura del barril.

### Los 3 modos de consenso multi-frame

`procesar_21_frames.py` toma 21 frames del video y combina las siluetas:

- `vol_consenso_A_mediana`: por cada columna X, toma la **mediana** de altura entre los frames.
- `vol_consenso_B_p75`: usa percentil 75 (más conservador).
- `vol_consenso_E_envelope`: usa `max(top) + min(bot)` por columna → **silueta máxima de la unión**.

El JSON guarda los 3, y reporta `vol_barril_litros = vol_consenso_E_envelope`.

---

## 2. Hallazgos sobre las mallas

### 2.1. `_3d.obj` no es manifold (12% de aristas abiertas)

`generar_modelos3d_grandes.py:854-877` construye dos mitades (Z>0 y Z<0) que
**comparten posición del rim pero NO indices**. Comentario en el código:

> *"NO se sueldan indices: cada mitad sigue siendo una malla independiente"*

Resultado: ~12% de las aristas pertenecen a un solo triángulo (manifold abierto).

**Sin embargo, el volumen geométrico medido por el teorema de divergencia es
correcto igual**, porque el rim abierto está en Z=0 y la contribución
`(x,y,0)·(0,0,±1) = 0`. La tapa faltante no aporta volumen al integral.

→ Se generó un script de reparación (`/tmp/repair_3d.py`) que suelda los vértices
del rim. Los `_3d_closed.obj` resultantes tienen 0 aristas abiertas y volumen
**idéntico** a los originales.

Zip con mallas reparadas: `/Users/usuario/Downloads/modelos3d_6mayo_14mayo_obj_FIXED.zip`

### 2.2. `_volumen.obj` sí es cerrado pero con tapas de normal invertida

0 aristas abiertas (manifold), pero las tapas elípticas de los extremos del tubo
tienen winding inverso respecto a las paredes laterales. **El teorema de
divergencia da resultado a la mitad del verdadero**.

Solución: integrar el volumen anillo-por-anillo usando la fórmula `π·a·b·dx`
sobre los 80 anillos. Ese cálculo da el volumen correcto (339.3 L para `116_316`,
coincide con `vol_consenso_E_envelope=344` dentro del 1.5%).

---

## 3. El problema central: el peso predicho NO correlaciona con el peso real

### 3.1. Regresión empírica con LOO cross-validation

Con los 34 individuos (peso real del nombre del archivo):

| Modelo | R² (LOO) | MAE (kg) |
|---|---|---|
| **Baseline (`vol_E × 1.03`, método actual)** | **0.072** | **60.2** |
| Solo `vol_3d_obj` (lineal) | 0.362 | 52.1 |
| Solo `vol_E_envelope` (lineal) | 0.300 | 55.1 |
| **Solo `vol_A_mediana` (lineal)** | **0.563** | 41.3 |
| **`vol_E + altura` (lineal)** | **0.700** | **33.9** |
| Todas + Ridge | 0.632 | 38.9 |

**Conclusión brutal:** el método actual (`vol × 1.03`) tiene R²=0.07 → **predice
peor que usar el promedio del dataset**. Es esencialmente random.

Curiosamente, en el mejor modelo (`vol_E + altura`) el coeficiente del volumen es
casi cero (-0.01 kg/L) y el de altura es +7.45 kg/cm. **Significa que la altura
sola predice casi tan bien como cualquier combinación con el volumen.** Eso es
síntoma de que el volumen extraído no aporta información predictiva real.

### 3.2. Posibles causas

| Causa | Impacto estimado |
|---|---|
| `K_DEPTH=0.25` hardcodeado para todas las razas/edades | Alto: ±30% volumen |
| Envelope multi-frame infla por outliers de segmentación | Muy alto (ver §4) |
| `barril_seg.pt` incluye partes no-barril (cuello, grupa) | Medio-alto |
| `cm_per_px` calibrado contra bbox completo de COCO | Medio: error² → volumen³ |
| Densidad 1.03 asume tejido muscular (vivo real ≈ 0.95-1.00) | Bajo: ~5% |

---

## 4. Experimento: ¿qué pasa si usamos menos frames?

Hipótesis del usuario: si usamos pocos frames (en vez de 21), el envelope va a
estar menos inflado porque hay menos chance de capturar outliers.

### Resultado (volumen vs N frames, 4 individuos)

| Individuo | env N=1 | env N=3 | env N=5 | env N=21 | mediana (cualquier N) |
|---|---|---|---|---|---|
| `116_316` | 227 | 260 | 315 | 361 (N=15) | **~205 L estable** |
| `120_321` | 174 | 199 | 265 | **445** | **~180 L estable** |
| `100_137.5` | 94 | 101 | **498** ⚠ | **606** | **~75 L estable** |
| `110_228` | 80 | 103 | 111 | 154 | **~88 L estable** |

(PNG: `/Users/usuario/Downloads/vol_vs_N_frames.png`)

### 3 conclusiones del experimento

1. **La mediana es robusta al N** — cambia <10% entre N=3 y N=21.
2. **El envelope crece monotónicamente con N** — más frames = más inflación.
3. **El envelope es frágil: un solo mal frame lo destruye.**
   `100_137.5` salta de 101 L (N=3) a **498 L (N=5)**. El 4º o 5º frame tiene
   una mask que se "fugó" (pasto, sombra, otra vaca). Como envelope = MAX,
   ese error contamina TODO.

---

## 5. Plan de acción propuesto

### Cambios chicos, alto impacto (corto plazo)

1. **Reemplazar `vol_barril_litros = vol_consenso_E_envelope` por `vol_consenso_A_mediana`**.
   - 1 línea en `procesar_21_frames.py:296` (y variantes filtrado/postes).
   - El R² del baseline pasa de 0.07 a 0.56 solo con este cambio.

2. **Aplicar factor de corrección empírico** (de la regresión):
   - `peso_kg = α × vol_mediana + β × altura + δ`
   - Calibrar con los 34 individuos disponibles.

3. **Auditoría visual de masks** en individuos donde el envelope explota
   (`100_137.5` salta de 101 a 498 L). Probablemente `barril_seg.pt` se está
   fugando a algo fuera del barril en 1-2 frames.

### Cambios medianos (mediano plazo)

4. **Calibrar `K_DEPTH` empíricamente** por dataset o por subgrupo (edad, sexo, etc.).
   Probablemente debería estar ~0.32-0.38 en vez de 0.25.

5. **Hacer `K_DEPTH(x)` variable** a lo largo del barril (más profundo en el pecho/rumen
   que en la grupa). Necesita medición top-view en al menos algunos individuos.

6. **Re-entrenar `barril_seg.pt`** con más data curada para que la mask sea
   estrictamente el barril anatómico (rumen + tórax), sin cuello/grupa/patas.

### Cambios grandes (largo plazo)

7. **Visual hull / space carving** con frames lateral + frontal + top-view.
   Elimina la suposición elíptica → volumen geométrico verdadero.

---

## 6. Artefactos generados en esta sesión

Todos en `/Users/usuario/Downloads/`:

| Archivo | Qué es |
|---|---|
| `modelos3d_6mayo_14mayo_obj.zip` | OBJ original (`_3d.obj` con rim abierto) |
| `modelos3d_6mayo_14mayo_obj_FIXED.zip` | OBJ con `_3d.obj` reparado (rim soldado) |
| `modelos3d_6mayo_14mayo_dxf.zip` | DXF para AutoCAD (138 archivos) |
| `volumenes_por_file.csv` | Volúmenes geométricos por individuo (`_volumen.obj`, `_3d.obj`) |
| `volumenes_3d_cerrado.csv` | Aristas abiertas antes/después de reparar |
| `volumenes_FIXED.csv` | Volúmenes sobre el zip FIXED |
| `volumenes_6mayo_14mayo.csv` | Volúmenes + comparación con JSON + peso predicho |
| `regresion_peso_real.png` | Scatter baseline vs mejor modelo |
| `regresion_peso_predicciones.csv` | Predicciones por individuo |
| `vol_vs_N_frames.png` | Volumen envelope/mediana en función de N frames |
| `rim_abierto_116_316.png`, `rim_abierto_100_137.5.png` | Visualización del rim abierto |

Scripts auxiliares (no commiteados al repo, en `/tmp/`):
- `ply_to_obj.py` — convierte PLY → OBJ
- `obj_to_dxf.py` — convierte OBJ → DXF
- `mesh_volumes_final.py` — calcula volumen por anillos (`_volumen.obj`) y divergencia (`_3d.obj`)
- `repair_3d.py` — repara `_3d.obj` soldando rim
- `viz_rim_open.py` — diagrama del rim abierto
- `regresion_peso.py` — regresión empírica peso real vs features
- `test_frame_subsets.py` — experimento de N frames variando

---

## 7. Próximo paso (donde quedamos)

El usuario plantea: **si la mediana es estable con N, y el envelope se inflado con un solo
mal frame, vale la pena explorar si usar pocos frames (3-5) mejora la calidad de la
predicción** — combinado con mediana o p75 en vez de envelope.

El experimento mostró que la mediana **no necesita** menos frames (ya es robusta).
Pero queda pendiente:

- [ ] Probar si la **estrategia óptima** es `mediana(N=21)` + corrección empírica,
      o si conviene un híbrido (ej. `mediana(top 5 frames más estables)`).
- [ ] Si hay frames "malos" que arruinan la segmentación, podríamos detectarlos
      y filtrarlos automáticamente (medir consistencia del width entre frames).
- [ ] Probar el cambio de envelope → mediana en producción y re-medir el R² real.
