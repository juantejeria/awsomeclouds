# static/

Archivos estáticos del frontend: CSS, JavaScript e imágenes.

## Estructura

```
static/
├── css/
│   └── main.css        # Estilos principales de la aplicación
├── js/
│   └── engine.js       # Lógica completa del frontend
└── img/
    └── cows-wallpaper.jpg  # Imagen de fondo
```

## `js/engine.js`

Contiene toda la lógica del frontend en JavaScript/jQuery:

- **Manejo de archivos**: detecta si el archivo subido es imagen o video, muestra preview
- **Requests AJAX**: envía la imagen/video a los endpoints Flask (`/predict`, `/predict_video`, `/detect_reference_points`)
- **Procesamiento de resultados de imagen**: muestra reconocimiento (tabla de probabilidades) y peso estimado con imagen anotada
- **Procesamiento de resultados de video** (`processVideoResults()`):
  - Construye tabla unificada con todas las vacas detectadas
  - Muestra identidad (votación ponderada), peso (media/rango), frame de detección
  - Galería de frames con peso y frames de calibración (postes)
  - Modal de ampliación con descarga de frames
- **Opciones de la UI**: toggle de reconocimiento/peso, selector de método de escala, corrección por raza/categoría/edad, opciones de video (debug, sample_rate)

## `css/main.css`

Estilos complementarios a Bootstrap 4:
- Layout responsive con fondo fijo (wallpaper con blur)
- Cards con transparencia y backdrop-filter
- Estilos para checkboxes grandes, labels de upload
- Estilos para tablas de resultados y galería de frames

## Comunicación con el backend

```
Frontend (engine.js)           Backend (app.py)
─────────────────              ───────────────
Upload imagen    ──POST /predict──►  Procesa imagen
                 ◄── JSON ────────   {recognition, weight, weight_image}

Upload video     ──POST /predict_video──►  Procesa video
                 ◄── JSON ──────────────   {cows: {cow_0: {...}, ...}, stats}

Detectar refs    ──POST /detect_reference_points──►  Detecta postes
                 ◄── JSON ────────────────────────   {poste1, poste2, image}
```

Las imágenes anotadas se envían como strings base64 (`data:image/jpeg;base64,...`) dentro del JSON de respuesta.
