# templates/

Templates HTML (Jinja2) para la interfaz web Flask.

## Archivos

### `base.html` — Template base

Template padre que define la estructura HTML compartida:
- CDN de Bootstrap 4, jQuery 3.3, Font Awesome 5
- Navbar (actualmente vacío, solo estilo oscuro)
- Bloque `{% block content %}` para contenido de cada página
- Carga `static/css/main.css` y `static/js/engine.js`

### `chooser.html` — Selector de granja

Página de inicio (`/`). Muestra:
- Jumbotron con fondo de imagen (wallpaper con blur)
- Dropdown con las granjas que tienen un modelo entrenado (`chckpt.best.h5`)
- Botón "Cargar modelo" que envía POST a `/load_model`

No extiende `base.html` (tiene su propio layout completo).

### `index.html` — Interfaz principal

Página principal tras seleccionar granja (`/load_model`). Extiende `base.html`. Contiene:

**Columna izquierda:**
- Upload de imagen/video (JPG, PNG, MP4, AVI, MOV, MKV)
- Checkboxes: reconocimiento de ganado, estimación de peso
- Selector de método de escala (ojos, poste rojo, ambos)
- Selectores de raza, categoría y edad (corrección de peso)
- Opciones de video (debug, sample_rate)
- Preview de la imagen/video seleccionado
- Botones: Analizar, Detectar Referencias, Detectar Postes

**Columna derecha:**
- Card de resultados del video (tabla unificada)
- Card de estimación de peso (imagen anotada)
- Card de referencias de escala (postes detectados)

**Modal de ampliación:**
- Muestra frames en tamaño completo al hacer click
- Botón de descarga del frame con anotaciones
- Cierre con ESC, click fuera o botones
