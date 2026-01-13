from __future__ import annotations

import os
import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Configurar variables de entorno ANTES de importar cualquier cosa de PyTorch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import json
from datetime import datetime
from PIL import Image

# Configuración de página
st.set_page_config(
    page_title="Sistema de Reconocimiento de Ganado",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado con paleta de colores natural y diseño mejorado
st.markdown("""
<style>
    /* Paleta de colores natural */
    :root {
        --primary-green: #4a7c59;
        --secondary-green: #6b8e5a;
        --accent-brown: #8b6f47;
        --light-beige: #f5f1e8;
        --dark-green: #2d4a3a;
        --earth-brown: #a0826d;
        --card-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main {
        background-color: #faf9f6;
    }
    
    .stButton>button {
        background-color: #4a7c59;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #2d4a3a;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton>button:focus {
        box-shadow: 0 0 0 0.2rem rgba(74, 124, 89, 0.5);
    }
    
    /* Cards con tonos sutiles diferentes para diferenciación - sin espacios innecesarios */
    .card-container {
        background-color: #f7f7f7;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    
    /* Ocultar card-container vacío */
    .card-container:empty {
        display: none !important;
    }
    
    .card-container:has(> :only-child.card-header:empty) {
        display: none !important;
    }
    
    .card-header {
        color: #2d4a3a;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e5e5e5;
    }
    
    .section-header {
        color: #2d4a3a;
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        margin-top: 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #d5d5d5;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .subsection-header {
        color: #4a7c59;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    .info-box {
        background-color: #f5f1e8;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background-color: #fff4e6;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    /* Columnas con tonos ligeramente diferentes */
    .column-container {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 6px;
        height: 100%;
    }
    
    .file-list-item {
        padding: 0.5rem 0.75rem;
        margin: 0.3rem 0;
        background-color: #f2f2f2;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    
    .metric-container {
        background-color: #f4f4f4;
        padding: 0.5rem;
        border-radius: 4px;
    }
    
    /* Eliminar espacios innecesarios de Streamlit */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    .stMarkdown {
        margin-bottom: 0.25rem !important;
    }
    
    /* Reducir espacios entre elementos de Streamlit */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    
    /* Reducir espacios de separadores */
    hr {
        margin: 0.5rem 0 !important;
        border: none;
        height: 1px;
        background: #e0e0e0;
    }
    
    /* Reducir espacios en columnas */
    [data-testid="column"] {
        padding: 0.25rem !important;
    }
    
    /* Reducir espacios en métricas */
    [data-testid="stMetricValue"] {
        margin-bottom: 0 !important;
    }
    
    /* Reducir espacios en containers */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Sidebar mejorado */
    [data-testid="stSidebar"] {
        background-color: #f5f1e8;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-weight: 500;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.25rem 0;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #e8e8e8;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .stButton>button {
            width: 100%;
            margin: 0.25rem 0;
        }
        
        .section-header {
            font-size: 1.4rem;
        }
        
        .card-container, .column-container {
            padding: 1rem;
        }
    }
    
    /* Separadores visuales */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, #4a7c59, transparent);
        margin: 2rem 0;
    }
    
    /* Estilo para el título del establecimiento en el header */
    h2 {
        color: #2d4a3a;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Importar módulos de gestión
from src.dataset_manager import (
    create_animal,
    create_establecimiento,
    extract_frames_from_uploaded_video,
    get_animal_dir,
    get_dataset_stats,
    get_establecimiento_dir,
    list_animales,
    list_establecimientos,
    save_uploaded_file,
    validate_dataset_for_training,
)
from src.io_utils import list_establecimientos as list_trained_establecimientos


def get_selected_establecimiento(establecimientos: list[str], default_key: str) -> str | None:
    """Obtiene el establecimiento seleccionado, usando el cache si existe."""
    if not establecimientos:
        return None
    
    # Si hay un establecimiento guardado en session_state y está en la lista, usarlo
    cached_estab = st.session_state.get("last_selected_estab")
    if cached_estab and cached_estab in establecimientos:
        return cached_estab
    
    # Si no hay cache, usar el primero de la lista
    return establecimientos[0] if establecimientos else None


def main():
    # Título principal
    st.title("Sistema de Reconocimiento de Ganado")
    
    # Obtener establecimiento seleccionado (del cache o de la lista)
    establecimientos = list_establecimientos()
    current_page = st.session_state.get("current_page", None)
    
    # Si hay una página activa, obtener el establecimiento del cache
    if current_page:
        selected_estab = st.session_state.get("last_selected_estab")
        if not selected_estab and establecimientos:
            # Si no hay cache pero hay establecimientos, usar el primero
            selected_estab = establecimientos[0]
            st.session_state["last_selected_estab"] = selected_estab
        
        # Si hay página activa y establecimiento seleccionado, mostrar header con título y botón volver
        if selected_estab:
            # Header con título del establecimiento y botón volver
            col_header1, col_header2 = st.columns([4, 1])
            with col_header1:
                st.markdown(f'<h2 style="margin-top: 0; margin-bottom: 0.5rem;">{selected_estab}</h2>', unsafe_allow_html=True)
            with col_header2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Volver al Dashboard", key="back_to_dashboard_header", use_container_width=True):
                    st.session_state["current_page"] = None
                    st.rerun()
            
            st.markdown("---")
            
            # Renderizar la página correspondiente
            if current_page == "Gestión de Datasets":
                render_dataset_management(selected_estab)
            elif current_page == "Entrenamiento":
                render_training(selected_estab)
            elif current_page == "Predicción":
                render_prediction(selected_estab)
            return
    
    # Si no hay página activa, mostrar el dashboard principal
    st.markdown("---")
    st.markdown('<div class="section-header">Dashboard Principal</div>', unsafe_allow_html=True)
    
    # Selección o creación de establecimiento
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if establecimientos:
            # Obtener el establecimiento guardado o el primero
            default_estab = get_selected_establecimiento(establecimientos, "main_estab")
            default_index = establecimientos.index(default_estab) if default_estab else 0
            
            selected_estab = st.selectbox(
                "Seleccionar establecimiento",
                options=establecimientos,
                index=default_index,
                key="main_estab",
                help="Selecciona el establecimiento con el que quieres trabajar"
            )
            
            # Guardar en cache
            st.session_state["last_selected_estab"] = selected_estab
        else:
            selected_estab = None
            st.info("No hay establecimientos creados. Crea uno nuevo para comenzar.")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Crear nuevo establecimiento", key="new_estab_main", use_container_width=True):
            st.session_state["show_new_estab"] = True
    
    # Formulario para crear nuevo establecimiento
    if st.session_state.get("show_new_estab", False):
        st.markdown("---")
        st.markdown('<div class="card-container"><div class="card-header">Crear Nuevo Establecimiento</div>', unsafe_allow_html=True)
        
        with st.form("create_estab_form"):
            new_estab_name = st.text_input("Nombre del establecimiento", placeholder="Ej: Establecimiento_01")
            
            col_submit1, col_submit2 = st.columns([1, 1])
            with col_submit1:
                submit = st.form_submit_button("Crear", type="primary")
            with col_submit2:
                cancel = st.form_submit_button("Cancelar")
            
            if submit:
                if new_estab_name.strip():
                    try:
                        create_establecimiento(new_estab_name.strip())
                        st.success(f"Establecimiento '{new_estab_name.strip()}' creado exitosamente.")
                        st.session_state["show_new_estab"] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al crear establecimiento: {e}")
                else:
                    st.error("El nombre no puede estar vacío.")
            
            if cancel:
                st.session_state["show_new_estab"] = False
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Si no hay establecimiento seleccionado, mostrar mensaje y salir
    if not selected_estab:
        st.markdown("---")
        st.info("Selecciona o crea un establecimiento para continuar.")
        return
    
    st.markdown("---")
    
    # Opciones de navegación
    st.markdown('<div class="card-container"><div class="card-header">Opciones Disponibles</div>', unsafe_allow_html=True)
    
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    
    with col_opt1:
        if st.button("Gestión de Dataset", key="nav_dataset", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "Gestión de Datasets"
            st.rerun()
    
    with col_opt2:
        if st.button("Entrenamiento", key="nav_training", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "Entrenamiento"
            st.rerun()
    
    with col_opt3:
        if st.button("Predicción", key="nav_prediction", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "Predicción"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mostrar información del establecimiento seleccionado
    if selected_estab:
        st.markdown("---")
        stats = get_dataset_stats(selected_estab)
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric("Total Animales", stats["total_animales"])
        with col_info2:
            st.metric("Total Muestras", sum(a["total"] for a in stats["animales"].values()))
        with col_info3:
            st.metric("Mínimo Muestras", stats["min_samples"])
        with col_info4:
            # Verificar si hay modelo entrenado
            from src.dataset_manager import get_artifacts_base_dir
            artifacts_dir = get_artifacts_base_dir() / selected_estab
            has_model = (artifacts_dir / "model.pt").exists()
            st.metric("Modelo Entrenado", "Sí" if has_model else "No")


def render_create_animal(selected_estab: str):
    """Formulario para crear un nuevo individuo."""
    st.markdown('<div class="card-container"><div class="card-header">Crear Nuevo Individuo</div>', unsafe_allow_html=True)
    
    with st.form("create_animal_form"):
        new_animal_name = st.text_input("Nombre del individuo", placeholder="Ej: v1_vaca_450", help="Nombre único para identificar este individuo")
        
        col_submit1, col_submit2 = st.columns([1, 1])
        with col_submit1:
            submit = st.form_submit_button("Crear Individuo", type="primary")
        with col_submit2:
            cancel = st.form_submit_button("Cancelar")
        
        if submit:
            if new_animal_name.strip():
                try:
                    create_animal(selected_estab, new_animal_name.strip())
                    st.success(f"Individuo '{new_animal_name.strip()}' creado exitosamente.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al crear individuo: {e}")
            else:
                st.error("El nombre no puede estar vacío.")
        
        if cancel:
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_edit_animal(selected_estab: str):
    """Sección para editar individuos existentes."""
    animales = list_animales(selected_estab)
    
    if not animales:
        st.info("No hay individuos creados. Crea uno nuevo en la pestaña 'Crear Nuevo Individuo'.")
        return
    
    st.markdown('<div class="card-container"><div class="card-header">Editar Individuo Existente</div>', unsafe_allow_html=True)
    
    selected_animal = st.selectbox(
        "Seleccionar individuo a editar",
        options=animales,
        key="edit_animal_select",
    )
    
    if selected_animal:
        st.markdown("---")
        st.markdown(f"**Editando: {selected_animal}**")
        
        # Mostrar archivos actuales
        animal_dir = get_animal_dir(selected_estab, selected_animal)
        if animal_dir.exists():
            image_files = sorted([f.name for f in animal_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            video_files = sorted([f.name for f in animal_dir.iterdir() if f.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']])
            
            col_files1, col_files2 = st.columns(2)
            
            with col_files1:
                st.markdown("**Imágenes:**")
                if image_files:
                    for img_file in image_files[:10]:
                        col_del1, col_del2 = st.columns([4, 1])
                        with col_del1:
                            st.text(img_file)
                        with col_del2:
                            if st.button("Eliminar", key=f"del_img_{img_file}", help="Eliminar"):
                                (animal_dir / img_file).unlink()
                                st.success(f"Eliminado: {img_file}")
                                st.rerun()
                    if len(image_files) > 10:
                        st.caption(f"... y {len(image_files) - 10} imágenes más")
                else:
                    st.info("No hay imágenes")
            
            with col_files2:
                st.markdown("**Videos:**")
                if video_files:
                    for vid_file in video_files:
                        col_del1, col_del2 = st.columns([4, 1])
                        with col_del1:
                            st.text(vid_file)
                        with col_del2:
                            if st.button("Eliminar", key=f"del_vid_{vid_file}", help="Eliminar"):
                                (animal_dir / vid_file).unlink()
                                st.success(f"Eliminado: {vid_file}")
                                st.rerun()
                else:
                    st.info("No hay videos")
        
        st.markdown("---")
        st.markdown("**Agregar nuevos archivos:**")
        # Reutilizar la lógica de subida de archivos
        render_upload_files(selected_estab, selected_animal, context="edit")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_upload_files(selected_estab: str, selected_animal: str, context: str = "default"):
    """Renderiza la sección de subida de archivos.
    
    Args:
        selected_estab: Nombre del establecimiento
        selected_animal: Nombre del animal
        context: Contexto de uso para hacer las claves únicas ("edit", "file_mgmt", etc.)
    """
    animal_dir = get_animal_dir(selected_estab, selected_animal)
    # Crear prefijo único para las claves basado en el contexto
    key_prefix = f"{context}_{selected_animal}"
    
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        st.markdown('<div class="column-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-header">Archivos de {selected_animal}</div>', unsafe_allow_html=True)
        
        if animal_dir.exists():
            image_files = sorted([f.name for f in animal_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            video_files = sorted([f.name for f in animal_dir.iterdir() if f.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']])
            
            if image_files:
                st.markdown("**Imágenes:**")
                for img_file in image_files[:10]:
                    st.markdown(f'<div class="file-list-item">{img_file}</div>', unsafe_allow_html=True)
                if len(image_files) > 10:
                    st.caption(f"... y {len(image_files) - 10} imágenes más")
            
            if video_files:
                st.markdown("**Videos:**")
                for vid_file in video_files:
                    st.markdown(f'<div class="file-list-item">{vid_file}</div>', unsafe_allow_html=True)
            
            if not image_files and not video_files:
                st.info("No hay archivos subidos aún.")
        else:
            st.info("El directorio del animal aún no existe.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="column-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-header">Subir Nuevos Archivos</div>', unsafe_allow_html=True)
        
        # Resto del código de subida de archivos (el que ya teníamos)
        file_type = st.radio(
            "Tipo de archivo a subir",
            ["Imágenes", "Videos"],
            horizontal=True,
            key=f"file_type_{key_prefix}",
        )
        
        if file_type == "Imágenes":
            capture_method = st.radio(
                "Método de captura",
                ["Subir archivo", "Tomar foto con cámara"],
                horizontal=True,
                key=f"capture_method_{key_prefix}",
            )
        else:
            capture_method = "Subir archivo"
        
        uploaded_files = []
        camera_photo = None
        
        if capture_method == "Tomar foto con cámara":
            camera_photo = st.camera_input(
                "Tomar foto",
                key=f"camera_input_{key_prefix}",
                help="Permite tomar una foto directamente desde la cámara de tu dispositivo"
            )
            if camera_photo:
                uploaded_files = [camera_photo]
        else:
            uploaded_files = st.file_uploader(
                f"Seleccionar {file_type.lower()}",
                type=["jpg", "jpeg", "png"] if file_type == "Imágenes" else ["mp4", "mov", "avi", "mkv"],
                accept_multiple_files=True,
                key=f"upload_files_{key_prefix}",
                help=f"Puedes subir múltiples {file_type.lower()} a la vez"
            )
        
        if uploaded_files:
            if capture_method == "Tomar foto con cámara":
                st.info("Foto capturada. Haz clic en 'Guardar foto' para guardarla.")
            else:
                st.info(f"Seleccionados: {len(uploaded_files)} archivo(s)")
            
            extract_frames = False
            stride = 30
            max_frames = 50
            filter_faces = False
            require_features = None
            
            if file_type == "Videos":
                st.markdown("**Opciones de extracción de frames:**")
                extract_frames = st.checkbox("Extraer frames automáticamente", value=True, key=f"extract_check_{key_prefix}")
                
                if extract_frames:
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        stride = st.number_input("Stride (cada N frames)", min_value=1, max_value=60, value=30, key=f"stride_input_{key_prefix}")
                    with col_s2:
                        max_frames = st.number_input("Máximo frames por video", min_value=10, max_value=200, value=50, key=f"max_frames_input_{key_prefix}")
                    
                    st.markdown("**Filtrado de frames:**")
                    filter_faces = st.checkbox(
                        "Solo guardar frames con animal detectado", 
                        value=False,
                        key=f"filter_faces_check_{key_prefix}",
                        help="Si está habilitado, solo se guardarán frames donde YOLO detecte un animal (vaca, perro, gato, etc.). Esto mejora la calidad del dataset al evitar frames sin animales visibles."
                    )
                    
                    if filter_faces:
                        feature_options = st.multiselect(
                            "Tipo de detección requerida (opcional)",
                            options=["cow", "animal"],
                            default=[],
                            key=f"require_features_select_{key_prefix}",
                            help="• 'cow': Solo acepta frames con vacas detectadas específicamente\n• 'animal': Acepta cualquier animal (vaca, perro, gato, caballo, etc.)\nSi no seleccionas ninguna, se acepta cualquier animal detectado."
                        )
                        if feature_options:
                            require_features = feature_options
            
            button_label = "Guardar foto" if capture_method == "Tomar foto con cámara" else "Guardar archivos"
            if st.button(button_label, key=f"save_files_{key_prefix}", type="primary"):
                animal_dir = get_animal_dir(selected_estab, selected_animal)
                progress_bar = st.progress(0)
                status_text = st.empty()
                saved_count = 0
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        file_name = uploaded_file.name if hasattr(uploaded_file, 'name') and uploaded_file.name else f"camera_photo_{idx + 1}.jpg"
                        status_text.text(f"Procesando {file_name}...")
                        
                        if file_type == "Imágenes":
                            if capture_method == "Tomar foto con cámara":
                                from datetime import datetime
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                file_name = f"camera_{timestamp}_{idx + 1}.jpg"
                            
                            dest_path = animal_dir / file_name
                            save_uploaded_file(uploaded_file, dest_path)
                            saved_count += 1
                        else:  # Videos
                            dest_path = animal_dir / file_name
                            save_uploaded_file(uploaded_file, dest_path)
                            
                            if extract_frames:
                                frames_extracted = extract_frames_from_uploaded_video(
                                    dest_path, 
                                    animal_dir, 
                                    stride=stride, 
                                    max_frames=max_frames,
                                    filter_faces=filter_faces if file_type == "Videos" else False,
                                    require_features=require_features if file_type == "Videos" else None,
                                )
                                saved_count += frames_extracted
                            else:
                                saved_count += 1
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    except Exception as e:
                        file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else f"archivo_{idx + 1}"
                        st.error(f"Error al guardar {file_name}: {e}")
                
                status_text.empty()
                progress_bar.empty()
                st.success(f"Se guardaron {saved_count} archivos exitosamente.")
                st.rerun()
        else:
            if capture_method == "Tomar foto con cámara":
                st.info("Toma una foto con la cámara para guardarla.")
            else:
                st.info("Selecciona archivos para subir usando el botón de arriba.")
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_dataset_management(selected_estab: str):
    """Sección de gestión de datasets."""
    st.markdown('<div class="section-header">Gestión de Dataset</div>', unsafe_allow_html=True)
    
    # Botones de navegación entre secciones
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    with col_nav1:
        st.markdown('<div style="text-align: center; padding: 0.5rem; background-color: #e8f5e9; border-radius: 6px; font-weight: 600;">Gestión de Dataset</div>', unsafe_allow_html=True)
    with col_nav2:
        if st.button("Entrenamiento", key="nav_to_training_from_mgmt", use_container_width=True):
            st.session_state["current_page"] = "Entrenamiento"
            st.rerun()
    with col_nav3:
        if st.button("Predicción", key="nav_to_prediction_from_mgmt", use_container_width=True):
            st.session_state["current_page"] = "Predicción"
            st.rerun()
    
    st.markdown("---")
    
    # Resumen del dataset
    stats = get_dataset_stats(selected_estab)
    st.markdown('<div class="card-container"><div class="card-header">Resumen del Dataset</div>', unsafe_allow_html=True)
    
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("Total Animales", stats["total_animales"])
    with col_info2:
        st.metric("Total Muestras", sum(a["total"] for a in stats["animales"].values()))
    with col_info3:
        st.metric("Mínimo Muestras", stats["min_samples"])
    with col_info4:
        # Verificar si hay modelo entrenado
        from src.dataset_manager import get_artifacts_base_dir
        artifacts_dir = get_artifacts_base_dir() / selected_estab
        has_model = (artifacts_dir / "model.pt").exists()
        st.metric("Modelo Entrenado", "Sí" if has_model else "No")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs para diferentes acciones
    tab1, tab2 = st.tabs(["Crear Nuevo Individuo", "Editar Individuo Existente"])
    
    with tab1:
        render_create_animal(selected_estab)
    
    with tab2:
        render_edit_animal(selected_estab)
    
def render_training(selected_estab: str):
    """Sección de entrenamiento."""
    st.markdown('<div class="section-header">Entrenamiento de Modelos</div>', unsafe_allow_html=True)
    
    # Botones de navegación entre secciones
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    with col_nav1:
        if st.button("Gestión de Dataset", key="nav_to_mgmt_from_training", use_container_width=True):
            st.session_state["current_page"] = "Gestión de Datasets"
            st.rerun()
    with col_nav2:
        st.markdown('<div style="text-align: center; padding: 0.5rem; background-color: #e8f5e9; border-radius: 6px; font-weight: 600;">Entrenamiento</div>', unsafe_allow_html=True)
    with col_nav3:
        if st.button("Predicción", key="nav_to_prediction_from_training", use_container_width=True):
            st.session_state["current_page"] = "Predicción"
            st.rerun()
    
    st.markdown("---")
    
    # Información sobre requisitos mínimos (no bloqueante)
    st.info(
        "**Recomendaciones para mejores resultados:**\n"
        "- Mínimo 30-60 imágenes por individuo\n"
        "- Variar iluminación, ángulos y distancia\n"
        "- Al menos 2 individuos diferentes\n"
        "\n"
        "Puedes entrenar con menos muestras, pero los resultados pueden ser menos precisos."
    )
    
    # Validar dataset (sin restricción de mínimo de muestras)
    is_valid, errors = validate_dataset_for_training(selected_estab, min_samples_per_animal=0)
    
    st.markdown("### Validación del Dataset")
    
    if is_valid:
        st.markdown(
            '<div class="success-box">Dataset válido para entrenamiento</div>',
            unsafe_allow_html=True,
        )
    else:
        # Mostrar errores de forma compacta
        errors_html = '<div class="warning-box"><strong>El dataset tiene problemas que deben corregirse antes de entrenar:</strong><ul style="margin-top: 0.75rem; margin-bottom: 0; padding-left: 1.5rem;">'
        for error in errors:
            errors_html += f'<li style="margin: 0.25rem 0; line-height: 1.4;">{error}</li>'
        errors_html += '</ul></div>'
        st.markdown(errors_html, unsafe_allow_html=True)
    
    # Estadísticas en layout mejorado
    stats = get_dataset_stats(selected_estab)
    
    col_stats1, col_stats2 = st.columns(2, gap="large")
    
    with col_stats1:
        st.markdown('<div class="card-container"><div class="card-header">Distribución por Animal</div>', unsafe_allow_html=True)
        
        if stats["animales"]:
            for animal, animal_stats in stats["animales"].items():
                total = animal_stats['total']
                images = animal_stats.get('imagenes', 0)
                frames = animal_stats.get('frames_video', 0)
                
                st.markdown(f"""
                <div style="padding: 0.75rem; margin: 0.5rem 0; background-color: #f9f9f9; border-radius: 6px; border-left: 3px solid #4a7c59;">
                    <strong>{animal}</strong><br>
                    <span style="color: #666; font-size: 0.9rem;">
                        Total: {total} muestras | Imágenes: {images} | Frames: {frames}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No hay animales en este establecimiento.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_stats2:
        st.markdown('<div class="card-container"><div class="card-header">Resumen Estadístico</div>', unsafe_allow_html=True)
        
        st.metric("Total animales", stats["total_animales"])
        
        # Mostrar estado de balanceo
        if stats["animales"]:
            diff = stats["max_samples"] - stats["min_samples"]
            if stats["balanceado"]:
                st.markdown(
                    '<div class="success-box" style="margin-top: 1rem; padding: 0.75rem;">Dataset balanceado correctamente</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="warning-box" style="margin-top: 1rem; padding: 0.75rem;">Dataset desbalanceado<br>Mínimo: {stats["min_samples"]} muestras<br>Máximo: {stats["max_samples"]} muestras<br>Diferencia: {diff} muestras<br><br><strong>Solución:</strong> Activa el balanceo automático abajo para equilibrar el dataset eliminando frames similares.</div>',
                    unsafe_allow_html=True,
                )
        st.metric("Mínimo muestras", stats["min_samples"])
        st.metric("Máximo muestras", stats["max_samples"])
        st.metric("Promedio muestras", f"{stats['promedio_samples']:.1f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Parámetros de entrenamiento en card
    st.markdown('<div class="card-container"><div class="card-header">Parámetros de Entrenamiento</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**Parámetros básicos:**")
        epochs = st.number_input("Épocas", min_value=1, max_value=100, value=10, step=1, help="Número de veces que el modelo verá todo el dataset")
        batch_size = st.number_input("Batch size", min_value=4, max_value=64, value=16, step=4, help="Número de muestras procesadas antes de actualizar el modelo")
        learning_rate = st.number_input("Learning rate", min_value=1e-5, max_value=1e-2, value=1e-4, step=1e-5, format="%.5f", help="Velocidad de aprendizaje del modelo")
    
    with col2:
        st.markdown("**Configuración de datos:**")
        img_size = st.number_input("Tamaño de imagen", min_value=128, max_value=512, value=224, step=32, help="Tamaño al que se redimensionarán las imágenes (píxeles)")
        val_frac = st.slider("Fracción de validación", min_value=0.1, max_value=0.4, value=0.2, step=0.05, help="Porcentaje del dataset usado para validación")
    
    st.markdown("---")
    
    # Sección de Balanceo de Dataset (más destacada)
    st.markdown('<div class="card-container"><div class="card-header">Balanceo de Dataset</div>', unsafe_allow_html=True)
    
    # Mostrar advertencia si está desbalanceado
    if stats and not stats.get("balanceado", True) and stats.get("animales"):
        diff = stats["max_samples"] - stats["min_samples"]
        st.warning(
            f"Tu dataset está desbalanceado. Diferencia de {diff} muestras entre animales. "
            f"Se recomienda activar el balanceo automático para mejorar el entrenamiento."
        )
    
    balance_dataset = st.checkbox(
        "Balancear dataset eliminando frames similares",
        value=True,
        help="Si está habilitado, antes de entrenar se balanceará el dataset llevando todas las clases al mismo número de muestras (el mínimo encontrado), eliminando frames similares de las clases con más muestras. Esto mejora significativamente la calidad del entrenamiento."
    )
    
    balance_similarity_threshold = 0.85
    if balance_dataset:
        col_bal1, col_bal2 = st.columns([2, 1])
        with col_bal1:
            balance_similarity_threshold = st.slider(
                "Umbral de similitud para balanceo",
                min_value=0.7,
                max_value=0.95,
                value=0.85,
                step=0.05,
                help="Umbral para considerar imágenes como similares. Valores más altos son más estrictos (eliminan más imágenes similares)."
            )
        with col_bal2:
            st.markdown("<br>", unsafe_allow_html=True)
            if stats and stats.get("animales"):
                min_samples = stats.get("min_samples", 0)
                st.caption(f"Muestras objetivo por animal: **{min_samples}**")
        
        st.info(
            "El balanceo analizará todas las imágenes con un modelo pre-entrenado para encontrar "
            "y eliminar frames similares. Esto puede tardar varios minutos dependiendo del tamaño del dataset."
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Botón de entrenamiento
    if st.button("Iniciar entrenamiento", disabled=not is_valid, type="primary"):
        with st.spinner("Iniciando entrenamiento..."):
            try:
                from src.training_ui import train_model
                
                process = train_model(
                    selected_estab,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=learning_rate,
                    img_size=img_size,
                    val_frac=val_frac,
                    filter_with_yolo=False,
                    yolo_require_features=None,
                    balance_dataset=balance_dataset,
                    balance_similarity_threshold=balance_similarity_threshold,
                )
                
                st.session_state["training_process"] = process
                st.session_state["training_estab"] = selected_estab
                st.success("Entrenamiento iniciado. El progreso aparecerá en unos segundos.")
                st.rerun()  # Refrescar para mostrar el progreso inmediatamente
                
            except Exception as e:
                st.error(f"Error al iniciar entrenamiento: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Mostrar estado del entrenamiento en curso
    training_estab = st.session_state.get("training_estab", selected_estab if is_valid else None)
    if training_estab:
        from src.training_status import get_training_status, is_training_running
        
        training_status = get_training_status(training_estab)
        
        if training_status and is_training_running(training_estab):
            st.markdown("---")
            st.markdown('<div class="card-container"><div class="card-header">Estado del Entrenamiento</div>', unsafe_allow_html=True)
            
            status_type = training_status.get("status", "running")
            current_epoch = training_status.get("current_epoch", 0)
            total_epochs = training_status.get("total_epochs", 0)
            
            # Mostrar mensaje de estado
            if status_type == "initializing" or current_epoch == 0:
                st.info(training_status.get("message", "Inicializando modelo y cargando datos... Esto puede tardar 30-60 segundos la primera vez (descarga de modelo pre-entrenado)."))
                st.caption("Si esto tarda más de 2 minutos, puede haber un problema. Revisa la terminal.")
            else:
                # Barra de progreso
                progress_percent = training_status.get("progress_percent", 0.0)
                st.progress(progress_percent / 100.0)
                st.caption(f"Época {current_epoch} de {total_epochs} ({progress_percent:.1f}%)")
                
                # Métricas actuales
                col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                with col_met1:
                    st.metric("Train Loss", f"{training_status.get('train_loss', 0.0):.4f}")
                with col_met2:
                    st.metric("Train Acc", f"{training_status.get('train_acc', 0.0):.3f}")
                with col_met3:
                    st.metric("Val Loss", f"{training_status.get('val_loss', 0.0):.4f}")
                with col_met4:
                    st.metric("Val Acc", f"{training_status.get('val_acc', 0.0):.3f}")
                
                # Mejor accuracy hasta ahora
                best_val_acc = training_status.get("best_val_acc", 0.0)
                if best_val_acc > 0:
                    st.info(f"Mejor accuracy de validación hasta ahora: {best_val_acc:.3f}")
            
            # Tiempo transcurrido
            elapsed_time = training_status.get("elapsed_time", 0.0)
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            st.caption(f"Tiempo transcurrido: {minutes}m {seconds}s")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Botón para refrescar manualmente y auto-refresh
            col_refresh1, col_refresh2 = st.columns([1, 3])
            with col_refresh1:
                if st.button("Actualizar progreso", key="refresh_training"):
                    st.rerun()
            
            # Auto-refrescar automáticamente después de un delay
            import time
            time.sleep(2)
            st.rerun()
        
        elif training_status and training_status.get("status") == "error":
            st.markdown("---")
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.error("Error en el entrenamiento")
            error_msg = training_status.get("error_message", "Error desconocido")
            st.code(error_msg)
            st.info("Revisa la terminal donde ejecutaste Streamlit para más detalles.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Limpiar estado del proceso
            if "training_process" in st.session_state:
                del st.session_state["training_process"]
            if "training_estab" in st.session_state:
                del st.session_state["training_estab"]
        
        elif training_status and training_status.get("status") == "completed":
            st.markdown("---")
            st.markdown('<div class="success-box">Entrenamiento completado exitosamente</div>', unsafe_allow_html=True)
            
            final_val_acc = training_status.get("val_acc", 0.0)
            best_val_acc = training_status.get("best_val_acc", 0.0)
            elapsed_time = training_status.get("elapsed_time", 0.0)
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            
            st.metric("Mejor Accuracy de Validación", f"{best_val_acc:.3f}")
            st.caption(f"Tiempo total: {minutes}m {seconds}s")
            
            # Limpiar estado del proceso
            if "training_process" in st.session_state:
                del st.session_state["training_process"]
            if "training_estab" in st.session_state:
                del st.session_state["training_estab"]
        
        elif st.session_state.get("training_process") or training_status is None:
            # Proceso iniciado pero aún no hay archivo de progreso o no se encontró estado
            st.markdown("---")
            st.markdown('<div class="card-container"><div class="card-header">Estado del Entrenamiento</div>', unsafe_allow_html=True)
            
            process = st.session_state.get("training_process")
            
            if process:
                # Verificar si el proceso aún está corriendo
                if process.poll() is not None:
                    # El proceso terminó
                    return_code = process.returncode
                    if return_code != 0:
                        st.error(f"El entrenamiento terminó con error (código {return_code}).")
                        st.info("Revisa la terminal donde ejecutaste Streamlit para ver los detalles del error.")
                        if "training_process" in st.session_state:
                            del st.session_state["training_process"]
                    else:
                        st.info("El proceso terminó. Verificando resultados...")
                        st.rerun()
                else:
                    st.info("Entrenamiento iniciando... Espera unos segundos para ver el progreso.")
                    st.caption("El proceso está corriendo. La página se actualizará automáticamente cuando haya progreso.")
                    
                    # Verificar si existe el archivo de progreso
                    from src.training_status import get_training_status
                    from pathlib import Path
                    artifacts_dir = Path("artifacts") / training_estab
                    progress_file = artifacts_dir / "training_progress.json"
                    
                    if progress_file.exists():
                        st.caption("Archivo de progreso encontrado. Actualizando...")
                        st.rerun()
                    else:
                        st.caption(f"Esperando creación del archivo de progreso en: {progress_file}")
                        import time
                        time.sleep(2)
                        st.rerun()
            else:
                st.warning("No se detectó un proceso de entrenamiento activo.")
            
            st.markdown('</div>', unsafe_allow_html=True)


def save_prediction_history(selected_estab: str, prediction_data: dict):
    """Guarda una predicción en el historial."""
    from src.dataset_manager import get_artifacts_base_dir
    artifacts_dir = get_artifacts_base_dir() / selected_estab
    history_file = artifacts_dir / "prediction_history.json"
    
    # Cargar historial existente
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding='utf-8'))
        except:
            history = []
    else:
        history = []
    
    # Agregar nueva predicción al inicio
    history.insert(0, prediction_data)
    
    # Mantener solo las últimas 100 predicciones
    history = history[:100]
    
    # Guardar
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding='utf-8')


def load_prediction_history(selected_estab: str) -> list[dict]:
    """Carga el historial de predicciones."""
    from src.dataset_manager import get_artifacts_base_dir
    artifacts_dir = get_artifacts_base_dir() / selected_estab
    history_file = artifacts_dir / "prediction_history.json"
    
    if history_file.exists():
        try:
            return json.loads(history_file.read_text(encoding='utf-8'))
        except:
            return []
    return []


def render_prediction(selected_estab: str):
    """Sección de predicción."""
    st.markdown('<div class="section-header">Predicción de Animales</div>', unsafe_allow_html=True)
    
    # Botones de navegación entre secciones
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    with col_nav1:
        if st.button("Gestión de Dataset", key="nav_to_mgmt_from_prediction", use_container_width=True):
            st.session_state["current_page"] = "Gestión de Datasets"
            st.rerun()
    with col_nav2:
        if st.button("Entrenamiento", key="nav_to_training_from_prediction", use_container_width=True):
            st.session_state["current_page"] = "Entrenamiento"
            st.rerun()
    with col_nav3:
        st.markdown('<div style="text-align: center; padding: 0.5rem; background-color: #e8f5e9; border-radius: 6px; font-weight: 600;">Predicción</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Verificar si hay modelo entrenado para este establecimiento
    from src.dataset_manager import get_artifacts_base_dir
    artifacts_dir = get_artifacts_base_dir() / selected_estab
    model_path = artifacts_dir / "model.pt"
    
    if not model_path.exists():
        st.warning(f"No hay modelo entrenado para '{selected_estab}'. Entrena un modelo primero en la sección de Entrenamiento.")
        return
    
    artifacts_dir = Path("artifacts") / selected_estab
    
    # Cargar modelo
    try:
        from src.infer import load_artifacts
        
        model, classes, tfm, dev = load_artifacts(artifacts_dir)
        st.success(f"Modelo cargado: {selected_estab} ({len(classes)} animales)")
        
    except Exception as e:
        st.error(f"Error al cargar modelo: {e}")
        return
    
    st.markdown("---")
    
    # Tabs para Predicción y Historial
    tab_pred, tab_history = st.tabs(["Nueva Predicción", "Historial"])
    
    with tab_pred:
        render_new_prediction(selected_estab, model, classes, tfm, dev)
    
    with tab_history:
        render_prediction_history_tab(selected_estab)


def render_new_prediction(selected_estab: str, model, classes, tfm, dev):
    """Sección para realizar nuevas predicciones."""
    
    # Parámetros de predicción
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Umbral de confianza", min_value=0.10, max_value=0.95, value=0.70, step=0.01)
    
    with col2:
        st.write("**Configuración de video:**")
        stride = st.number_input("Stride (cada N frames)", min_value=1, max_value=60, value=10, step=1)
        max_frames = st.number_input("Máximo frames a evaluar", min_value=10, max_value=600, value=150, step=10)
    
    # Método de captura
    capture_method_pred = st.radio(
        "Método de captura",
        ["Subir archivo", "Tomar foto con cámara"],
        horizontal=True,
        key="pred_capture_method",
    )
    
    uploaded = None
    camera_photo = None
    
    # Uploader o cámara según el método seleccionado
    if capture_method_pred == "Tomar foto con cámara":
        camera_photo = st.camera_input(
            "Tomar foto para predicción",
            key="pred_camera_input",
            help="Permite tomar una foto directamente desde la cámara de tu dispositivo"
        )
        if camera_photo:
            uploaded = camera_photo
    else:
        uploaded = st.file_uploader(
            "Subir imagen o video",
            type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"],
            key="pred_upload",
        )
    
    if uploaded:
        # Determinar si es video o imagen
        if capture_method_pred == "Tomar foto con cámara":
            is_video = False
            file_name = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            name = uploaded.name.lower() if hasattr(uploaded, 'name') and uploaded.name else "archivo"
            is_video = any(name.endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv"])
            file_name = uploaded.name if hasattr(uploaded, 'name') else "archivo"
        
        if not is_video:
            # Procesar imagen
            img = Image.open(uploaded)
            caption_text = "Foto tomada con cámara" if capture_method_pred == "Tomar foto con cámara" else "Imagen subida"
            st.image(img, caption=caption_text, use_container_width=True)
            
            try:
                from src.infer import predict_image
                
                with st.spinner("Procesando imagen..."):
                    pred = predict_image(model, classes, tfm, dev, img, threshold=float(threshold))
                
                if pred.decision == "unknown":
                    st.error(f"**Resultado: DESCONOCIDA** (confianza: {pred.confidence:.3f})")
                    st.caption(f"Mejor coincidencia: {pred.label}")
                else:
                    st.success(f"**Resultado: {pred.label}** (confianza: {pred.confidence:.3f})")
                
                # Guardar en historial
                prediction_data = {
                    "timestamp": datetime.now().isoformat(),
                    "file_name": file_name,
                    "file_type": "imagen",
                    "result": pred.label,
                    "confidence": pred.confidence,
                    "decision": pred.decision,
                    "threshold": threshold,
                }
                save_prediction_history(selected_estab, prediction_data)
                    
            except Exception as e:
                st.error(f"Error al procesar imagen: {e}")
        
        else:
            # Procesar video
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            
            st.video(tmp_path)
            
            try:
                from src.infer import predict_video
                
                with st.spinner("Procesando video..."):
                    pred = predict_video(
                        model,
                        classes,
                        tfm,
                        dev,
                        tmp_path,
                        threshold=float(threshold),
                        stride=int(stride),
                        max_frames=int(max_frames),
                    )
                
                if pred.decision == "unknown":
                    st.error(f"**Resultado: DESCONOCIDA** (confianza promedio: {pred.confidence:.3f})")
                    st.caption(f"Mejor coincidencia promedio: {pred.label}")
                else:
                    st.success(f"**Resultado: {pred.label}** (confianza promedio: {pred.confidence:.3f})")
                
                # Guardar en historial
                prediction_data = {
                    "timestamp": datetime.now().isoformat(),
                    "file_name": file_name,
                    "file_type": "video",
                    "result": pred.label,
                    "confidence": pred.confidence,
                    "decision": pred.decision,
                    "threshold": threshold,
                    "stride": stride,
                    "max_frames": max_frames,
                }
                save_prediction_history(selected_estab, prediction_data)
                    
            except Exception as e:
                st.error(f"Error al procesar video: {e}")


def render_prediction_history_tab(selected_estab: str):
    """Muestra el historial de predicciones en un tab."""
    history = load_prediction_history(selected_estab)
    
    if not history:
        st.info("No hay predicciones en el historial. Realiza una predicción en la pestaña 'Nueva Predicción'.")
        return
    
    st.markdown('<div class="card-container"><div class="card-header">Historial de Predicciones</div>', unsafe_allow_html=True)
    
    # Mostrar estadísticas generales
    total_predictions = len(history)
    known_count = sum(1 for p in history if p.get("decision") == "known")
    unknown_count = total_predictions - known_count
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Predicciones", total_predictions)
    with col_stat2:
        st.metric("Reconocidas", known_count)
    with col_stat3:
        st.metric("Desconocidas", unknown_count)
    
    st.markdown("---")
    
    # Mostrar lista de predicciones
    st.markdown("**Predicciones recientes:**")
    
    for idx, pred in enumerate(history[:50]):  # Mostrar máximo 50
        timestamp_str = pred.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(timestamp_str)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp_str
        
        file_name = pred.get("file_name", "N/A")
        file_type = pred.get("file_type", "N/A")
        result = pred.get("result", "N/A")
        confidence = pred.get("confidence", 0.0)
        decision = pred.get("decision", "unknown")
        
        # Card para cada predicción
        if decision == "known":
            status_color = "#e8f5e9"
            status_text = "Reconocida"
        else:
            status_color = "#ffebee"
            status_text = "Desconocida"
        
        st.markdown(f'''
        <div style="background-color: {status_color}; padding: 0.75rem; border-radius: 6px; margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{result}</strong> ({confidence:.3f})<br>
                    <small>{file_name} ({file_type})</small><br>
                    <small>{formatted_time}</small>
                </div>
                <div style="background-color: white; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.85rem;">
                    {status_text}
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    if len(history) > 50:
        st.caption(f"... y {len(history) - 50} predicciones más")
    
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
