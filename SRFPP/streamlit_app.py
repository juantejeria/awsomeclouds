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
    page_title='SRFPP "Sistema de Reconocimiento Facial y Prediccion de Peso"',
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

    .danger-button button {
        background-color: #c0392b !important;
        color: white !important;
        border: none !important;
    }

    .danger-button button:hover {
        background-color: #962d22 !important;
        color: white !important;
    }

    .danger-button button[disabled] {
        background-color: #e0e0e0 !important;
        color: #777 !important;
        cursor: not-allowed !important;
        opacity: 0.7;
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

    .image-list-scroll {
        max-height: 180px; /* ~4 filas visibles */
        overflow-y: auto;
        padding-right: 0.5rem;
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

    /* Ajuste de tamaño del componente de cámara */
    [data-testid="stCameraInput"] {
        max-width: 680px;
        margin: 0 auto;
    }

    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] img {
        width: 100% !important;
        height: auto !important;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Importar módulos de gestión
from src.dataset_manager import (
    create_animal,
    create_establecimiento,
    extract_frames_from_uploaded_video,
    get_animal_dir,
    get_animal_display_name,
    get_dataset_stats,
    get_establecimiento_display_name,
    get_establecimiento_dir,
    list_animales,
    list_establecimientos,
    save_uploaded_file,
    update_establecimiento_display_name,
    validate_dataset_for_training,
)
from src.training_status import append_training_history, load_training_history, save_training_history
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


@st.dialog("Predecir")
def render_prediction_dialog():
    """Popup inicial para tomar foto o subir archivo antes de ir a predicción."""
    st.markdown("Selecciona una opción para continuar:")
    method = st.radio(
        "Método de captura",
        ["Tomar foto con cámara", "Subir archivo o video"],
        horizontal=True,
        key="pred_dialog_method",
    )

    uploaded = None
    if method == "Tomar foto con cámara":
        uploaded = st.camera_input(
            "Tomar foto",
            key="pred_dialog_camera",
            label_visibility="collapsed",
        )
    else:
        uploaded = st.file_uploader(
            "Subir imagen o video",
            type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"],
            key="pred_dialog_upload",
        )

    if uploaded is None:
        return

    if st.button("Predecir", type="primary"):
        st.session_state["pred_initial_upload"] = uploaded
        st.session_state["pred_initial_source"] = "camera" if method == "Tomar foto con cámara" else "upload"
        st.session_state["show_pred_dialog"] = False
        st.session_state["current_page"] = "Predicción"
        st.rerun()


@st.dialog("Editar nombre del establecimiento")
def render_estab_name_dialog(selected_estab: str):
    """Popup para editar el nombre visible del establecimiento."""
    current_display_name = get_establecimiento_display_name(selected_estab)
    new_display_name = st.text_input(
        "Nombre visible",
        value=current_display_name,
        key="estab_display_name_input",
    )
    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Guardar", type="primary", use_container_width=True):
            if new_display_name.strip():
                update_establecimiento_display_name(selected_estab, new_display_name)
                st.session_state["show_estab_name_dialog"] = False
                st.success("Nombre actualizado.")
                st.rerun()
            else:
                st.error("El nombre no puede estar vacío.")
    with col_cancel:
        if st.button("Cancelar", use_container_width=True):
            st.session_state["show_estab_name_dialog"] = False
            st.rerun()


@st.dialog("Eliminar individuo")
def render_delete_animal_dialog(selected_estab: str, selected_animal: str):
    """Popup de confirmación para eliminar un individuo."""
    display_name = get_animal_display_name(selected_estab, selected_animal)
    st.markdown(
        f"Vas a eliminar el individuo **{display_name}** y **todos sus archivos**. "
        "Esta acción no se puede deshacer."
    )
    col_delete, col_cancel = st.columns(2)
    with col_delete:
        if st.button("Eliminar", type="primary", use_container_width=True):
            try:
                animal_dir = get_animal_dir(selected_estab, selected_animal)
                import shutil
                if animal_dir.exists():
                    shutil.rmtree(animal_dir)
                st.success(f"Individuo '{display_name}' eliminado.")
                st.session_state["show_delete_animal_dialog"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Error al eliminar individuo: {e}")
    with col_cancel:
        if st.button("Cancelar", use_container_width=True):
            st.session_state["show_delete_animal_dialog"] = False
            st.rerun()


def main():
    # Título principal
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 0.5rem;">
            <h1 style="margin-bottom: 0.25rem;">SRFPP</h1>
            <h3 style="margin-top: 0;">Sistema de Reconocimiento Facial y Prediccion de Peso</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Botón superior para crear establecimiento (dashboard)
    if st.session_state.get("current_page") is None:
        col_top_spacer, col_top_button = st.columns([6, 1])
        with col_top_button:
            if st.button("Crear nuevo establecimiento", key="new_estab_top", use_container_width=True):
                st.session_state["show_new_estab"] = True
                st.session_state["show_dashboard"] = True
                st.rerun()

    # Obtener establecimiento seleccionado (del cache o de la lista)
    establecimientos = list_establecimientos()
    current_page = st.session_state.get("current_page", None)
    show_dashboard = st.session_state.get("show_dashboard", False)
    
    # Si hay una página activa, obtener el establecimiento del cache
    if current_page:
        # En predicción, no mostramos dataset hasta que el usuario lo elija
        if current_page == "Predicción":
            render_prediction(None)
            return
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
                display_estab = get_establecimiento_display_name(selected_estab)
                st.markdown(f'<h2 style="margin-top: 0; margin-bottom: 0.5rem;">{display_estab}</h2>', unsafe_allow_html=True)
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
            return
    
    # Si no hay página activa, mostrar landing simple o dashboard
    if not current_page and not show_dashboard:
        st.markdown("---")

        col_a, col_b, col_c = st.columns([2, 1, 2])
        with col_b:
            if st.button("+ Identificar vaca", key="start_identify", use_container_width=True, type="primary"):
                st.session_state.pop("last_selected_estab", None)
                st.session_state.pop("main_estab", None)
                st.session_state["show_pred_dialog"] = True
                st.rerun()
            if st.button("Entrar al Dashboard", key="start_dashboard", use_container_width=True):
                st.session_state["show_dashboard"] = True
                st.rerun()
        if st.session_state.get("show_pred_dialog"):
            render_prediction_dialog()
        return

    # Si no hay página activa, mostrar el dashboard principal
    st.session_state["show_dashboard"] = True
    st.markdown("---")
    st.markdown('<div class="section-header">Dashboard Principal</div>', unsafe_allow_html=True)
    
    # Selección de establecimiento
    if establecimientos:
        col_estab_select, col_estab_spacer, col_estab_button = st.columns([1, 1, 1])
        with col_estab_select:
            estab_labels = {
                estab: get_establecimiento_display_name(estab)
                for estab in establecimientos
            }
            # Sin selección por defecto
            selected_estab = st.selectbox(
                "Seleccionar establecimiento",
                options=["Selecciona un establecimiento..."] + establecimientos,
                index=0,
                key="main_estab",
                help="Selecciona el establecimiento con el que quieres trabajar",
                format_func=lambda x: estab_labels.get(x, x),
            )
            
            if selected_estab == "Selecciona un establecimiento...":
                selected_estab = None
                st.session_state.pop("last_selected_estab", None)
            else:
                # Guardar en cache
                st.session_state["last_selected_estab"] = selected_estab
            
        if selected_estab:
            with col_estab_button:
                if st.button("Editar nombre", key="edit_estab_name", use_container_width=True):
                    st.session_state["show_estab_name_dialog"] = True
    else:
        selected_estab = None
        st.info("No hay establecimientos creados. Crea uno nuevo para comenzar.")

    if selected_estab and st.session_state.get("show_estab_name_dialog"):
        render_estab_name_dialog(selected_estab)
    
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
        return
    
    st.markdown("---")
    
    # Mostrar información del establecimiento seleccionado
    if selected_estab:
        st.markdown("---")
        st.markdown("---")
        render_dataset_management(selected_estab, show_nav=False, show_header=False)
        st.markdown("---")
        render_training(selected_estab, show_nav=False)


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
    
    col_animal_select, col_animal_spacer, col_animal_delete = st.columns([1, 1, 1])
    with col_animal_select:
        animal_labels = {
            animal: get_animal_display_name(selected_estab, animal)
            for animal in animales
        }
        selected_animal = st.selectbox(
            "Seleccionar individuo a editar",
            options=["Selecciona un individuo..."] + animales,
            index=0,
            key="edit_animal_select",
            format_func=lambda x: animal_labels.get(x, x),
        )
        if selected_animal == "Selecciona un individuo...":
            selected_animal = None
            st.session_state.pop("show_delete_animal_dialog", None)
    with col_animal_delete:
        st.markdown('<div class="danger-button">', unsafe_allow_html=True)
        if st.button(
            "Eliminar individuo",
            key="delete_animal_trigger",
            use_container_width=True,
            disabled=selected_animal is None,
        ):
            st.session_state["show_delete_animal_dialog"] = True
        st.markdown("</div>", unsafe_allow_html=True)
    
    if selected_animal:
        st.markdown("---")
        st.markdown(f"**Editando: {animal_labels.get(selected_animal, selected_animal)}**")
        
        # Mostrar archivos actuales
        animal_dir = get_animal_dir(selected_estab, selected_animal)
        if animal_dir.exists():
            image_files = sorted([f.name for f in animal_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            video_files = sorted([f.name for f in animal_dir.iterdir() if f.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']])
            
            col_files1, col_files2 = st.columns(2)
            
            with col_files1:
                st.markdown("**Imágenes:**")
                if image_files:
                    image_rows = [{"Imagen": name, "Eliminar": False} for name in image_files]
                    edited_rows = st.data_editor(
                        image_rows,
                        use_container_width=True,
                        hide_index=True,
                        height=180,
                        key=f"edit_images_{selected_animal}",
                    )
                    if st.button("Eliminar seleccionadas", key=f"delete_images_{selected_animal}"):
                        deleted_any = False
                        for row in edited_rows:
                            if row.get("Eliminar"):
                                img_name = row.get("Imagen")
                                if img_name:
                                    (animal_dir / img_name).unlink()
                                    deleted_any = True
                        if deleted_any:
                            st.success("Imágenes eliminadas.")
                            st.rerun()
                        else:
                            st.info("No hay imágenes seleccionadas para eliminar.")
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
        # Reutilizar la lógica de subida de archivos (solo subir nuevos)
        render_upload_files(selected_estab, selected_animal, context="edit", show_existing_files=False)
    
    st.markdown('</div>', unsafe_allow_html=True)

    if selected_animal and st.session_state.get("show_delete_animal_dialog"):
        render_delete_animal_dialog(selected_estab, selected_animal)


def render_upload_files(
    selected_estab: str,
    selected_animal: str,
    context: str = "default",
    show_existing_files: bool = True,
):
    """Renderiza la sección de subida de archivos.
    
    Args:
        selected_estab: Nombre del establecimiento
        selected_animal: Nombre del animal
        context: Contexto de uso para hacer las claves únicas ("edit", "file_mgmt", etc.)
    """
    animal_dir = get_animal_dir(selected_estab, selected_animal)
    # Crear prefijo único para las claves basado en el contexto
    key_prefix = f"{context}_{selected_animal}"
    
    if show_existing_files:
        col_left, col_right = st.columns(2, gap="large")
    else:
        col_right = st.container()
    
    if show_existing_files:
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
        capture_method = st.radio(
            "Método de captura",
            ["Subir archivo", "Tomar foto con cámara"],
            horizontal=True,
            key=f"capture_method_{key_prefix}",
        )
        
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
                "Seleccionar archivos",
                type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"],
                accept_multiple_files=True,
                key=f"upload_files_{key_prefix}",
                help="Puedes subir múltiples archivos a la vez (imágenes o videos)"
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

            is_any_video = False
            for f in uploaded_files:
                name = f.name.lower() if hasattr(f, "name") and f.name else ""
                if any(name.endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv"]):
                    is_any_video = True
                    break

            if is_any_video:
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

                        is_video = False
                        if capture_method != "Tomar foto con cámara":
                            name = uploaded_file.name.lower() if hasattr(uploaded_file, "name") and uploaded_file.name else ""
                            is_video = any(name.endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv"])

                        if not is_video:
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
                                    filter_faces=filter_faces,
                                    require_features=require_features,
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
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_dataset_management(
    selected_estab: str,
    show_nav: bool = True,
    show_header: bool = True,
):
    """Sección de gestión de datasets."""
    if show_header:
        st.markdown('<div class="section-header">Gestión de Dataset</div>', unsafe_allow_html=True)
    
    if show_nav:
        # Botones de navegación entre secciones
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            st.markdown('<div style="text-align: center; padding: 0.5rem; background-color: #e8f5e9; border-radius: 6px; font-weight: 600;">Gestión de Dataset</div>', unsafe_allow_html=True)
        with col_nav2:
            if st.button("Entrenamiento", key="nav_to_training_from_mgmt", use_container_width=True):
                st.session_state["current_page"] = "Entrenamiento"
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
    
    with st.expander("Individuos", expanded=False):
        st.markdown('<div class="card-container"><div class="card-header">Individuos</div>', unsafe_allow_html=True)
        section_choice = st.radio(
            "Sección",
            ["Crear Nuevo Individuo", "Editar Individuo Existente"],
            horizontal=True,
            key="individuo_section_choice",
        )
        if section_choice == "Crear Nuevo Individuo":
            render_create_animal(selected_estab)
        else:
            render_edit_animal(selected_estab)
        st.markdown('</div>', unsafe_allow_html=True)
    
def render_training(selected_estab: str, show_nav: bool = True):
    """Sección de entrenamiento."""
    def _make_dataset_snapshot(stats: dict) -> dict:
        animales = stats.get("animales", {})
        animal_counts = {
            name: {
                "total": data.get("total", 0),
                "imagenes": data.get("imagenes", 0),
                "frames_video": data.get("frames_video", 0),
            }
            for name, data in sorted(animales.items())
        }
        return {
            "total_animales": stats.get("total_animales", 0),
            "min_samples": stats.get("min_samples", 0),
            "max_samples": stats.get("max_samples", 0),
            "balanceado": stats.get("balanceado", False),
            "animales": animal_counts,
        }

    # Validar dataset (sin restricción de mínimo de muestras)
    is_valid, errors = validate_dataset_for_training(selected_estab, min_samples_per_animal=0)
    status_text = "Dataset válido para entrenamiento" if is_valid else "Dataset no válido para entrenamiento"
    status_class = "success-box" if is_valid else "warning-box"
    
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown('<div class="section-header">Entrenamiento de Modelos</div>', unsafe_allow_html=True)
    with col_status:
        st.markdown(f'<div class="{status_class}" style="margin-top: 0.5rem; text-align: center;">{status_text}</div>', unsafe_allow_html=True)
    
    if show_nav:
        # Botones de navegación entre secciones
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("Gestión de Dataset", key="nav_to_mgmt_from_training", use_container_width=True):
                st.session_state["current_page"] = "Gestión de Datasets"
                st.rerun()
        with col_nav2:
            st.markdown('<div style="text-align: center; padding: 0.5rem; background-color: #e8f5e9; border-radius: 6px; font-weight: 600;">Entrenamiento</div>', unsafe_allow_html=True)
    
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
    
    # Estadísticas en layout mejorado
    stats = get_dataset_stats(selected_estab)
    dataset_snapshot = _make_dataset_snapshot(stats)
    
    with st.expander("Estadísticas del Dataset", expanded=False):
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
    
    with st.expander("Parámetros de Entrenamiento", expanded=False):
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
    
    # Sección de Balanceo de Dataset (más destacada)
    with st.expander("Balanceo de Dataset", expanded=False):
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
    
    params_snapshot = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "img_size": int(img_size),
        "val_frac": float(val_frac),
        "balance_dataset": bool(balance_dataset),
        "balance_similarity_threshold": float(balance_similarity_threshold),
    }
    last_trained_dataset_key = f"last_trained_dataset_{selected_estab}"
    last_trained_params_key = f"last_trained_params_{selected_estab}"
    last_trained_dataset = st.session_state.get(last_trained_dataset_key)
    last_trained_params = st.session_state.get(last_trained_params_key)
    dataset_changed = last_trained_dataset is None or last_trained_dataset != dataset_snapshot
    params_changed = last_trained_params is None or last_trained_params != params_snapshot
    training_enabled = is_valid and (dataset_changed or params_changed)

    # Botón de entrenamiento (alineado a la derecha)
    col_train_spacer, col_train_btn = st.columns([3, 1])
    with col_train_btn:
        if st.button("Iniciar entrenamiento", disabled=not training_enabled, type="primary", use_container_width=True):
            with st.spinner("Iniciando entrenamiento..."):
                try:
                    from src.training_ui import train_model
                    
                    st.session_state[last_trained_dataset_key] = dataset_snapshot
                    st.session_state[last_trained_params_key] = params_snapshot
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
            display_name_snapshot = training_status.get("display_name") or get_establecimiento_display_name(training_estab)
            history = load_training_history(training_estab)
            last_entry = history[0] if history else None
            status_signature = (
                float(training_status.get("best_val_acc", 0.0)),
                float(training_status.get("elapsed_time", 0.0)),
                int(training_status.get("total_epochs", 0)),
            )
            last_signature = None
            if last_entry:
                last_signature = (
                    float(last_entry.get("best_val_acc", 0.0)),
                    float(last_entry.get("elapsed_time", 0.0)),
                    int(last_entry.get("total_epochs", 0)),
                )
            if last_signature != status_signature:
                append_training_history(
                    training_estab,
                    {
                        "display_name": display_name_snapshot,
                        "best_val_acc": float(training_status.get("best_val_acc", 0.0)),
                        "val_acc": float(training_status.get("val_acc", 0.0)),
                        "train_acc": float(training_status.get("train_acc", 0.0)),
                        "val_loss": float(training_status.get("val_loss", 0.0)),
                        "train_loss": float(training_status.get("train_loss", 0.0)),
                        "total_epochs": int(training_status.get("total_epochs", 0)),
                        "elapsed_time": float(training_status.get("elapsed_time", 0.0)),
                    },
                )
                history = load_training_history(training_estab)
                last_entry = history[0] if history else None

            st.markdown('<div class="card-container"><div class="card-header">Historial de entrenamientos</div>', unsafe_allow_html=True)
            if last_entry:
                if history:
                    rows = []
                    for entry in history:
                        entry_time = entry.get("completed_at", "")
                        try:
                            dt = datetime.fromisoformat(entry_time)
                            entry_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except Exception:
                            pass
                        elapsed = float(entry.get("elapsed_time", 0.0))
                        minutes = int(elapsed // 60)
                        seconds = int(elapsed % 60)
                        rows.append(
                            {
                                "Fecha": entry_time,
                                "Nombre": entry.get("display_name", training_estab),
                                "Accuracy validación": f"{float(entry.get('best_val_acc', 0.0)):.3f}",
                                "Épocas": int(entry.get("total_epochs", 0)),
                                "Tiempo total": f"{minutes}m {seconds}s",
                                "Dispositivo": entry.get("device", ""),
                                "Otros": entry.get("otros", ""),
                            }
                        )
                    edited_rows = st.data_editor(
                        rows,
                        use_container_width=True,
                        hide_index=True,
                        disabled=["Fecha", "Nombre", "Accuracy validación", "Épocas", "Tiempo total"],
                        key="training_history_editor",
                    )
                    has_changes = False
                    for idx, row in enumerate(edited_rows):
                        if idx >= len(history):
                            continue
                        original_device = history[idx].get("device", "")
                        original_otros = history[idx].get("otros", "")
                        if row.get("Dispositivo", "") != original_device or row.get("Otros", "") != original_otros:
                            has_changes = True
                            break
                    if has_changes:
                        if st.button("Guardar cambios", type="primary"):
                            for idx, row in enumerate(edited_rows):
                                if idx >= len(history):
                                    continue
                                history[idx]["device"] = row.get("Dispositivo", "")
                                history[idx]["otros"] = row.get("Otros", "")
                            save_training_history(training_estab, history)
                            st.success("Historial actualizado.")
                            st.rerun()
            else:
                st.info("Aún no hay entrenamientos registrados.")
            st.markdown("</div>", unsafe_allow_html=True)
            
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


def render_prediction(selected_estab: str | None):
    """Sección de predicción."""
    st.markdown('<div class="section-header">Predicción de Animales</div>', unsafe_allow_html=True)
    
    # Encabezado de sección
    col_nav1, col_nav2 = st.columns([1, 3])
    with col_nav1:
        st.markdown('<div style="text-align: center; padding: 0.5rem; background-color: #e8f5e9; border-radius: 6px; font-weight: 600;">Predicción</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Verificar modelos entrenados disponibles
    trained_estabs = list_trained_establecimientos()
    if not trained_estabs:
        st.warning("No hay modelos entrenados disponibles. Entrena un modelo primero en la sección de Entrenamiento.")
        return
    
    st.markdown("---")
    
    # Tabs para Predicción y Historial
    tab_pred, tab_history = st.tabs(["Nueva Predicción", "Historial"])
    trained_labels = {
        estab: get_establecimiento_display_name(estab)
        for estab in trained_estabs
    }
    
    with tab_pred:
        render_new_prediction(selected_estab, trained_estabs, trained_labels)
    
    with tab_history:
        history_dataset = st.selectbox(
            "Seleccionar dataset para ver historial",
            options=["Selecciona un dataset..."] + trained_estabs,
            index=0,
            key="pred_history_dataset_select",
            format_func=lambda x: trained_labels.get(x, x),
        )
        if history_dataset == "Selecciona un dataset...":
            st.info("Selecciona un dataset para ver el historial.")
            return
        render_prediction_history_tab(history_dataset)


def render_unknown_actions(
    selected_estab: str,
    uploaded,
    file_name: str,
    is_video: bool,
    default_stride: int,
    default_max_frames: int,
):
    """Opciones cuando la predicción es desconocida."""
    st.markdown("---")
    st.markdown(
        '<div class="warning-box"><strong>No se encontró coincidencia en este dataset.</strong> '
        'Puedes intentar otra captura o crear un nuevo animal.</div>',
        unsafe_allow_html=True,
    )

    col_action1, col_action2 = st.columns(2)
    with col_action1:
        if st.button("Subir otra foto o video", key="unknown_retry_upload", use_container_width=True):
            st.session_state["pred_upload_key"] = st.session_state.get("pred_upload_key", 0) + 1
            st.session_state["pred_camera_key"] = st.session_state.get("pred_camera_key", 0) + 1
            st.rerun()

    with col_action2:
        if st.button("Ir a Gestión de Dataset", key="unknown_go_dataset", use_container_width=True):
            st.session_state["current_page"] = "Gestión de Datasets"
            st.rerun()

    with st.expander("Crear nuevo animal en este dataset", expanded=True):
        new_animal_name = st.text_input(
            "Nombre del nuevo animal",
            placeholder="Ej: vaca_011",
            key="unknown_new_animal_name",
        )
        save_original = st.checkbox(
            "Guardar este archivo en el nuevo animal",
            value=True,
            key="unknown_save_original",
        )

        extract_frames = False
        stride = default_stride
        max_frames = default_max_frames
        filter_faces = False
        require_features = None

        if is_video:
            st.markdown("**Opciones para video:**")
            extract_frames = st.checkbox(
                "Extraer frames automáticamente",
                value=True,
                key="unknown_extract_frames",
            )
            if extract_frames:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    stride = st.number_input(
                        "Stride (cada N frames)",
                        min_value=1,
                        max_value=60,
                        value=default_stride,
                        key="unknown_stride_input",
                    )
                with col_s2:
                    max_frames = st.number_input(
                        "Máximo frames por video",
                        min_value=10,
                        max_value=200,
                        value=default_max_frames,
                        key="unknown_max_frames_input",
                    )

                st.markdown("**Filtrado de frames:**")
                filter_faces = st.checkbox(
                    "Solo guardar frames con animal detectado",
                    value=False,
                    key="unknown_filter_faces_check",
                    help="Si está habilitado, solo se guardarán frames donde YOLO detecte un animal.",
                )

                if filter_faces:
                    feature_options = st.multiselect(
                        "Tipo de detección requerida (opcional)",
                        options=["cow", "animal"],
                        default=[],
                        key="unknown_require_features_select",
                    )
                    if feature_options:
                        require_features = feature_options

        if st.button("Crear animal", key="unknown_create_animal", type="primary"):
            if not new_animal_name.strip():
                st.error("El nombre del animal no puede estar vacío.")
                return

            try:
                animal_dir = create_animal(selected_estab, new_animal_name.strip())
                saved_count = 0

                if save_original:
                    safe_name = Path(file_name).name
                    dest_path = animal_dir / safe_name
                    save_uploaded_file(uploaded, dest_path)
                    saved_count += 1

                    if is_video and extract_frames:
                        frames_extracted = extract_frames_from_uploaded_video(
                            dest_path,
                            animal_dir,
                            stride=int(stride),
                            max_frames=int(max_frames),
                            filter_faces=filter_faces,
                            require_features=require_features,
                        )
                        saved_count += frames_extracted

                st.success(
                    f"Animal '{new_animal_name.strip()}' creado. "
                    f"Archivos guardados: {saved_count}."
                )
            except Exception as e:
                st.error(f"Error al crear animal: {e}")


def render_new_prediction(
    selected_estab: str | None,
    trained_estabs: list[str],
    trained_labels: dict[str, str],
):
    """Sección para realizar nuevas predicciones."""
    
    col_left, col_right = st.columns([1, 1], gap="large")

    initial_upload = st.session_state.get("pred_initial_upload")
    initial_source = st.session_state.get("pred_initial_source", "upload")
    initial_index = 0 if initial_source == "camera" else 1

    with col_left:
        threshold = st.slider("Umbral de confianza", min_value=0.10, max_value=0.95, value=0.70, step=0.01)
        st.write("**Configuración de video:**")
        stride = st.number_input("Stride (cada N frames)", min_value=1, max_value=60, value=10, step=1)
        max_frames = st.number_input("Máximo frames a evaluar", min_value=10, max_value=600, value=150, step=10)

        capture_method_pred = st.radio(
            "Método de captura",
            ["Tomar foto con cámara", "Subir archivo"],
            horizontal=True,
            key="pred_capture_method",
            index=initial_index,
            disabled=initial_upload is not None,
        )

    uploaded = None
    camera_photo = None

    # Uploader o cámara según el método seleccionado
    upload_key = st.session_state.get("pred_upload_key", 0)
    camera_key = st.session_state.get("pred_camera_key", 0)

    with col_right:
        if initial_upload is not None:
            uploaded = initial_upload
            if st.button("Cambiar archivo", key="pred_change_initial", use_container_width=True):
                st.session_state.pop("pred_initial_upload", None)
                st.session_state.pop("pred_initial_source", None)
                st.rerun()
        elif capture_method_pred == "Tomar foto con cámara":
            camera_photo = st.camera_input(
                "Predecir",
                key=f"pred_camera_input_{camera_key}",
                help="Permite tomar una foto directamente desde la cámara de tu dispositivo"
            )
            if camera_photo:
                uploaded = camera_photo
        else:
            uploaded = st.file_uploader(
                "Subir imagen o video",
                type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"],
                key=f"pred_upload_{upload_key}",
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

        img = None
        tmp_path = None
        with col_right:
            # Mostrar preview antes de seleccionar dataset
            if not is_video:
                img = Image.open(uploaded)
                caption_text = "Foto tomada con cámara" if capture_method_pred == "Tomar foto con cámara" else "Imagen subida"
                if capture_method_pred == "Tomar foto con cámara":
                    st.image(img, caption=caption_text, width=360)
                else:
                    st.image(img, caption=caption_text, use_container_width=True)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                st.video(tmp_path)

            # Seleccionar dataset para validar después de cargar el archivo
            selected_dataset = st.selectbox(
                "Seleccionar dataset para validar",
                options=["Selecciona un dataset..."] + trained_estabs,
                index=0,
                key="pred_dataset_select",
                format_func=lambda x: trained_labels.get(x, x),
            )
            st.session_state["pred_dataset"] = selected_dataset

            if selected_dataset == "Selecciona un dataset...":
                st.info("Selecciona un dataset para continuar con la validación.")
                return

            # Cargar modelo según dataset seleccionado
            artifacts_dir = Path("artifacts") / selected_dataset
            try:
                from src.infer import load_artifacts
                model, classes, tfm, dev = load_artifacts(artifacts_dir)
            except Exception as e:
                st.error(f"Error al cargar modelo de '{selected_dataset}': {e}")
                return

            if not is_video:
                try:
                    from src.infer import predict_image

                    with st.spinner("Procesando imagen..."):
                        pred = predict_image(model, classes, tfm, dev, img, threshold=float(threshold))

                    if pred.decision == "unknown":
                        st.error(f"**Resultado: DESCONOCIDA** (confianza: {pred.confidence:.3f})")
                        st.caption(f"Mejor coincidencia: {pred.label}")
                        render_unknown_actions(
                            selected_dataset,
                            uploaded,
                            file_name,
                            is_video=False,
                            default_stride=int(stride),
                            default_max_frames=int(max_frames),
                        )
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
                    save_prediction_history(selected_dataset, prediction_data)

                except Exception as e:
                    st.error(f"Error al procesar imagen: {e}")

            else:
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
                        render_unknown_actions(
                            selected_dataset,
                            uploaded,
                            file_name,
                            is_video=True,
                            default_stride=int(stride),
                            default_max_frames=int(max_frames),
                        )
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
                    save_prediction_history(selected_dataset, prediction_data)

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
