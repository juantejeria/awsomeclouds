"""
Video Identifier UI - Identificacion de individuos en video.

Usa el mismo pipeline del sistema original (predict_video_multi_animal con
ByteTrack + face model + body validation) y genera un video anotado con
bounding boxes, IDs y resumen final.
"""

import os
import tempfile
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import streamlit as st
import torch
import json
from collections import defaultdict
from torchvision import transforms

from src.model import build_model
from src.infer import predict_video_multi_animal, load_centroids

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


# ── Helpers ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_recognition_model(artifacts_path: str):
    """Carga modelo + clases + centroides + transform."""
    artifacts = Path(artifacts_path)
    classes = json.loads((artifacts / "classes.json").read_text())
    config = json.loads((artifacts / "config.json").read_text())

    img_size = config.get("img_size", 224)
    dropout = config.get("dropout", 0.25)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(num_classes=len(classes), pretrained=False, dropout=dropout)
    model.load_state_dict(
        torch.load(str(artifacts / "model.pt"), map_location=device)
    )
    model.eval().to(device)

    centroids = load_centroids(artifacts)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return model, classes, centroids, device, tfm


def draw_annotations(frame_bgr, detections):
    """Dibuja bounding boxes y labels sobre el frame.
    detections: list of FrameDetection from infer.py
    """
    frame = frame_bgr.copy()

    palette = [
        (0, 255, 0), (255, 165, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (128, 255, 0), (0, 128, 255), (255, 128, 0),
        (128, 0, 255), (0, 255, 128), (255, 255, 0), (0, 128, 128),
    ]

    for det in detections:
        track_id = det.track_id
        label = det.label
        conf = det.recognition_confidence
        x1, y1, x2, y2 = [int(v) for v in det.bbox_padded]

        color = palette[track_id % len(palette)] if track_id >= 0 else (128, 128, 128)

        # Bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label text
        if conf >= 0.5:
            display = f"#{track_id} {label} ({conf:.0%})"
        else:
            display = f"#{track_id} ? ({conf:.0%})"

        (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, display, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def generate_annotated_video(video_path, multi_result, progress_bar, status_text):
    """Genera video con anotaciones a partir de los resultados del pipeline original."""
    # Indexar detecciones por frame_idx
    dets_by_frame = defaultdict(list)
    for tr in multi_result.tracks:
        for fd in tr.frame_detections:
            fd_copy = fd
            # Asegurar que tiene el track_id correcto
            dets_by_frame[fd.frame_idx].append(fd_copy)
    # Tambien incluir detecciones de tracks descartados (all_frame_detections)
    # pero dar prioridad a los tracks validos
    frames_with_tracks = set(dets_by_frame.keys())
    for fd in multi_result.all_frame_detections:
        if fd.frame_idx not in frames_with_tracks:
            dets_by_frame[fd.frame_idx].append(fd)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(tmp_out.name, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in dets_by_frame:
            frame = draw_annotations(frame, dets_by_frame[frame_idx])

        out.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            pct = frame_idx / max(total_frames, 1)
            progress_bar.progress(min(pct, 1.0))
            status_text.text(f"Generando video anotado: frame {frame_idx}/{total_frames}...")

    cap.release()
    out.release()
    progress_bar.progress(1.0)
    status_text.text("Video anotado generado.")

    return tmp_out.name


# ── Streamlit UI ─────────────────────────────────────────────────────────

def get_available_datasets():
    """Lista datasets entrenados (con model.pt)."""
    datasets = []
    if not ARTIFACTS_DIR.exists():
        return datasets
    for d in sorted(ARTIFACTS_DIR.iterdir()):
        if d.is_dir() and (d / "model.pt").exists() and (d / "classes.json").exists():
            datasets.append(d.name)
    return datasets


def main():
    st.set_page_config(page_title="Video Identifier", layout="wide")
    st.title("Video Identifier")
    st.markdown("Sube un video y mira como se identifica cada individuo.")

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuracion")

        datasets = get_available_datasets()
        if not datasets:
            st.error("No hay datasets entrenados en artifacts/")
            return

        dataset = st.selectbox("Dataset entrenado", datasets)

        st.divider()
        stride = st.slider("Procesar 1 de cada N frames", 1, 30, 10)
        max_frames = None  # Procesar todos los frames
        threshold = st.slider("Umbral confianza (softmax)", 0.3, 0.95, 0.60, 0.05)
        sim_threshold = st.slider("Umbral similitud (coseno)", 0.2, 0.8, 0.45, 0.05)
        min_track = st.slider("Min detecciones por track", 1, 15, 3)

        st.divider()
        classes = json.loads(
            (ARTIFACTS_DIR / dataset / "classes.json").read_text()
        )
        st.markdown(f"**Dataset:** {dataset}")
        st.markdown(f"**Individuos entrenados:** {len(classes)}")
        with st.expander("Ver clases"):
            for c in classes:
                st.markdown(f"- {c}")

    # ── Main ─────────────────────────────────────────────────────────
    input_mode = st.radio(
        "Origen del video",
        ["Subir archivo", "Ruta local"],
        horizontal=True,
    )

    video_path = None
    is_tmp = False

    if input_mode == "Subir archivo":
        uploaded = st.file_uploader(
            "Subi un video", type=["mp4", "mov", "avi", "mkv"],
        )
        if uploaded is not None:
            tmp_input = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_input.write(uploaded.read())
            tmp_input.flush()
            video_path = tmp_input.name
            is_tmp = True
    else:
        local_path = st.text_input(
            "Ruta al video",
            placeholder="/ruta/al/video.mp4",
        )
        if local_path and Path(local_path).exists():
            video_path = local_path
        elif local_path:
            st.error(f"No se encontro: {local_path}")

    if video_path is None:
        st.info("Subi un video o ingresa una ruta local para comenzar.")
        return

    with st.expander("Video original", expanded=False):
        st.video(video_path)

    if st.button("Procesar video", type="primary", use_container_width=True):
        st.divider()

        # Cargar modelo
        with st.spinner("Cargando modelo..."):
            model, classes, centroids, device, tfm = load_recognition_model(
                str(ARTIFACTS_DIR / dataset)
            )
        st.success(f"Modelo cargado: {len(classes)} clases, device={device}")

        # ── Paso 1: Pipeline original (ByteTrack + face + body validation) ──
        st.subheader("Paso 1: Deteccion y tracking")
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def progress_cb(current, total):
            pct = current / max(total, 1)
            progress_bar.progress(min(pct, 1.0))
            status_text.text(f"Analizando: frame {current}/{total}")

        multi_result = predict_video_multi_animal(
            model=model,
            classes=classes,
            tfm=tfm,
            dev=device,
            video_path=str(video_path),
            threshold=threshold,
            similarity_threshold=sim_threshold,
            stride=stride,
            max_frames=max_frames,
            yolo_min_confidence=0.25,
            min_track_length=min_track,
            centroids=centroids,
            progress_callback=progress_cb,
        )

        progress_bar.progress(1.0)
        status_text.text(f"Analisis completo: {len(multi_result.tracks)} tracks encontrados")

        # ── Resumen ──────────────────────────────────────────────────
        st.subheader("Resumen de identificacion")

        tracks = multi_result.tracks
        known = [t for t in tracks if t.prediction.decision == "known"]
        unknown = [t for t in tracks if t.prediction.decision == "unknown"]
        known_labels = set(t.prediction.label for t in known)
        missing_labels = set(classes) - known_labels

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total individuos", len(tracks))
        col2.metric("Identificados", len(known))
        col3.metric("Desconocidos", len(unknown))
        col4.metric("No aparecen", len(missing_labels))

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Frames extraidos", multi_result.total_frames_extracted)
        col_b.metric("Con detecciones", multi_result.frames_with_detections)
        col_c.metric("Tracks descartados", multi_result.noise_tracks_discarded)

        # ── Individuos identificados ─────────────────────────────────
        if known:
            st.subheader("Individuos identificados")
            for t in sorted(known, key=lambda x: x.prediction.label):
                face_pct = (t.frames_with_face / max(len(t.frame_detections), 1)) * 100
                st.markdown(
                    f"**{t.prediction.label}** (Track #{t.track_id}) — "
                    f"Confianza: {t.prediction.confidence:.0%} | "
                    f"Similitud: {t.prediction.cosine_similarity:.3f} | "
                    f"Acuerdo: {t.agreement_ratio:.0%} | "
                    f"Detecciones: {len(t.frame_detections)} "
                    f"(rostro: {t.frames_with_face}, cuerpo: {t.frames_with_body_only}) | "
                    f"Rostro: {face_pct:.0f}%"
                )

        # ── Desconocidos ─────────────────────────────────────────────
        if unknown:
            st.subheader("Individuos desconocidos")
            for t in unknown:
                st.markdown(
                    f"**Track #{t.track_id}** — "
                    f"Mejor match: {t.prediction.label} | "
                    f"Confianza: {t.prediction.confidence:.0%} | "
                    f"Similitud: {t.prediction.cosine_similarity:.3f} | "
                    f"Detecciones: {len(t.frame_detections)}"
                )

        # ── No encontrados ───────────────────────────────────────────
        if missing_labels:
            st.subheader("No aparecen en el video")
            st.markdown(", ".join(f"**{l}**" for l in sorted(missing_labels)))

        # Cleanup
        if is_tmp:
            try:
                os.unlink(video_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
