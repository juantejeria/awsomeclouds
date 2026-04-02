"""
Compara resultados del sistema original (infer.predict_video_multi_animal)
vs el nuevo Video Identifier, usando el mismo video y dataset.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import json

from src.infer import predict_video_multi_animal, load_centroids
from src.model import build_model, extract_embeddings
from src.face_detection import detect_cow_face_boxes, detect_animal_boxes
from src.tracker import MultiObjectTracker

# Reusar funciones del nuevo sistema
from video_identifier import (
    detect_and_recognize_frame,
    recognize_crop,
    get_transform,
    compile_track_results,
)

import cv2

ARTIFACTS = Path("artifacts/Entrga_Reconocimiento")
THRESHOLD = 0.60
SIM_THRESHOLD = 0.45
STRIDE = 3


def run_original_system(video_path):
    """Corre el sistema original de infer.py"""
    print("=" * 60)
    print("SISTEMA ORIGINAL (predict_video_multi_animal)")
    print("=" * 60)

    # Cargar modelo
    classes = json.loads((ARTIFACTS / "classes.json").read_text())
    config = json.loads((ARTIFACTS / "config.json").read_text())
    img_size = config.get("img_size", 224)
    dropout = config.get("dropout", 0.25)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(num_classes=len(classes), pretrained=False, dropout=dropout)
    model.load_state_dict(torch.load(str(ARTIFACTS / "model.pt"), map_location=device))
    model.eval().to(device)
    centroids = load_centroids(ARTIFACTS)

    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    result = predict_video_multi_animal(
        model=model,
        classes=classes,
        tfm=tfm,
        dev=device,
        video_path=str(video_path),
        threshold=THRESHOLD,
        similarity_threshold=SIM_THRESHOLD,
        stride=STRIDE,
        max_frames=300,
        yolo_min_confidence=0.25,
        min_track_length=3,
        centroids=centroids,
    )

    print(f"\nTotal frames extraidos: {result.total_frames_extracted}")
    print(f"Frames con detecciones: {result.frames_with_detections}")
    print(f"Tracks descartados (ruido): {result.noise_tracks_discarded}")
    print(f"Tracks validos: {len(result.tracks)}")

    for tr in result.tracks:
        det_count = len(tr.frame_detections)
        face_count = tr.frames_with_face
        body_count = tr.frames_with_body_only
        print(f"\n  Track #{tr.track_id}: {tr.prediction.label} "
              f"({tr.prediction.decision}) "
              f"conf={tr.prediction.confidence:.2%} "
              f"sim={tr.prediction.cosine_similarity:.3f} "
              f"acuerdo={tr.agreement_ratio:.0%} "
              f"dets={det_count} (face={face_count}, body={body_count})")

    return result


def run_new_system(video_path):
    """Corre el nuevo sistema (video_identifier)"""
    print("\n" + "=" * 60)
    print("SISTEMA NUEVO (video_identifier)")
    print("=" * 60)

    # Cargar modelo
    classes = json.loads((ARTIFACTS / "classes.json").read_text())
    config = json.loads((ARTIFACTS / "config.json").read_text())
    img_size = config.get("img_size", 224)
    dropout = config.get("dropout", 0.25)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(num_classes=len(classes), pretrained=False, dropout=dropout)
    model.load_state_dict(torch.load(str(ARTIFACTS / "model.pt"), map_location=device))
    model.eval().to(device)
    centroids = load_centroids(ARTIFACTS)
    transform = get_transform(img_size)

    tracker = MultiObjectTracker(max_center_dist=0.6, max_age=15, min_track_length=3)
    track_labels = defaultdict(list)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    frames_with_det = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % STRIDE == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detect_and_recognize_frame(
                rgb, model, classes, centroids, device, transform,
                THRESHOLD, SIM_THRESHOLD,
            )

            if detections:
                frames_with_det += 1
                track_assignments = tracker.update(frame_idx, detections)
                for track_id, det in track_assignments:
                    track_labels[track_id].append({
                        "label": det["label"],
                        "confidence": det["confidence"],
                        "decision": det["decision"],
                        "cos_sim": det["cos_sim"],
                    })
            else:
                tracker.update(frame_idx, [])

        frame_idx += 1

    cap.release()

    track_results = compile_track_results(track_labels, classes, THRESHOLD, SIM_THRESHOLD)

    print(f"\nTotal frames: {total_frames}")
    print(f"Frames con detecciones: {frames_with_det}")
    print(f"Tracks validos: {len(track_results)}")

    for tr in track_results:
        ids_str = ", ".join(f"#{tid}" for tid in tr["track_ids"])
        print(f"\n  Track {ids_str}: {tr['label']} "
              f"({tr['decision']}) "
              f"conf={tr['avg_confidence']:.2%} "
              f"sim={tr['avg_cosine_sim']:.3f} "
              f"acuerdo={tr['agreement']:.0%} "
              f"dets={tr['total_detections']}")

    return track_results


def main():
    videos = [
        Path("data/Entrga_Reconocimiento/vaca_1/vaca1.MOV"),
        Path("data/Entrga_Reconocimiento/vaca_3/vaca3.MOV"),
    ]

    for video in videos:
        if not video.exists():
            print(f"SKIP: {video} no existe")
            continue

        print("\n" + "#" * 70)
        print(f"VIDEO: {video}")
        print("#" * 70)

        orig = run_original_system(video)
        new = run_new_system(video)

        # Comparar
        print("\n" + "=" * 60)
        print("COMPARACION")
        print("=" * 60)

        orig_labels = set()
        for tr in orig.tracks:
            orig_labels.add((tr.prediction.label, tr.prediction.decision))

        new_labels = set()
        for tr in new:
            new_labels.add((tr["label"], tr["decision"]))

        print(f"Original: {len(orig.tracks)} tracks -> {orig_labels}")
        print(f"Nuevo:    {len(new)} tracks -> {new_labels}")

        common = orig_labels & new_labels
        only_orig = orig_labels - new_labels
        only_new = new_labels - orig_labels
        if common:
            print(f"Coinciden: {common}")
        if only_orig:
            print(f"Solo en original: {only_orig}")
        if only_new:
            print(f"Solo en nuevo: {only_new}")


if __name__ == "__main__":
    main()
