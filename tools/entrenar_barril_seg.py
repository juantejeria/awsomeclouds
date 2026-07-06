"""
Entrenar YOLO-seg para segmentar el BARRIL de la vaca.

Usa los 72 frames validados del editor de barril como ground truth.
Fine-tunea yolov8s-seg.pt para predecir clase 0 = "barril".

Uso:
    python3 entrenar_barril_seg.py          # preparar dataset + entrenar
    python3 entrenar_barril_seg.py --solo-dataset   # solo preparar dataset
    python3 entrenar_barril_seg.py --out-name barril_seg_v7.pt --run-name barril_seg_v7
"""

import argparse
import cv2
import numpy as np
import json
import shutil
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / 'output_modelos3d_grandes' / '_barril_training'
INDEX_FILE = DATA_DIR / 'frames_index.json'
DATASET_DIR = PROJECT / 'dataset_barril_seg'


def mask_to_yolo_polygon(mask, img_w, img_h, epsilon_factor=0.002):
    """Convierte máscara binaria a polígono normalizado YOLO-seg."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Tomar el contorno más grande
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return None

    # Simplificar polígono
    perimeter = cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)

    if len(contour) < 3:
        return None

    # Normalizar coordenadas [0, 1]
    points = contour.reshape(-1, 2).astype(float)
    points[:, 0] /= img_w
    points[:, 1] /= img_h

    # Clamp
    points = np.clip(points, 0.0, 1.0)

    return points


def preparar_dataset():
    """Convierte los frames validados a formato YOLO-seg."""
    with open(INDEX_FILE) as f:
        frames = json.load(f)

    validated = [f for f in frames if f.get('status') == 'validated']
    print(f"Frames validados: {len(validated)}")

    # Verificar que existan los _barrel.png
    valid_frames = []
    for fr in validated:
        barrel_path = DATA_DIR / f"{fr['id']}_barrel.png"
        img_path = DATA_DIR / fr['img']
        if barrel_path.exists() and img_path.exists():
            valid_frames.append(fr)
        else:
            print(f"  SKIP {fr['id']}: falta barrel o img")

    print(f"Frames con barrel mask: {len(valid_frames)}")

    if len(valid_frames) < 10:
        print("ERROR: muy pocos frames para entrenar")
        sys.exit(1)

    # Split train/val por individuo (no mezclar mismo individuo en train y val)
    individuos = list(set(f['individuo'] for f in valid_frames))
    individuos.sort()

    # 75/25 split por individuo
    if len(individuos) >= 4:
        train_ind, val_ind = train_test_split(individuos, test_size=0.25, random_state=42)
    else:
        # Pocos individuos: split por frame
        train_ind = individuos
        val_ind = individuos

    print(f"Train individuos: {train_ind}")
    print(f"Val individuos: {val_ind}")

    train_frames = [f for f in valid_frames if f['individuo'] in train_ind]
    val_frames = [f for f in valid_frames if f['individuo'] in val_ind]

    # Si split por individuo dejó val vacío, hacer split por frame
    if not val_frames:
        np.random.seed(42)
        np.random.shuffle(valid_frames)
        split = int(0.75 * len(valid_frames))
        train_frames = valid_frames[:split]
        val_frames = valid_frames[split:]

    print(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

    # Crear estructura YOLO
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split_name, split_frames in [('train', train_frames), ('val', val_frames)]:
        img_dir = DATASET_DIR / split_name / 'images'
        lbl_dir = DATASET_DIR / split_name / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        ok = 0
        for fr in split_frames:
            img_path = DATA_DIR / fr['img']
            barrel_path = DATA_DIR / f"{fr['id']}_barrel.png"

            img = cv2.imread(str(img_path))
            barrel = cv2.imread(str(barrel_path), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]

            polygon = mask_to_yolo_polygon(barrel, w, h)
            if polygon is None:
                print(f"  SKIP {fr['id']}: no se pudo extraer polígono")
                continue

            # Copiar imagen
            dst_img = img_dir / f"{fr['id']}.jpg"
            shutil.copy2(str(img_path), str(dst_img))

            # Escribir label: clase 0 (barril) + polígono normalizado
            dst_lbl = lbl_dir / f"{fr['id']}.txt"
            coords = ' '.join(f"{p[0]:.6f} {p[1]:.6f}" for p in polygon)
            dst_lbl.write_text(f"0 {coords}\n")
            ok += 1

        print(f"  {split_name}: {ok} samples escritos")

    # Crear dataset.yaml
    yaml_content = f"""path: {DATASET_DIR}
train: train/images
val: val/images

names:
  0: barril
"""
    yaml_path = DATASET_DIR / 'dataset.yaml'
    yaml_path.write_text(yaml_content)
    print(f"\nDataset listo en: {DATASET_DIR}")
    print(f"Config: {yaml_path}")

    return yaml_path


def entrenar(yaml_path, out_name='barril_seg.pt', run_name='barril_seg'):
    """Fine-tunea yolov8s-seg para segmentar el barril."""
    from ultralytics import YOLO

    # Entrenar desde cero con yolov8n-seg (más chico, menos sesgo de COCO)
    # El fine-tune de yolov8s-seg arrastra el sesgo de detectar "vaca entera"
    base_model = PROJECT / 'yolov8n-seg.pt'
    print(f"\nBase model: {base_model}")
    print(f"Run name: {run_name} | salida: {out_name}")
    print("Iniciando entrenamiento (freeze backbone para reducir sesgo COCO)...\n")

    model = YOLO(str(base_model))

    results = model.train(
        data=str(yaml_path),
        epochs=200,
        imgsz=640,
        batch=8,
        patience=40,
        save=True,
        project=str(PROJECT / 'runs_barril'),
        name=run_name,
        exist_ok=True,
        lr0=0.005,
        # Augmentations agresivas (originalmente calibrado para ~72 imgs)
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10,
        translate=0.15,
        scale=0.4,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.15,
        copy_paste=0.1,
    )

    # Copiar mejor modelo
    best = PROJECT / 'runs_barril' / run_name / 'weights' / 'best.pt'
    dst = PROJECT / 'models' / out_name
    if best.exists():
        shutil.copy2(str(best), str(dst))
        print(f"\n{'='*50}")
        print(f"Modelo guardado: {dst}")
        print(f"{'='*50}")
    else:
        print("WARN: no se encontró best.pt")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solo-dataset', action='store_true', help='Solo prepara dataset, no entrena')
    parser.add_argument('--out-name', default='barril_seg.pt', help='Nombre del .pt final en raíz del proyecto')
    parser.add_argument('--run-name', default='barril_seg', help='Subcarpeta dentro de runs_barril/')
    args = parser.parse_args()

    yaml_path = preparar_dataset()

    if not args.solo_dataset:
        entrenar(yaml_path, out_name=args.out_name, run_name=args.run_name)
    else:
        print("\n--solo-dataset: dataset preparado, no se entrena.")
