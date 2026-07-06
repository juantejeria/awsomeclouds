"""
Entrenar YOLO-pose para detectar el PUNTO DE LA CRUZ de la vaca.

Modelo INDEPENDIENTE del barril: usa únicamente el punto `cruz` anotado en el
editor (frames_index.json) como un único keypoint. La caja del objeto se toma de
la máscara de silueta (campo 'mask'), expandida para contener siempre la cruz.

Clase 0 = "vaca", 1 keypoint = "cruz".

Uso:
    python3 entrenar_cruz_pose.py                  # preparar dataset + entrenar
    python3 entrenar_cruz_pose.py --solo-dataset   # solo preparar dataset
    python3 entrenar_cruz_pose.py --out-name cruz_pose_v2.pt --run-name cruz_pose_v2
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
DATASET_DIR = PROJECT / 'dataset_cruz_pose'

# Padding (fracción del tamaño de la caja) al expandir la silueta para asegurar
# que la cruz quede dentro de la caja del objeto.
BOX_PAD = 0.04


def bbox_para_frame(mask, cruz, w, h):
    """Devuelve (x0, y0, x1, y1) en píxeles: bbox de la silueta unido al punto
    de cruz, con un pequeño padding y recortado a la imagen. Fallback: imagen
    completa si no hay silueta válida."""
    x0, y0, x1, y1 = 0, 0, w, h
    if mask is not None:
        ys, xs = (mask > 128).nonzero()
        if len(xs) > 0:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())

    # Unir el punto de cruz (por si cae fuera de la silueta)
    cx, cy = float(cruz['x']), float(cruz['y'])
    x0 = min(x0, cx); x1 = max(x1, cx)
    y0 = min(y0, cy); y1 = max(y1, cy)

    # Padding proporcional
    pw, ph = (x1 - x0) * BOX_PAD, (y1 - y0) * BOX_PAD
    x0 -= pw; x1 += pw; y0 -= ph; y1 += ph

    # Clamp a la imagen
    x0 = max(0.0, x0); y0 = max(0.0, y0)
    x1 = min(float(w), x1); y1 = min(float(h), y1)
    return x0, y0, x1, y1


def preparar_dataset():
    """Convierte los frames validados con cruz a formato YOLO-pose (1 keypoint)."""
    with open(INDEX_FILE) as f:
        frames = json.load(f)

    valid = []
    for fr in frames:
        if fr.get('status') != 'validated':
            continue
        if not fr.get('cruz'):
            continue
        img_path = DATA_DIR / fr['img']
        if img_path.exists():
            valid.append(fr)
        else:
            print(f"  SKIP {fr['id']}: falta imagen")

    print(f"Frames validados con cruz: {len(valid)}")
    if len(valid) < 10:
        print("ERROR: muy pocos frames para entrenar")
        sys.exit(1)

    # Split por individuo (no mezclar mismo individuo en train y val)
    individuos = sorted(set(f['individuo'] for f in valid))
    if len(individuos) >= 4:
        train_ind, val_ind = train_test_split(individuos, test_size=0.2, random_state=42)
    else:
        train_ind = val_ind = individuos
    train_ind, val_ind = set(train_ind), set(val_ind)

    train_frames = [f for f in valid if f['individuo'] in train_ind]
    val_frames = [f for f in valid if f['individuo'] in val_ind]
    if not val_frames:
        np.random.seed(42)
        np.random.shuffle(valid)
        split = int(0.8 * len(valid))
        train_frames, val_frames = valid[:split], valid[split:]

    print(f"Train: {len(train_frames)} frames / {len(train_ind)} individuos")
    print(f"Val:   {len(val_frames)} frames / {len(val_ind)} individuos")

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
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  SKIP {fr['id']}: no se pudo leer img")
                continue
            h, w = img.shape[:2]

            mask = cv2.imread(str(DATA_DIR / fr['mask']), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            cruz = fr['cruz']
            kx = min(max(float(cruz['x']), 0.0), w - 1)
            ky = min(max(float(cruz['y']), 0.0), h - 1)

            x0, y0, x1, y1 = bbox_para_frame(mask, cruz, w, h)
            bw, bh = (x1 - x0), (y1 - y0)
            if bw < 2 or bh < 2:
                print(f"  SKIP {fr['id']}: caja degenerada")
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

            # Normalizar
            cx_n, cy_n = cx / w, cy / h
            bw_n, bh_n = bw / w, bh / h
            kx_n, ky_n = kx / w, ky / h

            # Copiar imagen (extensión original)
            ext = Path(fr['img']).suffix
            dst_img = img_dir / f"{fr['id']}{ext}"
            shutil.copy2(str(img_path), str(dst_img))

            # Label YOLO-pose: cls cx cy w h  kx ky v   (v=2 visible)
            dst_lbl = lbl_dir / f"{fr['id']}.txt"
            dst_lbl.write_text(
                f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f} "
                f"{kx_n:.6f} {ky_n:.6f} 2\n"
            )
            ok += 1

        print(f"  {split_name}: {ok} samples escritos")

    yaml_content = f"""path: {DATASET_DIR}
train: train/images
val: val/images

kpt_shape: [1, 3]
flip_idx: [0]

names:
  0: vaca
"""
    yaml_path = DATASET_DIR / 'dataset.yaml'
    yaml_path.write_text(yaml_content)
    print(f"\nDataset listo en: {DATASET_DIR}")
    print(f"Config: {yaml_path}")
    return yaml_path


def entrenar(yaml_path, out_name='cruz_pose.pt', run_name='cruz_pose'):
    """Fine-tunea yolov8-pose para detectar la cruz (1 keypoint)."""
    from ultralytics import YOLO

    base_model = 'yolov8n-pose.pt'  # ultralytics lo descarga si no está local
    print(f"\nBase model: {base_model}")
    print(f"Run name: {run_name} | salida: {out_name}")
    print("Iniciando entrenamiento YOLO-pose (1 keypoint = cruz)...\n")

    model = YOLO(base_model)

    results = model.train(
        data=str(yaml_path),
        epochs=200,
        imgsz=640,
        batch=8,
        patience=40,
        save=True,
        project=str(PROJECT / 'runs_cruz'),
        name=run_name,
        exist_ok=True,
        lr0=0.005,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10,
        translate=0.15,
        scale=0.4,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.15,
    )

    best = PROJECT / 'runs_cruz' / run_name / 'weights' / 'best.pt'
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
    parser.add_argument('--out-name', default='cruz_pose.pt', help='Nombre del .pt final en raíz del proyecto')
    parser.add_argument('--run-name', default='cruz_pose', help='Subcarpeta dentro de runs_cruz/')
    args = parser.parse_args()

    yaml_path = preparar_dataset()

    if not args.solo_dataset:
        entrenar(yaml_path, out_name=args.out_name, run_name=args.run_name)
    else:
        print("\n--solo-dataset: dataset preparado, no se entrena.")
