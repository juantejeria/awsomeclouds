"""Para cada video en checkpoints/22abril/ samplea a 15 fps y stage-ea TODOS
los frames donde la vaca esta COMPLETA (no entrando/saliendo del frame, no muy
chica, silueta razonable) en _barril_training/.

Mascara inicial: silueta_seg sobre el FRAME COMPLETO + postprocesar_barril
(recorta patas usando perfil de anchos). El editor manual recorta cabeza/cuello
y cola.

Filtros para "vaca completa":
  - bbox de la vaca a >= EDGE_MARGIN px de los bordes del frame
  - bbox dimensiones >= MIN_BBOX_W x MIN_BBOX_H
  - silueta cubre >= MIN_SILUETA_BBOX_RATIO del bbox de la vaca
  - silueta cobertura X >= MIN_SILUETA_COB_X% del bbox

Idempotente: al re-correr borra entradas previas con source=22abril.
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from ultralytics import YOLO

from generar_modelos3d_grandes import postprocesar_barril


PROJECT = Path(__file__).parent
VIDEOS_DIR = PROJECT / "checkpoints" / "22abril"
BARRIL_DIR = PROJECT / "output_modelos3d_grandes" / "_barril_training"
INDEX_FILE = BARRIL_DIR / "frames_index.json"

SAMPLING_FPS = 15.0
COW_CONF = 0.2

# Filtros de "vaca completa"
EDGE_MARGIN = 10           # px de margen al borde del frame
MIN_BBOX_W = 200           # ancho minimo del bbox de la vaca (px del frame original)
MIN_BBOX_H = 130           # alto minimo
MIN_SILUETA_BBOX_RATIO = 0.45  # area silueta / area bbox vaca
MIN_SILUETA_COB_X = 80.0   # cobertura X de la silueta dentro del bbox (%)
PAD_RATIO = 0.30           # padding del crop relativo al lado mayor del bbox


def load_index():
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            return json.load(f)
    return []


def save_index(data):
    with open(INDEX_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def silueta_full_frame(frame, silueta_model, bbox, conf=0.25):
    """Silueta_seg sobre el FRAME COMPLETO. Devuelve mascara binaria 0/255 del
    tamano del frame, eligiendo la instancia con mayor IoU con bbox."""
    H, W = frame.shape[:2]
    r = silueta_model(frame, conf=conf, verbose=False)
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        return None
    masks = r[0].masks.data.cpu().numpy()
    bx1, by1, bx2, by2 = bbox
    bm = np.zeros((H, W), dtype=np.uint8)
    bm[by1:by2, bx1:bx2] = 1
    best, best_iou = None, -1.0
    for m in masks:
        m_b = (cv2.resize(m, (W, H)) > 0.5).astype(np.uint8)
        inter = int(np.sum(m_b & bm))
        union = int(np.sum(m_b | bm))
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou, best = iou, m_b
    if best is None or best_iou < 0.05:
        return None
    return (best * 255).astype(np.uint8)


def vaca_completa(bbox, frame_shape, sil_full):
    """Aplica los filtros de calidad. Devuelve (ok: bool, motivo: str)."""
    H, W = frame_shape[:2]
    bx1, by1, bx2, by2 = bbox
    bw, bh = bx2 - bx1, by2 - by1

    # Bordes
    if bx1 < EDGE_MARGIN or by1 < EDGE_MARGIN or bx2 > W - EDGE_MARGIN or by2 > H - EDGE_MARGIN:
        return False, "bbox_en_borde"
    # Tamano minimo
    if bw < MIN_BBOX_W or bh < MIN_BBOX_H:
        return False, f"bbox_chico_{bw}x{bh}"
    # Silueta presente
    if sil_full is None:
        return False, "silueta_vacia"
    sil_in_bbox = sil_full[by1:by2, bx1:bx2]
    sil_area = int((sil_in_bbox > 0).sum())
    bbox_area = bw * bh
    if sil_area / bbox_area < MIN_SILUETA_BBOX_RATIO:
        return False, f"silueta_chica_{sil_area / bbox_area:.2f}"
    # Cobertura X de la silueta dentro del bbox
    cols_with_sil = np.where((sil_in_bbox > 0).any(axis=0))[0]
    if len(cols_with_sil) == 0:
        return False, "silueta_sin_cols"
    cob_x = (cols_with_sil.max() - cols_with_sil.min() + 1) / bw * 100
    if cob_x < MIN_SILUETA_COB_X:
        return False, f"silueta_truncada_X_{cob_x:.0f}"
    return True, "ok"


def make_id(video_stem, frame_idx):
    safe = video_stem.replace("'", "_").replace(" ", "_")
    return f"22abril_{safe}_f{frame_idx:05d}"


def main():
    BARRIL_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted([p for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in (".mov", ".mp4", ".avi")])
    print(f"Videos: {len(videos)}")

    print("Cargando modelos...")
    coco = YOLO(str(PROJECT / "yolov8n.pt"))
    silueta = YOLO(str(PROJECT / "silueta_seg.pt"))

    frames_idx = load_index()
    a_borrar = [e for e in frames_idx if e.get("source") == "22abril"]
    for e in a_borrar:
        for k in ("img", "mask"):
            p = BARRIL_DIR / e.get(k, "")
            if p.exists():
                try: p.unlink()
                except Exception: pass
    if a_borrar:
        print(f"Limpieza: removidas {len(a_borrar)} entradas previas de 22abril")
    frames_idx = [e for e in frames_idx if e.get("source") != "22abril"]
    existing_ids = {e["id"] for e in frames_idx}

    n_added = 0
    motivos_skip = {}
    per_video_stats = []

    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps / SAMPLING_FPS)))
        indices = list(range(0, n_total, step))
        n_added_v = n_skip_v = 0
        skip_reasons_v = {}

        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            H, W = frame.shape[:2]

            rc = coco(frame, classes=[19], conf=COW_CONF, verbose=False)
            if not rc or len(rc[0].boxes) == 0:
                skip_reasons_v["sin_vaca"] = skip_reasons_v.get("sin_vaca", 0) + 1
                n_skip_v += 1
                continue
            boxes = rc[0].boxes.xyxy.cpu().numpy()
            scores = rc[0].boxes.conf.cpu().numpy()
            bi = int(np.argmax(scores))
            bx1, by1, bx2, by2 = [int(v) for v in boxes[bi]]
            bbox = (bx1, by1, bx2, by2)

            sil_full = silueta_full_frame(frame, silueta, bbox)
            ok_v, motivo = vaca_completa(bbox, frame.shape, sil_full)
            if not ok_v:
                skip_reasons_v[motivo] = skip_reasons_v.get(motivo, 0) + 1
                n_skip_v += 1
                continue

            # Crop con padding generoso
            pad = max(80, int(PAD_RATIO * max(bx2 - bx1, by2 - by1)))
            cx1 = max(0, bx1 - pad); cy1 = max(0, by1 - pad)
            cx2 = min(W, bx2 + pad); cy2 = min(H, by2 + pad)
            crop = frame[cy1:cy2, cx1:cx2]
            ch, cw = crop.shape[:2]

            # Guardar silueta CRUDA como mascara inicial. NO usar postprocesar_barril
            # porque su recorte de patas falla en algunas poses (corta demasiado alto
            # cuando la "panza" detectada por perfil de anchos cae fuera del torso).
            # El editor manual recorta patas + cabeza + cola.
            mask_init = sil_full[cy1:cy2, cx1:cx2]

            fid = make_id(vp.stem, fi)
            if fid in existing_ids:
                continue
            img_filename = f"{fid}_img.png"
            mask_filename = f"{fid}_mask.png"
            cv2.imwrite(str(BARRIL_DIR / img_filename), crop)
            cv2.imwrite(str(BARRIL_DIR / mask_filename), mask_init)

            frames_idx.append({
                "id": fid,
                "individuo": f"22abril_{vp.stem}",
                "video": vp.name,
                "frame_idx": fi,
                "bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "crop_w": int(cw),
                "crop_h": int(ch),
                "img": img_filename,
                "mask": mask_filename,
                "status": "pending",
                "cuts": [],
                "source": "22abril",
                "fps_sample": SAMPLING_FPS,
                "vaca_bbox_full": [bx1, by1, bx2, by2],
            })
            existing_ids.add(fid)
            n_added += 1
            n_added_v += 1
        cap.release()

        for k, v in skip_reasons_v.items():
            motivos_skip[k] = motivos_skip.get(k, 0) + v
        per_video_stats.append((vp.name, len(indices), n_added_v, n_skip_v, skip_reasons_v))
        print(f"  {vp.name}: {len(indices)} sampleados -> {n_added_v} agregados  ({n_skip_v} descartados)")

    save_index(frames_idx)

    print("\n" + "=" * 72)
    print(f"{'video':<32} {'sampled':>8} {'added':>6} {'skip':>5}")
    print("-" * 72)
    for v, ns, na, sk, _ in per_video_stats:
        print(f"{v:<32} {ns:>8} {na:>6} {sk:>5}")
    print("-" * 72)
    print(f"{'TOTAL':<32} {sum(s[1] for s in per_video_stats):>8} {n_added:>6} {sum(s[3] for s in per_video_stats):>5}")
    print(f"\nMotivos de descarte:")
    for k, v in sorted(motivos_skip.items(), key=lambda x: -x[1]):
        print(f"  {k:<28} {v}")
    print(f"\nTotal en index: {len(frames_idx)}  (22abril: {n_added})")


if __name__ == "__main__":
    main()
