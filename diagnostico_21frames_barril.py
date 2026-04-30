"""Genera un grid PNG con los 21 recortes de cow_crop + máscara de barril
overlay, para inspección visual.

Uso:
    python diagnostico_21frames_barril.py <carpeta_frames> [--out salida.png]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def correr_barril_pipeline(img, coco_model, barril_model):
    H, W = img.shape[:2]
    rc = coco_model(img, classes=[19], conf=0.2, verbose=False)
    if not rc or len(rc[0].boxes) == 0:
        return None
    boxes = rc[0].boxes.xyxy.cpu().numpy()
    scores = rc[0].boxes.conf.cpu().numpy()
    bi = int(np.argmax(scores))
    bx1, by1, bx2, by2 = [int(v) for v in boxes[bi]]
    pad = max(20, int(0.08 * max(bx2 - bx1, by2 - by1)))
    cx1 = max(0, bx1 - pad)
    cy1 = max(0, by1 - pad)
    cx2 = min(W, bx2 + pad)
    cy2 = min(H, by2 + pad)
    crop = img[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if ch < 30 or cw < 30:
        return None
    rb = barril_model(crop, conf=0.25, verbose=False)
    if not rb or rb[0].masks is None or len(rb[0].masks.data) == 0:
        return dict(crop=crop, mask=None, conf_max=0.0)
    masks = rb[0].masks.data.cpu().numpy()
    confs = rb[0].boxes.conf.cpu().numpy() if rb[0].boxes is not None else None
    areas = np.array([float(m.sum()) for m in masks])
    keep = areas >= 0.05 * areas.max()
    union = np.max(masks[keep], axis=0)
    if union.shape != (ch, cw):
        union = cv2.resize(union, (cw, ch))
    mask = (union > 0.5).astype(np.uint8)
    conf_max = float(confs.max()) if confs is not None and len(confs) else 0.0
    return dict(crop=crop, mask=mask, conf_max=conf_max)


def overlay_mask(crop, mask, color=(0, 140, 230), alpha=0.45):
    out = crop.copy()
    if mask is None:
        return out
    color_layer = np.zeros_like(out)
    color_layer[mask > 0] = color
    out = cv2.addWeighted(out, 1.0, color_layer, alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def make_tile(crop_overlay, label, tile_w=320):
    h, w = crop_overlay.shape[:2]
    if w == 0:
        return None
    scale = tile_w / w
    th = int(h * scale)
    img = cv2.resize(crop_overlay, (tile_w, th))
    bar = np.zeros((28, tile_w, 3), dtype=np.uint8)
    cv2.putText(bar, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cols", type=int, default=7)
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[error] no existe {folder}"); return
    out_path = Path(args.out) if args.out else (folder / "diagnostico_barril_grid.png")

    proj = Path(__file__).parent
    print("[init] cargando modelos...")
    barril = YOLO(str(proj / "barril_seg.pt"))
    coco = YOLO(str(proj / "yolov8n.pt"))

    files = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".jpg"])
    print(f"[init] {len(files)} frames")

    tiles = []
    for fp in files:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        res = correr_barril_pipeline(img, coco, barril)
        if res is None:
            continue
        crop = res["crop"]
        mask = res["mask"]
        cov = "no-mask"
        if mask is not None and mask.sum() > 0:
            cols_any = (mask.sum(axis=0) > 0).sum()
            cov = f"covX={cols_any}/{mask.shape[1]} ({100*cols_any/mask.shape[1]:.0f}%)"
        ov = overlay_mask(crop, mask)
        label = f"{fp.name} | conf={res['conf_max']:.2f} | {cov}"
        tile = make_tile(ov, label)
        if tile is not None:
            tiles.append(tile)
        print(f"  {fp.name:30s} conf={res['conf_max']:.2f} {cov}")

    if not tiles:
        print("[error] sin tiles válidos"); return

    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    grid_tiles = []
    for r in range(rows):
        row_imgs = tiles[r * cols : (r + 1) * cols]
        if len(row_imgs) < cols:
            blank = np.zeros((th, tw, 3), dtype=np.uint8)
            row_imgs = row_imgs + [blank] * (cols - len(row_imgs))
        row_imgs = [cv2.resize(t, (tw, th)) for t in row_imgs]
        grid_tiles.append(np.hstack(row_imgs))
    grid = np.vstack(grid_tiles)
    cv2.imwrite(str(out_path), grid)
    print(f"[ok] grid → {out_path}")


if __name__ == "__main__":
    main()
