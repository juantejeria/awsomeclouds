"""Recorre todos los videos de checkpoints/22abril/, samplea a 15 fps, corre
el pipeline real de barril_seg (igual que app.py) sobre cada frame, y reporta
cuantos salen truncados (cobertura X < 75% del cow_crop).

Genera:
  - checkpoints/22abril_diagnostico/<video>_grid.png  (truncados + muestra de OK)
  - checkpoints/22abril_diagnostico/resumen.csv       (una fila por frame)
  - resumen final en consola
"""
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT = Path(__file__).parent
VIDEOS_DIR = PROJECT / "checkpoints" / "22abril"
OUT_DIR = PROJECT / "checkpoints" / "22abril_diagnostico"
SAMPLING_FPS = 15.0
COBERTURA_THRESHOLD = 75.0  # cobertura X menor a esto -> truncado
MAX_OK_TILES = 6              # mostrar hasta 6 OK de muestra por video


def correr_barril_pipeline(img, coco_model, barril_model):
    """Replica app.py /generate_3d_from_frame: bbox -> crop+pad -> barril_seg -> union."""
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
        return dict(
            crop=crop, mask=None, n_masks=0, area=0, n_cc=0,
            cobertura_x=0, cobertura_y=0, conf_max=0,
        )
    masks = rb[0].masks.data.cpu().numpy()
    confs = rb[0].boxes.conf.cpu().numpy() if rb[0].boxes is not None else None
    areas = np.array([float(m.sum()) for m in masks])
    keep = areas >= 0.05 * areas.max()
    union_raw = np.max(masks[keep], axis=0)
    if union_raw.shape != (ch, cw):
        union_raw = cv2.resize(union_raw, (cw, ch))
    mask = (union_raw > 0.5).astype(np.uint8)

    n_cc, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        cob_x = cob_y = 0.0
    else:
        cob_x = (xs.max() - xs.min() + 1) / cw * 100
        cob_y = (ys.max() - ys.min() + 1) / ch * 100

    return dict(
        crop=crop,
        mask=mask,
        n_masks=int(len(masks)),
        area=int(mask.sum()),
        n_cc=int(n_cc - 1),
        cobertura_x=float(cob_x),
        cobertura_y=float(cob_y),
        conf_max=float(confs.max()) if confs is not None else 0.0,
    )


def render_tile(crop, mask, label, ok):
    out = crop.copy()
    if mask is not None and mask.sum() > 0:
        ov = np.zeros_like(out)
        color = (0, 255, 100) if ok else (0, 100, 255)
        ov[mask > 0] = color
        out = cv2.addWeighted(out, 0.55, ov, 0.45, 0)
    cv2.putText(out, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 3)
    cv2.putText(out, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted([p for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in (".mov", ".mp4", ".avi")])
    print(f"Videos a procesar: {len(videos)}")

    print("Cargando modelos...")
    coco = YOLO(str(PROJECT / "yolov8n.pt"))
    barril = YOLO(str(PROJECT / "barril_seg.pt"))

    csv_path = OUT_DIR / "resumen.csv"
    fcsv = open(csv_path, "w", newline="")
    writer = csv.writer(fcsv)
    writer.writerow([
        "video", "frame_idx", "n_masks", "n_cc", "area_px",
        "cobertura_x_pct", "cobertura_y_pct", "conf_max", "estado",
    ])

    resumen_global = []

    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps / SAMPLING_FPS)))
        indices = list(range(0, n_total, step))
        print(f"\n{vp.name}: {n_total} frames @ {fps:.1f}fps -> step {step} -> {len(indices)} sampleados")

        tiles_truncados = []
        tiles_ok = []
        n_ok = n_trunc = n_nodet = 0
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            res = correr_barril_pipeline(frame, coco, barril)
            if res is None:
                writer.writerow([vp.name, fi, 0, 0, 0, 0, 0, 0, "sin_vaca"])
                n_nodet += 1
                continue
            if res["mask"] is None or res["area"] == 0:
                estado = "sin_barril"
                n_nodet += 1
            else:
                es_ok = res["cobertura_x"] >= COBERTURA_THRESHOLD
                estado = "ok" if es_ok else "truncado"
                if es_ok: n_ok += 1
                else: n_trunc += 1
            writer.writerow([
                vp.name, fi, res["n_masks"], res["n_cc"], res["area"],
                f"{res['cobertura_x']:.1f}", f"{res['cobertura_y']:.1f}",
                f"{res['conf_max']:.2f}", estado,
            ])
            label = f"f{fi} {res['cobertura_x']:.0f}% cc{res['n_cc']}"
            tile = render_tile(res["crop"], res["mask"], label, estado == "ok")
            if estado == "truncado":
                tiles_truncados.append(tile)
            elif estado == "ok" and len(tiles_ok) < MAX_OK_TILES:
                # Sample equiespaciada de OK
                tiles_ok.append(tile)
        cap.release()
        print(f"  -> ok={n_ok}  truncado={n_trunc}  sin_deteccion={n_nodet}")
        resumen_global.append((vp.name, len(indices), n_ok, n_trunc, n_nodet))

        # Grid: truncados primero, OK de muestra al final
        tiles = tiles_truncados + tiles_ok
        if tiles:
            tw = 200
            ratios = [t.shape[0] / t.shape[1] for t in tiles]
            th = int(tw * (sum(ratios) / len(ratios)))
            tiles_r = [cv2.resize(t, (tw, th)) for t in tiles]
            cols = 8
            rows = (len(tiles_r) + cols - 1) // cols
            grid = np.full((rows * th, cols * tw, 3), 30, dtype=np.uint8)
            for i, t in enumerate(tiles_r):
                r, c = divmod(i, cols)
                grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
            header_h = 36
            full = np.full((grid.shape[0] + header_h, grid.shape[1], 3), 50, dtype=np.uint8)
            full[header_h:] = grid
            txt = (f"{vp.name}  total={len(indices)}  ok={n_ok}  TRUNCADO={n_trunc}  "
                   f"sin_det={n_nodet}  (truncados primero, luego muestra de OK)")
            cv2.putText(full, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imwrite(str(OUT_DIR / f"{vp.stem}_grid.png"), full)

    fcsv.close()

    print("\n" + "=" * 72)
    print(f"{'video':<30} {'total':>6} {'ok':>5} {'trunc':>6} {'no_det':>7}")
    print("-" * 72)
    tot = ok_t = tr_t = nd_t = 0
    for v, n, k, t, d in resumen_global:
        tot += n; ok_t += k; tr_t += t; nd_t += d
        print(f"{v:<30} {n:>6} {k:>5} {t:>6} {d:>7}")
    print("-" * 72)
    print(f"{'TOTAL':<30} {tot:>6} {ok_t:>5} {tr_t:>6} {nd_t:>7}")
    pct_trunc = tr_t / max(1, tot) * 100
    print(f"\n>> {tr_t}/{tot} frames ({pct_trunc:.1f}%) con barril TRUNCADO (cobertura X < {COBERTURA_THRESHOLD}%)")
    print(f">> CSV: {csv_path}")
    print(f">> Grids: {OUT_DIR}/")


if __name__ == "__main__":
    main()
