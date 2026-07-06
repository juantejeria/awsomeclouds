"""Diagnostico: replica el pipeline de app.py /generate_3d_from_frame
para entender por que el barril sale cortado en el setup nuevo.

Pipeline real (app.py:1162-1199):
  1. coco yolov8n class=cow conf=0.2 -> bbox
  2. padding 8% -> cow_crop
  3. barril_seg sobre cow_crop, conf=0.25
  4. union de mascaras con area >= 5% del max
  5. _reparar_mascara_oclusion (interpola columnas vacias DENTRO del rango X)

Uso:
    python diagnostico_barril_corte.py <imagen_o_video> [--frame N] [--out salida.png]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def reparar_oclusion(binmask):
    """Copia de _reparar_mascara_oclusion de app.py."""
    bh, bw = binmask.shape
    cols_sum = binmask.sum(axis=0)
    cols_valid = np.where(cols_sum > 0)[0]
    if cols_valid.size < 2:
        return 0
    col_first = int(cols_valid[0])
    col_last = int(cols_valid[-1])
    top_arr = np.full(bw, -1, dtype=np.int32)
    bot_arr = np.full(bw, -1, dtype=np.int32)
    for _c in cols_valid:
        _rows = np.where(binmask[:, _c] > 0)[0]
        top_arr[_c] = int(_rows[0])
        bot_arr[_c] = int(_rows[-1])
    n_filled = 0
    c = col_first + 1
    while c < col_last:
        if top_arr[c] < 0:
            gap_start = c
            gap_end = c
            while gap_end + 1 < col_last and top_arr[gap_end + 1] < 0:
                gap_end += 1
            left = gap_start - 1
            right = gap_end + 1
            top_L, bot_L = int(top_arr[left]), int(bot_arr[left])
            top_R, bot_R = int(top_arr[right]), int(bot_arr[right])
            span = right - left
            for k, col_k in enumerate(range(gap_start, gap_end + 1), start=1):
                alpha = k / span
                top_k = int(round((1 - alpha) * top_L + alpha * top_R))
                bot_k = int(round((1 - alpha) * bot_L + alpha * bot_R))
                if bot_k >= top_k:
                    binmask[top_k:bot_k + 1, col_k] = 1
                    top_arr[col_k] = top_k
                    bot_arr[col_k] = bot_k
                    n_filled += 1
            c = gap_end + 1
        else:
            c += 1
    return n_filled


def cargar_imagen(path: Path, frame_idx: int):
    """Si es video, extrae el frame indicado. Si es imagen, lo carga."""
    if path.suffix.lower() in (".mov", ".mp4", ".avi", ".mkv"):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    return cv2.imread(str(path))


def annotate(panel, text):
    cv2.putText(panel, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(panel, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada", type=str, help="Imagen PNG/JPG o video MP4/MOV")
    ap.add_argument("--frame", type=int, default=150, help="Si entrada es video, frame a usar")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    project = Path(__file__).parent
    in_path = Path(args.entrada)
    if not in_path.exists():
        print(f"ERROR: no existe {in_path}")
        sys.exit(1)

    img = cargar_imagen(in_path, args.frame)
    if img is None:
        print("ERROR: no pude leer la imagen/frame")
        sys.exit(1)
    h_orig, w_orig = img.shape[:2]
    suffix = f"_f{args.frame}" if in_path.suffix.lower() in (".mov", ".mp4") else ""
    out_path = Path(args.out) if args.out else in_path.parent / f"diag_barril_{in_path.stem}{suffix}.png"
    print(f"Entrada: {in_path.name}{suffix}  {w_orig}x{h_orig}")

    coco_model = YOLO(str(project / "yolov8n.pt"))
    barril_model = YOLO(str(project / "barril_seg.pt"))
    silueta_path = project / "silueta_seg.pt"
    silueta_model = YOLO(str(silueta_path)) if silueta_path.exists() else None

    # ── 1. Detectar vaca con yolov8n class=cow (igual que app.py) ──
    r_cow = coco_model(img, classes=[19], conf=0.2, verbose=False)
    if not r_cow or len(r_cow[0].boxes) == 0:
        print("ERROR: yolov8n no detecto vaca")
        sys.exit(1)
    boxes = r_cow[0].boxes.xyxy.cpu().numpy()
    scores = r_cow[0].boxes.conf.cpu().numpy()
    bi = int(np.argmax(scores))
    bx1, by1, bx2, by2 = [int(v) for v in boxes[bi]]
    print(f"Vaca: bbox=({bx1},{by1})-({bx2},{by2}) conf={scores[bi]:.2f}")

    # ── 2. Cow crop con padding 8% ──
    pad = max(20, int(0.08 * max(bx2 - bx1, by2 - by1)))
    cx1 = max(0, bx1 - pad)
    cy1 = max(0, by1 - pad)
    cx2 = min(w_orig, bx2 + pad)
    cy2 = min(h_orig, by2 + pad)
    cow_crop = img[cy1:cy2, cx1:cx2]
    ch, cw = cow_crop.shape[:2]
    print(f"Cow crop: {cw}x{ch}  pad={pad}px")

    # ── 3. barril_seg sobre cow_crop ──
    r_bar = barril_model(cow_crop, conf=args.conf, verbose=False)
    if not r_bar or r_bar[0].masks is None or len(r_bar[0].masks.data) == 0:
        print(f"ERROR: barril_seg no detecto nada en cow_crop (conf={args.conf})")
        # Probar con conf bajisima para descartar
        for c2 in [0.1, 0.05, 0.01]:
            r2 = barril_model(cow_crop, conf=c2, verbose=False)
            if r2 and r2[0].masks is not None and len(r2[0].masks.data) > 0:
                print(f"  -> con conf={c2} si detecta {len(r2[0].masks.data)} mascara(s)")
                r_bar = r2
                break
        else:
            sys.exit(1)

    masks = r_bar[0].masks.data.cpu().numpy()
    confs = r_bar[0].boxes.conf.cpu().numpy() if r_bar[0].boxes is not None else None
    print(f"\n=== barril_seg devolvio {len(masks)} mascara(s) en cow_crop ===")

    info = []
    for i, m in enumerate(masks):
        m_b = (cv2.resize(m, (cw, ch)) > 0.5).astype(np.uint8)
        area = int(m_b.sum())
        n_cc, _, stats, _ = cv2.connectedComponentsWithStats(m_b, connectivity=8)
        ccs = sorted([stats[k, cv2.CC_STAT_AREA] for k in range(1, n_cc)], reverse=True)
        info.append(dict(idx=i, area=area, ccs=ccs, conf=float(confs[i]) if confs is not None else None, m=m_b))
        print(f"  Mask #{i}: conf={info[-1]['conf']:.2f}  area={area}px  componentes={len(ccs)} -> {ccs[:5]}")

    # ── 4. Union (replicando app.py:1184-1190) ──
    areas_arr = np.array([float(np.sum(masks[i])) for i in range(len(masks))])
    max_area = float(areas_arr.max())
    keep = areas_arr >= 0.05 * max_area
    sil_mask_raw = np.max(masks[keep], axis=0)
    if sil_mask_raw.shape != (ch, cw):
        sil_mask_raw = cv2.resize(sil_mask_raw, (cw, ch))
    union_mask = (sil_mask_raw > 0.5).astype(np.uint8)
    n_cc_u, _, stats_u, _ = cv2.connectedComponentsWithStats(union_mask, connectivity=8)
    ccs_u = sorted([stats_u[k, cv2.CC_STAT_AREA] for k in range(1, n_cc_u)], reverse=True)
    print(
        f"\n>> Union (mascaras con area >= 5% max): area={int(union_mask.sum())}, "
        f"componentes_conexos={len(ccs_u)} -> {ccs_u[:5]}"
    )

    # ── 5. Reparar oclusion ──
    rep_mask = union_mask.copy()
    n_filled = reparar_oclusion(rep_mask)
    n_cc_r, _, stats_r, _ = cv2.connectedComponentsWithStats(rep_mask, connectivity=8)
    ccs_r = sorted([stats_r[k, cv2.CC_STAT_AREA] for k in range(1, n_cc_r)], reverse=True)
    print(
        f">> Reparada (cols rellenadas={n_filled}): area={int(rep_mask.sum())}, "
        f"componentes={len(ccs_r)} -> {ccs_r[:5]}"
    )

    # ── 6. Bbox de la mascara reparada vs bbox de la vaca ──
    ys_r, xs_r = np.where(rep_mask > 0)
    if xs_r.size:
        bxR = (int(xs_r.min()), int(ys_r.min()), int(xs_r.max()), int(ys_r.max()))
        cobertura_x = (bxR[2] - bxR[0]) / cw * 100
        print(f">> Bbox mask reparada en cow_crop: {bxR}  cubre {cobertura_x:.0f}% del ancho del crop")
        if cobertura_x < 70:
            print(f"   ATENCION: la mascara cubre menos del 70% del ancho del crop -> probable barril TRUNCADO (no cortado por el medio)")

    # ── Visualizacion ──
    panels = []
    panel_w = 360
    panel_h = int(panel_w * ch / cw)

    p0 = cow_crop.copy()
    p0 = cv2.resize(p0, (panel_w, panel_h))
    annotate(p0, "cow_crop")
    panels.append(p0)

    colors = [(0, 165, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
    for k, e in enumerate(info):
        p = cow_crop.copy()
        ov = np.zeros_like(p)
        ov[e["m"] > 0] = colors[k % len(colors)]
        p = cv2.addWeighted(p, 0.55, ov, 0.45, 0)
        n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(e["m"], connectivity=8)
        for c in range(1, n_cc):
            x_, y_, w_, h_ = stats[c, cv2.CC_STAT_LEFT], stats[c, cv2.CC_STAT_TOP], stats[c, cv2.CC_STAT_WIDTH], stats[c, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(p, (x_, y_), (x_ + w_, y_ + h_), (0, 0, 255), 2)
        p = cv2.resize(p, (panel_w, panel_h))
        annotate(p, f"mask#{e['idx']} c={e['conf']:.2f} cc={len(e['ccs'])}")
        panels.append(p)

    p_un = cow_crop.copy()
    ov = np.zeros_like(p_un)
    ov[union_mask > 0] = (255, 200, 0)
    p_un = cv2.addWeighted(p_un, 0.55, ov, 0.45, 0)
    p_un = cv2.resize(p_un, (panel_w, panel_h))
    annotate(p_un, f"UNION cc={len(ccs_u)} area={int(union_mask.sum())}")
    panels.append(p_un)

    p_rep = cow_crop.copy()
    ov = np.zeros_like(p_rep)
    ov[rep_mask > 0] = (0, 255, 100)
    p_rep = cv2.addWeighted(p_rep, 0.55, ov, 0.45, 0)
    p_rep = cv2.resize(p_rep, (panel_w, panel_h))
    annotate(p_rep, f"REPARADA cols={n_filled} cc={len(ccs_r)}")
    panels.append(p_rep)

    cols = min(len(panels), 3)
    rows = (len(panels) + cols - 1) // cols
    grid = np.full((rows * panel_h, cols * panel_w, 3), 30, dtype=np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        grid[r * panel_h:(r + 1) * panel_h, c * panel_w:(c + 1) * panel_w] = p
    cv2.imwrite(str(out_path), grid)
    print(f"\nVisualizacion: {out_path}")


if __name__ == "__main__":
    main()
