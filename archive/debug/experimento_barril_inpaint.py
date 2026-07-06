"""Experimento: borrar el poste rojo via inpainting antes de barril_seg.

Si la causa del barril truncado es el poste confundiendo al modelo, al borrarlo
visualmente con cv2.inpaint el modelo deberia segmentar las dos mitades.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def cargar_imagen(path: Path, frame_idx: int):
    if path.suffix.lower() in (".mov", ".mp4", ".avi", ".mkv"):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, fr = cap.read()
        cap.release()
        return fr if ok else None
    return cv2.imread(str(path))


def detectar_poste_rojo(img):
    """Devuelve mascara binaria de pixeles del poste rojo.
    Usa HSV: rojo saturado en dos rangos (0-10 y 170-180).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 110, 70), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 110, 70), (180, 255, 255))
    red = m1 | m2
    # Cerrar pequenos huecos del poste y dilatar un poco
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel, iterations=1)
    red = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=2)
    return red


def correr_barril(model, crop, conf=0.25):
    r = model(crop, conf=conf, verbose=False)
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        return None, []
    masks = r[0].masks.data.cpu().numpy()
    confs = r[0].boxes.conf.cpu().numpy() if r[0].boxes is not None else None
    h, w = crop.shape[:2]
    binarias = []
    for i, m in enumerate(masks):
        m_b = (cv2.resize(m, (w, h)) > 0.5).astype(np.uint8)
        binarias.append((m_b, float(confs[i]) if confs is not None else None))
    # Union app.py-style: area >= 5% del max
    areas = np.array([m.sum() for m, _ in binarias])
    keep = areas >= 0.05 * areas.max()
    union = np.zeros((h, w), dtype=np.uint8)
    for k, (m, _) in enumerate(binarias):
        if keep[k]:
            union = cv2.bitwise_or(union, m)
    return union, binarias


def stats(mask):
    if mask is None or mask.sum() == 0:
        return dict(area=0, n_cc=0, ccs=[], cobertura_x=0, cobertura_y=0)
    ys, xs = np.where(mask > 0)
    n_cc, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ccs = sorted([st[k, cv2.CC_STAT_AREA] for k in range(1, n_cc)], reverse=True)
    h, w = mask.shape
    return dict(
        area=int(mask.sum()),
        n_cc=n_cc - 1,
        ccs=ccs,
        cobertura_x=(xs.max() - xs.min() + 1) / w * 100,
        cobertura_y=(ys.max() - ys.min() + 1) / h * 100,
    )


def overlay(img, mask, color, label):
    out = img.copy()
    ov = np.zeros_like(out)
    ov[mask > 0] = color
    out = cv2.addWeighted(out, 0.55, ov, 0.45, 0)
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada", type=str)
    ap.add_argument("--frame", type=int, default=150)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--inpaint-radius", type=int, default=7)
    args = ap.parse_args()

    project = Path(__file__).parent
    p = Path(args.entrada)
    img = cargar_imagen(p, args.frame)
    if img is None:
        print("ERROR: no pude leer la imagen")
        sys.exit(1)
    suffix = f"_f{args.frame}" if p.suffix.lower() in (".mov", ".mp4") else ""
    out_path = Path(args.out) if args.out else p.parent / f"exp_inpaint_{p.stem}{suffix}.png"

    coco = YOLO(str(project / "yolov8n.pt"))
    barril = YOLO(str(project / "barril_seg.pt"))

    # Cow crop (igual que app.py)
    rc = coco(img, classes=[19], conf=0.2, verbose=False)
    if not rc or len(rc[0].boxes) == 0:
        print("ERROR: no detecto vaca")
        sys.exit(1)
    boxes = rc[0].boxes.xyxy.cpu().numpy()
    scores = rc[0].boxes.conf.cpu().numpy()
    bi = int(np.argmax(scores))
    bx1, by1, bx2, by2 = [int(v) for v in boxes[bi]]
    pad = max(20, int(0.08 * max(bx2 - bx1, by2 - by1)))
    H, W = img.shape[:2]
    cx1 = max(0, bx1 - pad); cy1 = max(0, by1 - pad)
    cx2 = min(W, bx2 + pad); cy2 = min(H, by2 + pad)
    crop = img[cy1:cy2, cx1:cx2].copy()
    ch, cw = crop.shape[:2]
    print(f"Cow crop: {cw}x{ch}")

    # Detectar poste rojo en el crop
    poste_mask = detectar_poste_rojo(crop)
    poste_pixeles = int(poste_mask.sum() / 255)
    print(f"Pixeles del poste detectados en el crop: {poste_pixeles}")

    # Inpainting
    crop_clean = cv2.inpaint(crop, poste_mask, args.inpaint_radius, cv2.INPAINT_TELEA)

    # barril_seg sobre original y sobre limpia
    union_orig, _ = correr_barril(barril, crop, conf=args.conf)
    union_clean, _ = correr_barril(barril, crop_clean, conf=args.conf)

    s_o = stats(union_orig)
    s_c = stats(union_clean)
    print("\n=== RESULTADOS ===")
    print(f"Original: area={s_o['area']}px  cc={s_o['n_cc']}  ccs={s_o['ccs'][:3]}  "
          f"cobertura X={s_o['cobertura_x']:.0f}%  Y={s_o['cobertura_y']:.0f}%")
    print(f"Inpaint : area={s_c['area']}px  cc={s_c['n_cc']}  ccs={s_c['ccs'][:3]}  "
          f"cobertura X={s_c['cobertura_x']:.0f}%  Y={s_c['cobertura_y']:.0f}%")
    if s_o['area'] > 0:
        delta = (s_c['area'] - s_o['area']) / s_o['area'] * 100
        print(f">> Cambio de area: {delta:+.1f}%")
        delta_x = s_c['cobertura_x'] - s_o['cobertura_x']
        print(f">> Cambio de cobertura X: {delta_x:+.1f} puntos porcentuales")

    # Visualizacion 2x2
    panels = []
    panels.append(("ORIGINAL crop", crop))
    p_post = crop.copy()
    p_post[poste_mask > 0] = (0, 255, 255)
    panels.append((f"poste detectado ({poste_pixeles}px)", p_post))
    panels.append(("INPAINT (poste borrado)", crop_clean))
    if union_orig is not None:
        panels.append((f"barril SOBRE ORIG  cc={s_o['n_cc']} covX={s_o['cobertura_x']:.0f}%",
                       overlay(crop, union_orig, (0, 165, 255), "")))
    else:
        panels.append(("barril ORIG: sin deteccion", crop.copy()))
    if union_clean is not None:
        panels.append((f"barril SOBRE INPAINT  cc={s_c['n_cc']} covX={s_c['cobertura_x']:.0f}%",
                       overlay(crop_clean, union_clean, (0, 255, 100), "")))
    else:
        panels.append(("barril INPAINT: sin deteccion", crop_clean.copy()))

    pw = 380
    ph = int(pw * ch / cw)
    rendered = []
    for title, im in panels:
        r = cv2.resize(im, (pw, ph))
        cv2.putText(r, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(r, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        rendered.append(r)
    cols = 3
    rows = (len(rendered) + cols - 1) // cols
    grid = np.full((rows * ph, cols * pw, 3), 30, dtype=np.uint8)
    for i, r in enumerate(rendered):
        rr, cc = divmod(i, cols)
        grid[rr * ph:(rr + 1) * ph, cc * pw:(cc + 1) * pw] = r
    cv2.imwrite(str(out_path), grid)
    print(f"\nVisualizacion: {out_path}")


if __name__ == "__main__":
    main()
