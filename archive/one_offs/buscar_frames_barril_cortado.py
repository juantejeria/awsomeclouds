"""Recorre un video y encuentra frames donde barril_seg.pt parte el barril."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str)
    ap.add_argument("--step", type=int, default=5, help="Procesar 1 cada N frames")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="checkpoints/frames_barril_cortado")
    args = ap.parse_args()

    project = Path(__file__).parent
    barril_model = YOLO(str(project / "barril_seg.pt"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Procesando {n_total} frames (1 cada {args.step})...")

    candidatos = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.step != 0:
            idx += 1
            continue

        res = barril_model(frame, conf=args.conf, verbose=False)
        if res and res[0].masks is not None:
            masks = res[0].masks.data.cpu().numpy()
            confs = res[0].boxes.conf.cpu().numpy() if res[0].boxes is not None else None
            h, w = frame.shape[:2]

            n_masks = len(masks)
            max_cc = 0
            for k, m in enumerate(masks):
                m_b = (cv2.resize(m, (w, h)) > 0.5).astype(np.uint8)
                n_cc, _, stats, _ = cv2.connectedComponentsWithStats(m_b, connectivity=8)
                ccs = sorted([stats[c, cv2.CC_STAT_AREA] for c in range(1, n_cc)], reverse=True)
                if len(ccs) > max_cc:
                    max_cc = len(ccs)

            es_problema = n_masks >= 2 or max_cc >= 2
            tag = "PROBLEMA" if es_problema else "ok"
            print(f"  frame {idx:3d}: masks={n_masks} max_cc={max_cc}  conf_max={confs.max() if confs is not None else 0:.2f}  {tag}")

            if es_problema:
                candidatos.append((idx, n_masks, max_cc))
                cv2.imwrite(str(out_dir / f"frame_{idx:04d}_m{n_masks}_cc{max_cc}.png"), frame)

        idx += 1

    cap.release()

    print(f"\n{len(candidatos)} frame(s) con barril cortado guardados en: {out_dir}/")
    for fr_idx, nm, cc in candidatos:
        print(f"  frame {fr_idx}: {nm} mascara(s), max {cc} componente(s) conexo(s)")


if __name__ == "__main__":
    main()
