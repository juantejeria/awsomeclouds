"""Diagnóstico batch para todas las vacas del setup 6mayo.

Para cada carpeta en checkpoints/6mayo/<folder>/:
  1. Parsea altura_real y peso_real del nombre (formato <altura>_<peso>[_resto]).
  2. Computa altura_calculada usando los postes del context.json y el frame
     central (bbox YOLO).
  3. Corre procesar_21_frames.py si no existe el resumen → genera el modelo 3D
     y el vol_barril_litros.
  4. Imprime una tabla y guarda diagnostico_6mayo.csv.

Uso:
    python batch_diagnostico_6mayo.py
    python batch_diagnostico_6mayo.py --skip-3d   # solo altura calculada, sin generar 3D
    python batch_diagnostico_6mayo.py --force     # re-procesa aunque exista resumen
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).parent
DATASET_DIR = PROJECT / "checkpoints" / "6mayo"
OUTPUT_DIR = PROJECT / "output_modelos3d_6mayo"
RESULTS_CSV = PROJECT / "diagnostico_6mayo.csv"

POSTE_REAL_CM = 110.0


def parse_folder_name(name):
    """<altura>_<peso>[_resto] → (altura_real, peso_real, suffix)."""
    m = re.match(r"^(\d+)_(\d+)(?:_(.*))?$", name)
    if not m:
        return None, None, name
    return float(m.group(1)), float(m.group(2)), (m.group(3) or "")


def find_central_frame(folder):
    """Busca frame_000_f<N>.jpg dentro de la carpeta."""
    for p in folder.glob("frame_000_*.jpg"):
        return p
    # Algunos formatos podrían usar frame_0_ o frame_000_
    for p in folder.glob("frame_0_*.jpg"):
        return p
    return None


def compute_altura_calculada(folder, coco_model):
    """Usa el context.json (locked_reference) + el frame central para computar
    altura_calculada via los postes.

    Returns (altura_cm, cm_per_px, bbox_h_px) o (None, None, None) si falla.
    """
    ctx_path = folder / "context.json"
    if not ctx_path.exists():
        return None, None, None, "no context.json"
    try:
        ctx = json.loads(ctx_path.read_text())
    except Exception as e:
        return None, None, None, f"context.json inválido: {e}"
    lref = ctx.get("locked_reference") or {}
    oc = lref.get("original_coords") or {}
    p1 = oc.get("post1") or {}
    p2 = oc.get("post2") or {}
    tape1 = p1.get("tape_px")
    tape2 = p2.get("tape_px")
    if not tape1 or not tape2:
        return None, None, None, "tape_px ausente en context"
    central = find_central_frame(folder)
    if not central:
        return None, None, None, "sin frame central"
    img = cv2.imread(str(central))
    if img is None:
        return None, None, None, f"no se pudo leer {central.name}"
    r = coco_model(img, classes=[19], conf=0.2, verbose=False)
    if not r or len(r[0].boxes) == 0:
        return None, None, None, "YOLO no detectó vaca"
    boxes = r[0].boxes.xyxy.cpu().numpy()
    scores = r[0].boxes.conf.cpu().numpy()
    bi = int(np.argmax(scores))
    bx1, by1, bx2, by2 = boxes[bi]
    bbox_h = float(by2 - by1)
    tape_avg = (float(tape1) + float(tape2)) / 2.0
    cm_per_px = POSTE_REAL_CM / tape_avg
    altura_cm = bbox_h * cm_per_px
    return altura_cm, cm_per_px, bbox_h, None


def run_procesar(folder, altura_real, cow_name):
    """Llama a procesar_21_frames.py via subprocess."""
    env = os.environ.copy()
    env["MODELO_OUTPUT_DIR"] = "output_modelos3d_6mayo"
    cmd = [
        sys.executable,
        str(PROJECT / "procesar_21_frames.py"),
        str(folder),
        str(altura_real),
        cow_name,
    ]
    print(f"  → procesar_21_frames {folder.name} altura={altura_real}cm")
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"    [error] returncode={result.returncode}")
            print(result.stdout[-500:])
            print(result.stderr[-500:])
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    [error] timeout")
        return False


def load_resumen(cow_name):
    """Lee output_modelos3d_6mayo/<cow_name>/<cow_name>_resumen.json"""
    for fname in (f"{cow_name}_resumen.json", "resumen.json"):
        p = OUTPUT_DIR / cow_name / fname
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-3d", action="store_true",
                    help="No regenerar los PLY 3D, solo altura_calculada")
    ap.add_argument("--force", action="store_true",
                    help="Re-procesar aunque ya exista resumen.json")
    args = ap.parse_args()

    if not DATASET_DIR.is_dir():
        print(f"[fatal] no existe {DATASET_DIR}")
        sys.exit(1)

    from ultralytics import YOLO
    print("[init] cargando yolov8n.pt...")
    coco = YOLO(str(PROJECT / "yolov8n.pt"))

    folders = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())
    print(f"[init] {len(folders)} carpetas en {DATASET_DIR}\n")

    results = []
    for f in folders:
        altura_real, peso_real, suffix = parse_folder_name(f.name)
        print(f"=== {f.name} ===")
        if altura_real is None:
            print(f"  [skip] no parseable, no se procesa")
            results.append({
                "nombre": f.name,
                "altura_real_cm": None,
                "peso_real_kg": None,
                "altura_calculada_cm": None,
                "vol_barril_litros": None,
                "error": "nombre no parseable",
            })
            continue
        print(f"  altura_real={altura_real:.0f}cm peso_real={peso_real:.0f}kg suffix='{suffix}'")

        # 1) altura calculada
        altura_calc, cm_px, bbox_h, err_calc = compute_altura_calculada(f, coco)
        if altura_calc is not None:
            err_pct = (altura_calc - altura_real) / altura_real * 100
            print(f"  altura_calc={altura_calc:.1f}cm  (cm/px={cm_px:.4f}, bbox_h={bbox_h:.0f}px, Δ={err_pct:+.1f}%)")
        else:
            print(f"  altura_calc=— ({err_calc})")

        # 2) procesar 3D / leer vol_barril
        cow_name = f.name
        vol_barril = None
        if not args.skip_3d:
            resumen_path = OUTPUT_DIR / cow_name / f"{cow_name}_resumen.json"
            need_run = args.force or not resumen_path.exists()
            if need_run:
                ok = run_procesar(f, altura_real, cow_name)
                if not ok:
                    print(f"  [warn] procesar_21_frames falló")
        resumen = load_resumen(cow_name)
        if resumen:
            vol_barril = resumen.get("vol_barril_litros") or resumen.get("barril_consenso_L")
            if vol_barril:
                print(f"  vol_barril={vol_barril:.1f}L")

        results.append({
            "nombre": cow_name,
            "altura_real_cm": altura_real,
            "peso_real_kg": peso_real,
            "altura_calculada_cm": round(altura_calc, 1) if altura_calc else None,
            "vol_barril_litros": round(vol_barril, 1) if vol_barril else None,
            "error": err_calc,
        })
        print()

    # Tabla final
    print()
    print("=" * 112)
    print(f"{'NOMBRE':<30} {'A.REAL':>10} {'P.REAL':>10} {'A.CALC':>10} {'Δ%':>8} {'VOL.BARRIL':>14}")
    print("=" * 112)
    for r in results:
        ar = f"{r['altura_real_cm']:.0f}cm" if r['altura_real_cm'] else "—"
        pr = f"{r['peso_real_kg']:.0f}kg" if r['peso_real_kg'] else "—"
        ac = f"{r['altura_calculada_cm']:.1f}cm" if r['altura_calculada_cm'] else "—"
        if r['altura_real_cm'] and r['altura_calculada_cm']:
            d = (r['altura_calculada_cm'] - r['altura_real_cm']) / r['altura_real_cm'] * 100
            dpct = f"{d:+.1f}%"
        else:
            dpct = "—"
        vb = f"{r['vol_barril_litros']:.1f}L" if r['vol_barril_litros'] else "—"
        print(f"{r['nombre']:<30} {ar:>10} {pr:>10} {ac:>10} {dpct:>8} {vb:>14}")

    # CSV
    with RESULTS_CSV.open("w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=[
            "nombre", "altura_real_cm", "peso_real_kg",
            "altura_calculada_cm", "vol_barril_litros", "error",
        ])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nCSV guardado en: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
