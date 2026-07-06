"""Recomputa altura_calculada usando /detect_cow_fast (la misma pipeline que
ve el usuario en la UI: YOLO + silueta_seg + barril_seg + interpolación por X).

Usa el frame central de cada carpeta en checkpoints/6mayo/ y el locked_reference
del context.json. Actualiza diagnostico_6mayo.csv en su lugar.

Requiere Flask corriendo en http://localhost:5001
"""
import csv
import json
import re
from pathlib import Path
import requests

PROJECT = Path(__file__).parent
DATASET_DIR = PROJECT / "checkpoints" / "6mayo"
RESULTS_CSV = PROJECT / "diagnostico_6mayo.csv"
FLASK_URL = "http://localhost:5001"


def parse_folder_name(name):
    m = re.match(r"^(\d+)_(\d+)(?:_(.*))?$", name)
    if not m:
        return None, None, name
    return float(m.group(1)), float(m.group(2)), (m.group(3) or "")


def find_central_frame(folder):
    for p in folder.glob("frame_000_*.jpg"):
        return p
    return None


def detect_cow_fast(central_frame, locked_reference):
    with open(central_frame, "rb") as f:
        files = {"frame": (central_frame.name, f, "image/jpeg")}
        data = {"locked_reference_json": json.dumps(locked_reference)}
        r = requests.post(f"{FLASK_URL}/detect_cow_fast",
                          files=files, data=data, timeout=60)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    j = r.json()
    if not j.get("success"):
        return None, j.get("error", "no success")
    if not j.get("detected"):
        return None, "no cow detected"
    return j, None


def main():
    # Leer CSV existente para no perder los volúmenes
    existing = {}
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open() as cf:
            for row in csv.DictReader(cf):
                existing[row["nombre"]] = row

    folders = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())
    print(f"[init] {len(folders)} carpetas\n")

    rows = []
    for f in folders:
        nombre = f.name
        altura_real, peso_real, _ = parse_folder_name(nombre)
        print(f"=== {nombre} ===")

        # Cargar context.json
        ctx_path = f / "context.json"
        if not ctx_path.exists():
            print(f"  [skip] sin context.json")
            rows.append(existing.get(nombre, {"nombre": nombre,
                "altura_real_cm": altura_real, "peso_real_kg": peso_real,
                "altura_calculada_cm": None, "vol_barril_litros": None,
                "error": "sin context.json"}))
            continue
        try:
            ctx = json.loads(ctx_path.read_text())
        except Exception as e:
            print(f"  [skip] context.json inválido: {e}")
            rows.append(existing.get(nombre, {"nombre": nombre, "error": str(e)}))
            continue
        lref = ctx.get("locked_reference")
        if not lref:
            print(f"  [skip] sin locked_reference")
            rows.append(existing.get(nombre, {"nombre": nombre,
                "altura_real_cm": altura_real, "peso_real_kg": peso_real,
                "altura_calculada_cm": None,
                "vol_barril_litros": existing.get(nombre, {}).get("vol_barril_litros"),
                "error": "sin locked_reference"}))
            continue

        central = find_central_frame(f)
        if not central:
            print(f"  [skip] sin frame central")
            rows.append(existing.get(nombre, {"nombre": nombre, "error": "sin frame central"}))
            continue

        result, err = detect_cow_fast(central, lref)
        altura_calc = None
        within_label = None
        if result is None:
            print(f"  [error] {err}")
            err_msg = err
        else:
            altura_calc = result.get("cow_height_cm")
            within = result.get("within_rectangle")
            within_label = "✓" if within else "fuera-rect"
            cm_px = result.get("cm_per_px")
            bbox = result.get("animal_bbox_original") or []
            # Si no estaba dentro del rectángulo, /detect_cow_fast omite la altura,
            # pero podemos recomputarla con cm_per_px + bbox para fines de diagnóstico.
            if altura_calc is None and cm_px and len(bbox) == 4:
                bbox_h = bbox[3] - bbox[1]
                altura_calc = float(bbox_h * cm_px)
            err_msg = None if altura_calc is not None else "no se pudo calcular"
            if altura_calc is not None:
                d = (altura_calc - altura_real) / altura_real * 100 if altura_real else 0
                print(f"  altura_calc={altura_calc:.1f}cm  cm/px={cm_px:.4f}  Δ={d:+.1f}%  ({within_label})")
            else:
                print(f"  altura_calc=— ({err_msg or 'sin datos'})")

        # Vol barril (de CSV anterior)
        vol_barril = None
        prev = existing.get(nombre, {})
        if prev.get("vol_barril_litros"):
            try:
                vol_barril = float(prev["vol_barril_litros"])
            except ValueError:
                vol_barril = None

        rows.append({
            "nombre": nombre,
            "altura_real_cm": altura_real,
            "peso_real_kg": peso_real,
            "altura_calculada_cm": round(altura_calc, 1) if altura_calc else None,
            "within_rectangle": within_label or "",
            "vol_barril_litros": vol_barril,
            "error": err_msg,
        })

    # Imprimir tabla
    print()
    print("=" * 112)
    print(f"{'NOMBRE':<30} {'A.REAL':>10} {'P.REAL':>10} {'A.CALC':>10} {'Δ%':>8} {'CRUCE':>10} {'VOL.BARRIL':>14}")
    print("=" * 122)
    for r in rows:
        ar = f"{r['altura_real_cm']:.0f}cm" if r.get('altura_real_cm') else "—"
        pr = f"{r['peso_real_kg']:.0f}kg" if r.get('peso_real_kg') else "—"
        ac = f"{r['altura_calculada_cm']:.1f}cm" if r.get('altura_calculada_cm') else "—"
        wr = r.get('within_rectangle') or "—"
        if r.get('altura_real_cm') and r.get('altura_calculada_cm'):
            d = (r['altura_calculada_cm'] - r['altura_real_cm']) / r['altura_real_cm'] * 100
            dpct = f"{d:+.1f}%"
        else:
            dpct = "—"
        vb = f"{r['vol_barril_litros']:.1f}L" if r.get('vol_barril_litros') else "—"
        print(f"{r['nombre']:<30} {ar:>10} {pr:>10} {ac:>10} {dpct:>8} {wr:>10} {vb:>14}")

    # Reescribir CSV
    with RESULTS_CSV.open("w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=[
            "nombre", "altura_real_cm", "peso_real_kg",
            "altura_calculada_cm", "within_rectangle",
            "vol_barril_litros", "error",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nCSV actualizado en: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
