"""
Elimina las fotos de 26marz y desfile26marz del dataset de silueta training
para poder re-generarlas con mejor calidad (mas padding, PNG, filtro resolucion).
"""
import json
import os
from pathlib import Path

SILUETA_DIR = Path(__file__).parent / 'output_modelos3d_grandes' / '_silueta_training'
INDEX_FILE = SILUETA_DIR / 'frames_index.json'

SOURCES_TO_REMOVE = {'26marz', 'desfile26marz'}


def main():
    with open(INDEX_FILE) as f:
        frames = json.load(f)

    keep = []
    removed = 0

    for frame in frames:
        source = frame.get('source', '')
        if source in SOURCES_TO_REMOVE:
            # Borrar archivos
            img_path = SILUETA_DIR / frame['img']
            mask_path = SILUETA_DIR / frame['mask']
            for p in [img_path, mask_path]:
                if p.exists():
                    os.remove(p)
            removed += 1
        else:
            keep.append(frame)

    with open(INDEX_FILE, 'w') as f:
        json.dump(keep, f, indent=2, ensure_ascii=False)

    print(f"Eliminados: {removed} frames de {SOURCES_TO_REMOVE}")
    print(f"Quedan: {len(keep)} frames en el dataset")


if __name__ == '__main__':
    main()
