"""
Prepara el dataset de entrenamiento para silueta completa.
Copia frames + mascaras GrabCut desde _barril_training como punto de partida,
y genera el indice para el editor de silueta.
"""
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).parent
BARRIL_DIR = PROJECT / 'output_modelos3d_grandes' / '_barril_training'
SILUETA_DIR = PROJECT / 'output_modelos3d_grandes' / '_silueta_training'
BARRIL_INDEX = BARRIL_DIR / 'frames_index.json'
SILUETA_INDEX = SILUETA_DIR / 'frames_index.json'


def main():
    SILUETA_DIR.mkdir(parents=True, exist_ok=True)

    with open(BARRIL_INDEX) as f:
        barril_frames = json.load(f)

    silueta_frames = []
    copied_img = 0
    copied_mask = 0

    for frame in barril_frames:
        fid = frame['id']

        # Copiar imagen original
        src_img = BARRIL_DIR / frame['img']
        dst_img = SILUETA_DIR / frame['img']
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)
            copied_img += 1

        # Copiar mascara GrabCut (silueta completa original) como base
        src_mask = BARRIL_DIR / frame['mask']
        dst_mask = SILUETA_DIR / frame['mask']
        if src_mask.exists() and not dst_mask.exists():
            shutil.copy2(src_mask, dst_mask)
            copied_mask += 1

        # Crear entrada para silueta - todos empiezan como pending
        silueta_frames.append({
            'id': fid,
            'individuo': frame['individuo'],
            'video': frame.get('video', ''),
            'frame_idx': frame.get('frame_idx', 0),
            'bbox': frame.get('bbox', []),
            'crop_w': frame.get('crop_w', 0),
            'crop_h': frame.get('crop_h', 0),
            'img': frame['img'],
            'mask': frame['mask'],  # mascara GrabCut original como base
            'status': 'pending',
            'cuts': [],
            'brush_rle': [],
        })

    with open(SILUETA_INDEX, 'w') as f:
        json.dump(silueta_frames, f, indent=2, ensure_ascii=False)

    print(f"\nDataset silueta preparado en: {SILUETA_DIR}")
    print(f"  Frames totales: {len(silueta_frames)}")
    print(f"  Imagenes copiadas: {copied_img}")
    print(f"  Mascaras copiadas: {copied_mask}")
    print(f"  Indice: {SILUETA_INDEX}")
    print(f"\nTodos los frames empiezan como 'pending'.")
    print(f"Usa editor_silueta_training.py para corregir las mascaras.")


if __name__ == '__main__':
    main()
