"""
Detecta franjas rojas de 50cm en las fotos para calibrar escala.
Muestra todas las detecciones para que el usuario confirme cuál es la correcta.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def detectar_rojo(img):
    """Detecta píxeles rojos con rangos amplios."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rojo bajo (0-15)
    mask1 = cv2.inRange(hsv, np.array([0, 50, 30]), np.array([15, 255, 255]))
    # Rojo alto (160-180)
    mask2 = cv2.inRange(hsv, np.array([160, 50, 30]), np.array([180, 255, 255]))

    mask = mask1 | mask2

    # Limpiar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def encontrar_franjas_verticales(mask, img, min_height=12, min_aspect=0.8):
    """Encuentra objetos rojos verticales (candidatos a franja de 50cm)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        if h < min_height:
            continue
        if area < 20:
            continue

        aspect = h / w if w > 0 else 0

        # Queremos objetos más altos que anchos (verticales)
        if aspect >= min_aspect:
            # Calcular qué porcentaje del bbox es rojo
            roi_mask = mask[y:y+h, x:x+w]
            fill = np.sum(roi_mask > 0) / (w * h) if w * h > 0 else 0

            candidatos.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect': round(aspect, 2),
                'height_px': h,
                'width_px': w,
                'fill': round(fill, 2),
                'contour': c,
            })

    # Ordenar por altura (más altos primero)
    candidatos.sort(key=lambda c: c['height_px'], reverse=True)
    return candidatos


def main():
    project = Path(__file__).parent
    fotos_dir = project / "checkpoints" / "dataset" / "modelo" / "vaca1"
    output_dir = project / "output_franjas"
    output_dir.mkdir(exist_ok=True)

    fotos = sorted([f for f in fotos_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])

    print(f"Buscando franjas rojas en {len(fotos)} fotos...\n")

    for i, foto in enumerate(fotos):
        img = cv2.imread(str(foto))
        if img is None:
            continue

        mask_rojo = detectar_rojo(img)
        candidatos = encontrar_franjas_verticales(mask_rojo, img)

        print(f"[{i+1}] {foto.name}: {len(candidatos)} candidatos rojos verticales")

        if not candidatos:
            print("    (ninguno)\n")
            continue

        for j, c in enumerate(candidatos[:8]):
            print(f"    #{j+1}: pos=({c['bbox'][0]},{c['bbox'][1]}) "
                  f"size={c['width_px']}x{c['height_px']}px "
                  f"aspect={c['aspect']} fill={c['fill']}")

        # Visualización
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{foto.name} - Franjas rojas detectadas', fontsize=11)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Original
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')

        # Máscara roja
        axes[1].imshow(mask_rojo, cmap='Reds')
        axes[1].set_title('Píxeles rojos detectados')
        axes[1].axis('off')

        # Original + candidatos marcados
        axes[2].imshow(img_rgb)
        colors_list = ['lime', 'cyan', 'yellow', 'magenta', 'white', 'orange', 'blue', 'red']
        for j, c in enumerate(candidatos[:8]):
            x, y, w, h = c['bbox']
            color = colors_list[j % len(colors_list)]
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
            axes[2].add_patch(rect)
            axes[2].text(x + w + 2, y + h // 2,
                        f"#{j+1}: {h}px",
                        color=color, fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        axes[2].set_title(f'{len(candidatos)} candidatos (top 8)')
        axes[2].axis('off')

        plt.tight_layout()
        out_path = output_dir / f"franjas_{i+1:02d}_{foto.stem}.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close()
        print()

    print(f"\nVisualizaciones guardadas en: {output_dir}/")
    print("Revisá las imágenes y decime cuál franja es la de 50cm.")


if __name__ == '__main__':
    main()
