"""
Modelado visual de vaca - Extrae métricas corporales desde fotos laterales.
Segmenta la vaca del fondo y calcula: área, largo, alto, proporciones.
"""

import cv2
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def segmentar_vaca(img_path):
    """
    Segmenta la vaca del fondo usando color (HSV) + morfología.
    La vaca Hereford es marrón/rojiza sobre pasto verde.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ERROR: No se pudo leer {img_path}")
        return None, None, None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Máscara para detectar la vaca (tonos marrones/rojizos)
    # Rango bajo de rojo-marrón
    lower_brown1 = np.array([0, 40, 40])
    upper_brown1 = np.array([25, 255, 200])
    mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)

    # Rango alto de rojo
    lower_brown2 = np.array([160, 40, 40])
    upper_brown2 = np.array([180, 255, 200])
    mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)

    # Blanco de la cara (Hereford)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Combinar máscaras
    mask = mask1 | mask2 | mask_white

    # Morfología para limpiar ruido y cerrar huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Encontrar el contorno más grande (la vaca)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"  WARN: No se encontró contorno en {img_path}")
        return None, None, None

    # Filtrar por área mínima y tomar el más grande
    min_area = img.shape[0] * img.shape[1] * 0.05  # al menos 5% de la imagen
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not contours:
        print(f"  WARN: Contornos muy pequeños en {img_path}")
        return None, None, None

    vaca_contour = max(contours, key=cv2.contourArea)

    # Crear máscara limpia solo con la vaca
    mask_final = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(mask_final, [vaca_contour], -1, 255, -1)

    return img, mask_final, vaca_contour


def extraer_metricas(img, mask, contour):
    """Extrae métricas del contorno de la vaca."""
    area_px = cv2.contourArea(contour)
    perimetro_px = cv2.arcLength(contour, True)

    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Bounding box rotado (ajustado al ángulo real)
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rect_w, rect_h), angle = rect
    largo = max(rect_w, rect_h)
    alto = min(rect_w, rect_h)

    # Proporción
    ratio_largo_alto = largo / alto if alto > 0 else 0

    # Solidez: qué tan "llena" está la forma
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidez = area_px / hull_area if hull_area > 0 else 0

    # Área relativa a la imagen
    img_area = img.shape[0] * img.shape[1]
    area_relativa = area_px / img_area

    # Compacidad (circularidad)
    compacidad = (4 * np.pi * area_px) / (perimetro_px ** 2) if perimetro_px > 0 else 0

    return {
        'area_px': int(area_px),
        'perimetro_px': round(perimetro_px, 1),
        'bbox_w': w,
        'bbox_h': h,
        'largo_px': round(largo, 1),
        'alto_px': round(alto, 1),
        'ratio_largo_alto': round(ratio_largo_alto, 3),
        'solidez': round(solidez, 3),
        'compacidad': round(compacidad, 3),
        'area_relativa': round(area_relativa, 4),
        'img_width': img.shape[1],
        'img_height': img.shape[0],
    }


def generar_visualizacion(img, mask, contour, metricas, nombre, output_path):
    """Genera imagen con la segmentación superpuesta."""
    vis = img.copy()

    # Overlay verde semitransparente sobre la máscara
    overlay = vis.copy()
    overlay[mask > 0] = [0, 255, 0]
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    # Dibujar contorno
    cv2.drawContours(vis, [contour], -1, (0, 0, 255), 2)

    # Bounding box rotado
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(vis, [box], 0, (255, 0, 0), 2)

    # Texto con métricas
    y_text = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    for key in ['area_px', 'largo_px', 'alto_px', 'ratio_largo_alto', 'solidez']:
        text = f"{key}: {metricas[key]}"
        cv2.putText(vis, text, (10, y_text), font, 0.4, (255, 255, 255), 1)
        y_text += 18

    cv2.imwrite(output_path, vis)


def main():
    base_dir = Path(__file__).parent / "checkpoints" / "dataset" / "modelo" / "vaca1"
    output_dir = Path(__file__).parent / "output_modelo_vaca1"
    output_dir.mkdir(exist_ok=True)

    fotos = sorted([f for f in base_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    print(f"Encontradas {len(fotos)} fotos en {base_dir}\n")

    todas_metricas = []

    for i, foto in enumerate(fotos):
        print(f"[{i+1}/{len(fotos)}] Procesando: {foto.name}")
        img, mask, contour = segmentar_vaca(str(foto))

        if img is None or mask is None:
            print("  -> Saltando (no se pudo segmentar)\n")
            continue

        metricas = extraer_metricas(img, mask, contour)
        metricas['foto'] = foto.name
        todas_metricas.append(metricas)

        print(f"  Area: {metricas['area_px']} px | Largo: {metricas['largo_px']} | Alto: {metricas['alto_px']} | Ratio: {metricas['ratio_largo_alto']}")

        # Guardar visualización
        vis_path = output_dir / f"seg_{i+1:02d}_{foto.stem}.png"
        generar_visualizacion(img, mask, contour, metricas, foto.name, str(vis_path))
        print(f"  -> Guardada: {vis_path.name}\n")

    if not todas_metricas:
        print("ERROR: No se pudo procesar ninguna foto.")
        return

    # === PERFIL PROMEDIO ===
    print("=" * 60)
    print("PERFIL DE VACA 1 (promedio de {} fotos)".format(len(todas_metricas)))
    print("=" * 60)

    campos_numericos = ['area_px', 'largo_px', 'alto_px', 'ratio_largo_alto',
                        'solidez', 'compacidad', 'area_relativa']

    perfil = {}
    for campo in campos_numericos:
        valores = [m[campo] for m in todas_metricas]
        perfil[campo] = {
            'promedio': round(np.mean(valores), 3),
            'std': round(np.std(valores), 3),
            'min': round(min(valores), 3),
            'max': round(max(valores), 3),
        }
        print(f"  {campo:20s}: {perfil[campo]['promedio']:>10} (±{perfil[campo]['std']})")

    # Guardar resultados
    resultado = {
        'vaca': 'vaca1',
        'num_fotos_procesadas': len(todas_metricas),
        'perfil_promedio': perfil,
        'metricas_por_foto': todas_metricas,
    }

    json_path = output_dir / "perfil_vaca1.json"
    with open(json_path, 'w') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en: {json_path}")

    # === GRÁFICO RESUMEN ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Perfil Visual - Vaca 1', fontsize=14, fontweight='bold')

    indices = range(1, len(todas_metricas) + 1)

    # Área
    axes[0, 0].bar(indices, [m['area_px'] for m in todas_metricas], color='#8B4513')
    axes[0, 0].set_title('Área (píxeles)')
    axes[0, 0].set_xlabel('Foto')
    axes[0, 0].axhline(y=perfil['area_px']['promedio'], color='red', linestyle='--', label='Promedio')
    axes[0, 0].legend()

    # Largo vs Alto
    axes[0, 1].bar(indices, [m['largo_px'] for m in todas_metricas], color='#CD853F', label='Largo')
    axes[0, 1].bar(indices, [-m['alto_px'] for m in todas_metricas], color='#DEB887', label='Alto')
    axes[0, 1].set_title('Largo vs Alto (píxeles)')
    axes[0, 1].set_xlabel('Foto')
    axes[0, 1].legend()

    # Ratio largo/alto
    axes[1, 0].plot(indices, [m['ratio_largo_alto'] for m in todas_metricas], 'o-', color='#8B4513')
    axes[1, 0].set_title('Ratio Largo/Alto')
    axes[1, 0].set_xlabel('Foto')
    axes[1, 0].axhline(y=perfil['ratio_largo_alto']['promedio'], color='red', linestyle='--')

    # Solidez y compacidad
    axes[1, 1].plot(indices, [m['solidez'] for m in todas_metricas], 'o-', color='#8B4513', label='Solidez')
    axes[1, 1].plot(indices, [m['compacidad'] for m in todas_metricas], 's-', color='#CD853F', label='Compacidad')
    axes[1, 1].set_title('Solidez y Compacidad')
    axes[1, 1].set_xlabel('Foto')
    axes[1, 1].legend()

    plt.tight_layout()
    chart_path = output_dir / "grafico_perfil_vaca1.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Gráfico guardado en: {chart_path}")


if __name__ == '__main__':
    main()
