"""
Recorte de "cresta" del LOMO en las mallas 3D de vacas.

Las mallas se generan espejando la silueta lateral con profundidad elíptica
(ver generar_modelos3d_grandes.py: profundidad_eliptica + guardar_ply). Eso hace
que el espesor (z) tienda a 0 en el borde superior, dejando el lomo como un FILO
afilado (cada sección transversal es una "almendra" con punta arriba). Ese filo
infla el yMax/alto/perímetro de las secciones sin aportar volumen.

`trim_top_crest` baja SOLO la punta superior de cada tramo (eje X) hasta la altura
donde la sección todavía tiene ancho real. No toca la panza (parte inferior).
No borra vértices (solo mueve hacia abajo los del filo) → no deja huecos.

Se usa en dos lugares (solución integral):
  - recortar_cresta_ply.py : limpia los *_3d.ply existentes.
  - generar_modelos3d_grandes.py : se aplica antes de guardar (modelos nuevos).
"""
import numpy as np


def trim_top_crest(V, frac=0.18, nbins=40, smooth=2, minpts=10):
    """Devuelve una copia de V (Nx3, columnas x=largo, y=alto, z=ancho) con la
    cresta del lomo recortada.

    Para cada tramo en X se calcula la "altura de lomo real" = el punto más alto
    que todavía tiene medio-ancho >= frac * (medio-ancho máximo del tramo). Los
    vértices por encima de esa altura (el filo, con ancho ~0) se bajan hasta ahí.

    frac   : umbral de ancho (fracción del ancho máximo del tramo) que define el
             borde del lomo "real". Más alto = recorta más agresivo.
    nbins  : tramos a lo largo del cuerpo para perfilar el lomo.
    smooth : suavizado (media móvil de 2*smooth+1 tramos) del perfil del lomo.
    """
    V = np.asarray(V, float).copy()
    if len(V) == 0:
        return V
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    xmin, xmax = x.min(), x.max()
    L = xmax - xmin
    if L <= 0:
        return V

    edges = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ylomo = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (x >= edges[i]) & (x <= edges[i + 1])
        # Saltear tramos con pocos vértices (caen entre anillos): se rellenan por
        # interpolación. Evita que un tramo ralo capture un vértice ancho de la
        # PANZA y tire el lomo al piso (sobre-recorte).
        if m.sum() < minpts:
            continue
        hw = np.abs(z[m])
        W = hw.max()
        if W <= 0:
            continue
        thr = frac * W
        wide = m & (np.abs(z) >= thr)
        if wide.sum() < 3:
            continue
        ylomo[i] = y[wide].max()  # punto más alto que aún tiene ancho real

    valid = ~np.isnan(ylomo)
    if valid.sum() < 2:
        return V
    ylomo = np.interp(centers, centers[valid], ylomo[valid])
    if smooth > 0:
        k = 2 * smooth + 1
        ylomo = np.convolve(np.pad(ylomo, smooth, mode='edge'),
                            np.ones(k) / k, mode='valid')

    ycap = np.interp(x, centers, ylomo)
    above = y > ycap
    V[above, 1] = ycap[above]
    return V
