# Backup: Sistema de Silueta Completa basado en GrabCut

**Fecha backup**: 2026-04-07  
**Archivo**: `generar_modelos3d_grandes_backup_silueta_grabcut.py`  
**Original**: `generar_modelos3d_grandes.py` (2084 lineas)

## Que contiene

El sistema de silueta completa (area verde) basado en heuristicas, sin modelo entrenado.

### Funciones clave

| Funcion | Lineas | Descripcion |
|---------|--------|-------------|
| `refinar_patas()` | 62-247 | Limpia pasto/sombra entre patas usando morfologia vertical, proyeccion y deteccion de picos |
| `eliminar_sombra()` | 250-486 | Remueve sombras usando luminosidad LAB, cromaticidad del suelo, y recorte lateral |
| `segmentar()` | 554-637 | Pipeline principal: GrabCut 2-pass con seeds inteligentes + post-procesamiento |

### Pipeline

```
Imagen + bbox
  -> GrabCut pass 1 (rectangulo)
  -> Smart seeding (FGD centro cuerpo, BGD bordes/suelo, PR_BGD zona patas)
  -> GrabCut pass 2 (mascara)
  -> Morfologia (close + open)
  -> Evaluacion leg_fill (>= 25% -> necesita limpieza)
  -> Si necesita: eliminar_sombra() + refinar_patas()
  -> mask_full, contorno_full
```

### Problemas conocidos

1. **Lomo cortado**: GrabCut conservador en bordes superiores cuando pelaje se confunde con fondo
2. **Patas cruzadas**: Proyeccion vertical fusiona patas cercanas en un solo pico
3. **Espacio entre patas**: Deteccion de sombra falla cuando color suelo ~ color animal

### Como restaurar

Para volver a usar este sistema, reemplazar `generar_modelos3d_grandes.py` con el backup:
```bash
cp generar_modelos3d_grandes_backup_silueta_grabcut.py generar_modelos3d_grandes.py
```
