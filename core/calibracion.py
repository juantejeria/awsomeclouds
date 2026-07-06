"""Constantes de calibración — ÚNICA fuente de verdad.

Los valores viven en config.ini, sección [calibracion]. Este módulo los lee
una sola vez al importarse y los expone como constantes para todo el backend.
El frontend (engine.js) los recibe vía template (window.VARA_CM en index.html).

Si config.ini no existe o falta la sección, se usan los defaults históricos,
así los scripts CLI siguen funcionando fuera del servidor.

NO hardcodear estos valores en otros archivos: importar desde acá.
    from core.calibracion import VARA_CM, K_DEPTH, DENSIDAD_KG_L
"""
import configparser
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_cfg = configparser.ConfigParser()
_cfg.read(_ROOT / 'config.ini')

#: Altura real de la vara/cinta roja de referencia (cm). Escala px→cm.
VARA_CM = _cfg.getfloat('calibracion', 'vara_cm', fallback=110.0)

#: Ratio profundidad/altura del corte elíptico del barril (modelo 3D).
K_DEPTH = _cfg.getfloat('calibracion', 'k_depth', fallback=0.25)

#: Densidad del animal (kg/L): peso_kg = volumen_L * DENSIDAD_KG_L.
DENSIDAD_KG_L = _cfg.getfloat('calibracion', 'densidad_kg_l', fallback=1.03)
