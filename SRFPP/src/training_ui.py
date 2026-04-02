"""
Funciones para ejecutar entrenamiento desde la UI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .dataset_manager import get_artifacts_base_dir, get_establecimiento_dir
from .training_status import clear_training_status


def train_model(
    establecimiento: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    img_size: int = 224,
    val_frac: float = 0.2,
    filter_with_yolo: bool = True,
    yolo_min_confidence: float = 0.3,
    yolo_require_features: list[str] | None = None,
    balance_dataset: bool = True,
    balance_similarity_threshold: float = 0.85,
    crop_to_face: bool = True,
) -> subprocess.Popen:
    """Inicia el entrenamiento de un modelo en background."""
    data_dir = get_establecimiento_dir(establecimiento)
    artifacts_dir = get_artifacts_base_dir() / establecimiento
    
    # Limpiar estado anterior si existe
    clear_training_status(establecimiento)
    
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--data_dir",
        str(data_dir),
        "--artifacts_dir",
        str(artifacts_dir),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--lr",
        str(lr),
        "--img_size",
        str(img_size),
        "--val_frac",
        str(val_frac),
    ]
    
    # Agregar opciones de filtrado YOLO si están habilitadas
    if filter_with_yolo:
        cmd.extend(["--filter_with_yolo"])
        cmd.extend(["--yolo_min_confidence", str(yolo_min_confidence)])
        if yolo_require_features:
            cmd.extend(["--yolo_require_features"] + yolo_require_features)
    
    # Agregar opciones de balanceo si están habilitadas
    if balance_dataset:
        cmd.extend(["--balance_dataset"])
        cmd.extend(["--balance_similarity_threshold", str(balance_similarity_threshold)])

    # Agregar opción de recorte facial
    if crop_to_face:
        cmd.extend(["--crop_to_face"])

    # Ejecutar en background pero mostrar output en terminal
    # Usar None para stdout/stderr hace que se muestre en la terminal actual
    process = subprocess.Popen(
        cmd,
        stdout=None,  # Mostrar en terminal
        stderr=None,  # Mostrar errores en terminal
        text=True,
    )
    
    return process

