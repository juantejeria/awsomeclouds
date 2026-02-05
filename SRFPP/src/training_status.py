"""
Funciones para leer el estado del entrenamiento desde la UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime

from .dataset_manager import get_artifacts_base_dir
from .io_utils import read_json, write_json


def get_training_status(establecimiento: str) -> dict[str, Any] | None:
    """Obtiene el estado actual del entrenamiento de un establecimiento."""
    artifacts_dir = get_artifacts_base_dir() / establecimiento
    progress_file = artifacts_dir / "training_progress.json"
    
    if not progress_file.exists():
        return None
    
    try:
        return read_json(progress_file)
    except Exception:
        return None


def is_training_running(establecimiento: str) -> bool:
    """Verifica si hay un entrenamiento en curso."""
    status = get_training_status(establecimiento)
    if status is None:
        return False
    return status.get("status") in ["running", "initializing"]


def clear_training_status(establecimiento: str) -> None:
    """Limpia el archivo de progreso (útil cuando se inicia un nuevo entrenamiento)."""
    artifacts_dir = get_artifacts_base_dir() / establecimiento
    progress_file = artifacts_dir / "training_progress.json"
    
    if progress_file.exists():
        progress_file.unlink()


def load_training_history(establecimiento: str) -> list[dict[str, Any]]:
    """Carga el historial de entrenamientos."""
    artifacts_dir = get_artifacts_base_dir() / establecimiento
    history_file = artifacts_dir / "training_history.json"
    if not history_file.exists():
        return []
    try:
        data = read_json(history_file)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def append_training_history(establecimiento: str, entry: dict[str, Any]) -> None:
    """Agrega una entrada al historial de entrenamientos."""
    artifacts_dir = get_artifacts_base_dir() / establecimiento
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    history_file = artifacts_dir / "training_history.json"
    history = load_training_history(establecimiento)
    entry = dict(entry)
    entry.setdefault("completed_at", datetime.utcnow().isoformat())
    history.insert(0, entry)
    history = history[:100]
    write_json(history_file, history)


def save_training_history(establecimiento: str, history: list[dict[str, Any]]) -> None:
    """Guarda el historial de entrenamientos."""
    artifacts_dir = get_artifacts_base_dir() / establecimiento
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    history_file = artifacts_dir / "training_history.json"
    write_json(history_file, history)

