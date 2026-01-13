from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_establecimientos(artifacts_base_dir: str | Path = "artifacts") -> list[str]:
    """Lista los establecimientos disponibles (carpetas con model.pt dentro de artifacts_base_dir)."""
    base = Path(artifacts_base_dir)
    if not base.exists():
        return []
    
    establecimientos = []
    for item in base.iterdir():
        if item.is_dir():
            # Verificar que tenga los archivos necesarios
            if (item / "model.pt").exists() and (item / "classes.json").exists():
                establecimientos.append(item.name)
    
    return sorted(establecimientos)


