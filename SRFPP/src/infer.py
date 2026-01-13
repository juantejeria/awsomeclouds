from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Configurar variables de entorno ANTES de importar PyTorch (previene crashes en macOS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .io_utils import read_json
from .model import build_model, predict_proba
from .video_utils import iter_video_frames


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    decision: Literal["known", "unknown"]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_artifacts(artifacts_dir: str | Path):
    artifacts_dir = Path(artifacts_dir)
    classes = read_json(artifacts_dir / "classes.json")
    config = read_json(artifacts_dir / "config.json")

    model = build_model(num_classes=len(classes), pretrained=False)
    state = torch.load(artifacts_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dev = _device()
    model.to(dev)

    img_size = int(config.get("img_size", 224))
    # Inference transforms: deterministic resize/center-crop + ImageNet normalization
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, classes, tfm, dev


def predict_image(
    model,
    classes: list[str],
    tfm,
    dev: torch.device,
    image: Image.Image,
    threshold: float = 0.70,
) -> Prediction:
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = tfm(image).unsqueeze(0).to(dev)
    probs = predict_proba(model, x)[0].detach().cpu().numpy()
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = classes[idx]
    decision: Literal["known", "unknown"] = "known" if conf >= threshold else "unknown"
    return Prediction(label=label, confidence=conf, decision=decision)


def predict_video(
    model,
    classes: list[str],
    tfm,
    dev: torch.device,
    video_path: str,
    threshold: float = 0.70,
    stride: int = 10,
    max_frames: int = 150,
) -> Prediction:
    all_probs = []
    for fr in iter_video_frames(video_path, stride=stride, max_frames=max_frames):
        img = Image.fromarray(fr.rgb)
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = tfm(img).unsqueeze(0).to(dev)
        probs = predict_proba(model, x)[0].detach().cpu().numpy()
        all_probs.append(probs)

    if not all_probs:
        return Prediction(label="(sin frames)", confidence=0.0, decision="unknown")

    mean_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    idx = int(np.argmax(mean_probs))
    conf = float(mean_probs[idx])
    label = classes[idx]
    decision: Literal["known", "unknown"] = "known" if conf >= threshold else "unknown"
    return Prediction(label=label, confidence=conf, decision=decision)


