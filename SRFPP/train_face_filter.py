"""
Entrena un clasificador binario (rostro_bueno vs fake) usando los frames
etiquetados de vaca_1.

- Positivos: frames sin "fake" en el nombre
- Negativos: frames con "fake" en el nombre (orejas, postes, objetos)

Usa transfer learning con MobileNetV2 (ligero y rapido).
Guarda el modelo en artifacts/face_filter/face_filter.pt
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

VACA1_DIR = Path(__file__).resolve().parent / "data" / "Entrga_Reconocimiento" / "vaca_1"
OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "face_filter"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 30
LR = 0.001


class FaceFilterDataset(Dataset):
    def __init__(self, img_dir: Path, transform=None):
        self.samples = []  # (path, label)  label: 1=good, 0=fake
        self.transform = transform

        for f in sorted(img_dir.glob("*.png")):
            is_fake = "fake" in f.name.lower()
            self.samples.append((f, 0 if is_fake else 1))

        print(f"Dataset: {len(self.samples)} samples "
              f"({sum(1 for _, l in self.samples if l == 1)} good, "
              f"{sum(1 for _, l in self.samples if l == 0)} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model():
    """MobileNetV2 con cabeza binaria."""
    base = models.mobilenet_v2(pretrained=True)
    # Congelar las primeras capas
    for i, param in enumerate(base.features.parameters()):
        if i < 40:  # congelar ~70% de las capas
            param.requires_grad = False
    base.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(base.last_channel, 1),
    )
    return base


def main():
    # Data augmentation fuerte para compensar pocas muestras
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = FaceFilterDataset(VACA1_DIR, transform=train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([22.0 / 74.0]).to(device)  # balancear clases
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Entrenar
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(imgs).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        acc = correct / total * 100
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.1f}%")

    # Evaluar en el mismo dataset sin augmentation
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_dataset = FaceFilterDataset(VACA1_DIR, transform=eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    fp = 0  # false positives (fake clasificado como good)
    fn = 0  # false negatives (good clasificado como fake)
    with torch.no_grad():
        for imgs, labels in eval_loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            outputs = model(imgs).squeeze(-1)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    print(f"\nEvaluacion final: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  False positives (fakes aceptados): {fp}")
    print(f"  False negatives (buenos rechazados): {fn}")

    # Guardar modelo
    save_path = OUTPUT_DIR / "face_filter.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nModelo guardado en: {save_path}")


if __name__ == "__main__":
    main()
