from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Configurar variables de entorno ANTES de importar PyTorch (previene crashes en macOS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from .io_utils import ensure_dir, write_json
from .dataset_manager import get_establecimiento_display_name
from .model import build_model
from .filtered_dataset import FilteredImageFolder


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_splits(ds: datasets.ImageFolder, val_frac: float, seed: int):
    """
    Crea splits de entrenamiento y validación.
    Usa split estratificado si todas las clases tienen al menos 2 muestras,
    sino usa split aleatorio simple.
    """
    targets = [y for _, y in ds.samples]
    idxs = list(range(len(targets)))
    
    # Verificar si todas las clases tienen al menos 2 muestras
    from collections import Counter
    class_counts = Counter(targets)
    min_class_count = min(class_counts.values())
    
    if min_class_count >= 2:
        # Split estratificado (mejor para validación balanceada)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_idx, val_idx = next(splitter.split(idxs, targets))
    else:
        # Split aleatorio simple (cuando hay clases con muy pocas muestras)
        import numpy as np
        np.random.seed(seed)
        n_val = max(1, int(len(idxs) * val_frac))
        shuffled = np.random.permutation(idxs)
        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]
        print(f"⚠️  Advertencia: Algunas clases tienen menos de 2 muestras. Usando split aleatorio simple.")
        print(f"   Distribución por clase: {dict(class_counts)}")
    
    return Subset(ds, train_idx.tolist()), Subset(ds, val_idx.tolist())


def train_one_epoch(model, loader, optim, loss_fn, dev: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(dev)
        y = y.to(dev)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()

        total_loss += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.inference_mode()
def eval_one_epoch(model, loader, loss_fn, dev: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(dev)
        y = y.to(dev)
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Ruta a data/cows (ImageFolder)")
    p.add_argument("--artifacts_dir", type=str, default="artifacts", help="Salida del modelo")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--filter_with_yolo", action="store_true", help="Filtrar imágenes sin detecciones YOLO antes de entrenar")
    p.add_argument("--yolo_min_confidence", type=float, default=0.3, help="Confianza mínima para detecciones YOLO")
    p.add_argument("--yolo_require_features", nargs="+", choices=["cow", "animal"], help="Características requeridas para YOLO (ej: --yolo_require_features cow)")
    p.add_argument("--balance_dataset", action="store_true", help="Balancear dataset eliminando frames similares antes de entrenar")
    p.add_argument("--balance_similarity_threshold", type=float, default=0.85, help="Umbral de similitud para balanceo (0-1, más alto = más estricto)")
    args = p.parse_args()

    dev = _device()
    artifacts_dir = ensure_dir(args.artifacts_dir)
    estab_name = artifacts_dir.name
    display_name_snapshot = get_establecimiento_display_name(estab_name)

    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfm = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Balancear dataset si se solicita (antes de cargar)
    original_data_dir = args.data_dir
    if args.balance_dataset:
        print("\n⚖️  Balanceando dataset eliminando frames similares...")
        from .dataset_balancer import balance_dataset_by_removing_similar
        import tempfile
        
        # Crear dataset balanceado temporal
        temp_balanced_dir = Path(tempfile.mkdtemp(prefix="balanced_dataset_"))
        
        balance_stats = balance_dataset_by_removing_similar(
            data_dir=Path(args.data_dir),
            output_dir=temp_balanced_dir,
            min_samples_per_class=None,  # Usar el mínimo encontrado
            similarity_threshold=args.balance_similarity_threshold,
            device=str(dev),
        )
        
        print(f"\n✅ Dataset balanceado:")
        for class_name in balance_stats["original_counts"]:
            orig = balance_stats["original_counts"][class_name]
            final = balance_stats["final_counts"][class_name]
            removed = balance_stats["removed_counts"][class_name]
            print(f"   {class_name}: {orig} → {final} (eliminadas: {removed})")
        
        # Usar el dataset balanceado para el entrenamiento
        args.data_dir = str(temp_balanced_dir)

    # We load twice to allow different transforms per split
    # Usar FilteredImageFolder si se solicita filtrado con YOLO
    try:
        if args.filter_with_yolo:
            base_ds = FilteredImageFolder(
                args.data_dir,
                filter_with_yolo=True,
                min_confidence=args.yolo_min_confidence,
                require_features=args.yolo_require_features,
                transform=None,
            )
        else:
            base_ds = datasets.ImageFolder(args.data_dir, transform=None)
    except Exception as e:
        write_json(progress_file, {
            "status": "error",
            "error_message": f"Error al cargar dataset: {e}",
            "current_epoch": 0,
            "total_epochs": args.epochs,
        })
        raise
    
    if len(base_ds.classes) < 2:
        error_msg = "Se necesitan al menos 2 clases (carpetas) en el dataset."
        write_json(progress_file, {
            "status": "error",
            "error_message": error_msg,
            "current_epoch": 0,
            "total_epochs": args.epochs,
        })
        raise RuntimeError(error_msg)
    
    # Validación mínima: al menos 2 imágenes totales (una por clase mínimo)
    if len(base_ds) < 2:
        error_msg = "Se necesitan al menos 2 imágenes en total para entrenar (mínimo 1 por clase)."
        write_json(progress_file, {
            "status": "error",
            "error_message": error_msg,
            "current_epoch": 0,
            "total_epochs": args.epochs,
        })
        raise RuntimeError(error_msg)

    # Split indices using a ds without transforms, then create split datasets with transforms.
    train_subset, val_subset = _make_splits(base_ds, val_frac=args.val_frac, seed=args.seed)

    # Crear datasets con transforms, usando el mismo tipo de dataset (filtrado o no)
    if args.filter_with_yolo:
        train_ds = FilteredImageFolder(
            args.data_dir,
            filter_with_yolo=False,  # Ya filtramos en base_ds, no filtrar de nuevo
            transform=train_tfm,
        )
        val_ds = FilteredImageFolder(
            args.data_dir,
            filter_with_yolo=False,  # Ya filtramos en base_ds, no filtrar de nuevo
            transform=val_tfm,
        )
        # Aplicar los mismos índices filtrados
        train_ds.samples = [base_ds.samples[i] for i in train_subset.indices]
        train_ds.targets = [base_ds.targets[i] for i in train_subset.indices]
        val_ds.samples = [base_ds.samples[i] for i in val_subset.indices]
        val_ds.targets = [base_ds.targets[i] for i in val_subset.indices]
    else:
        train_ds = datasets.ImageFolder(args.data_dir, transform=train_tfm)
        val_ds = datasets.ImageFolder(args.data_dir, transform=val_tfm)
        train_ds = Subset(train_ds, train_subset.indices)
        val_ds = Subset(val_ds, val_subset.indices)

    # Usar num_workers=0 en macOS para evitar problemas de multiprocessing
    num_workers = 0 if sys.platform == "darwin" else 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(num_classes=len(base_ds.classes), pretrained=True).to(dev)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_state = None
    
    # Archivo de progreso para la UI
    progress_file = artifacts_dir / "training_progress.json"

    t0 = time.time()
    
    # Inicializar progreso
    write_json(progress_file, {
        "status": "initializing",
        "current_epoch": 0,
        "total_epochs": args.epochs,
        "progress_percent": 0.0,
        "train_loss": 0.0,
        "train_acc": 0.0,
        "val_loss": 0.0,
        "val_acc": 0.0,
        "best_val_acc": 0.0,
        "elapsed_time": 0.0,
        "message": "Inicializando modelo y cargando datos...",
    })
    
    # Actualizar progreso: modelo listo, comenzando entrenamiento
    write_json(progress_file, {
        "status": "running",
        "current_epoch": 0,
        "total_epochs": args.epochs,
        "progress_percent": 0.0,
        "train_loss": 0.0,
        "train_acc": 0.0,
        "val_loss": 0.0,
        "val_acc": 0.0,
        "best_val_acc": 0.0,
        "elapsed_time": 0.0,
        "message": f"Modelo inicializado. Comenzando entrenamiento con {len(train_ds)} imágenes de entrenamiento y {len(val_ds)} de validación.",
    })
    
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, loss_fn, dev)
        va_loss, va_acc = eval_one_epoch(model, val_loader, loss_fn, dev)
        
        elapsed = time.time() - t0
        progress_percent = (epoch / args.epochs) * 100
        
        print(
            f"epoch={epoch:02d}/{args.epochs} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}"
        )
        
        # Actualizar progreso
        write_json(progress_file, {
            "status": "running",
            "current_epoch": epoch,
            "total_epochs": args.epochs,
            "progress_percent": progress_percent,
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "val_loss": float(va_loss),
            "val_acc": float(va_acc),
            "best_val_acc": float(best_val_acc),
            "elapsed_time": float(elapsed),
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    if best_state is None:
        best_state = model.state_dict()

    torch.save(best_state, artifacts_dir / "model.pt")
    write_json(artifacts_dir / "classes.json", base_ds.classes)
    write_json(
        artifacts_dir / "config.json",
        {
            "img_size": args.img_size,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "best_val_acc": float(best_val_acc),
            "device_used": str(dev),
            "train_seconds": float(elapsed),
        },
    )
    
    # Marcar entrenamiento como completado
    write_json(progress_file, {
        "status": "completed",
        "current_epoch": args.epochs,
        "total_epochs": args.epochs,
        "progress_percent": 100.0,
        "train_loss": float(tr_loss),
        "train_acc": float(tr_acc),
        "val_loss": float(va_loss),
        "val_acc": float(va_acc),
        "best_val_acc": float(best_val_acc),
        "elapsed_time": float(elapsed),
        "display_name": display_name_snapshot,
    })

    print(f"\nGuardado: {(artifacts_dir / 'model.pt').as_posix()}")
    print(f"Clases: {base_ds.classes}")
    print(f"Mejor val_acc: {best_val_acc:.3f}")
    
    # Generar reporte de entrenamiento
    try:
        print("\nGenerando reporte de entrenamiento...")
        from .training_report import generate_training_report
        report_path = generate_training_report(
            data_dir=Path(args.data_dir),
            artifacts_dir=artifacts_dir,
            max_samples_per_class=20,
        )
        print(f"Reporte generado: {report_path}")
    except Exception as e:
        print(f"Advertencia: No se pudo generar el reporte: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


