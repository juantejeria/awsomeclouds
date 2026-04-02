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
from .model import build_model, freeze_backbone, unfreeze_last_block, unfreeze_all
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
    p.add_argument("--crop_to_face", action="store_true", help="Pre-procesar dataset recortando cada imagen al rostro/cuerpo detectado antes de entrenar")
    args = p.parse_args()

    dev = _device()
    artifacts_dir = ensure_dir(args.artifacts_dir)
    estab_name = artifacts_dir.name
    display_name_snapshot = get_establecimiento_display_name(estab_name)

    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
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

    # Archivo de progreso para la UI (definir temprano para manejo de errores)
    progress_file = artifacts_dir / "training_progress.json"

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

    # --- Pre-procesar dataset: recortar imágenes a rostro/cuerpo ---
    if args.crop_to_face:
        print("\n🔍 Pre-procesando dataset: recortando a rostro/cuerpo de vaca...")
        from .dataset_manager import crop_dataset_to_faces
        import tempfile

        source_dir = Path(args.data_dir)
        temp_face_dir = Path(tempfile.mkdtemp(prefix="face_cropped_dataset_"))

        crop_stats = crop_dataset_to_faces(
            data_dir=source_dir,
            output_dir=temp_face_dir,
        )

        print(f"\n✅ Pre-procesamiento de rostros completado:")
        print(f"   Total imágenes:  {crop_stats['total']}")
        print(f"   Rostros (face):  {crop_stats['face_cropped']}")
        print(f"   Cuerpos (body):  {crop_stats['body_cropped']}")
        print(f"   Sin detección:   {crop_stats['no_detection']} (se mantiene imagen completa)")

        # Use the face-cropped dataset from now on
        args.data_dir = str(temp_face_dir)

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

    # --- Regularización adaptada al tamaño del dataset ---
    # Datasets pequeños (<30 imgs/clase) necesitan menos regularización
    targets_all = [y for _, y in base_ds.samples]
    class_counts = Counter(targets_all)
    n_classes = len(base_ds.classes)
    avg_samples_per_class = len(targets_all) / max(n_classes, 1)

    # Ajustar regularización según tamaño del dataset
    if avg_samples_per_class < 30:
        dropout_val = 0.15
        label_smooth = 0.02
        print(f"📊 Dataset pequeño ({avg_samples_per_class:.0f} imgs/clase promedio) → "
              f"regularización ligera (dropout={dropout_val}, label_smooth={label_smooth})")
    elif avg_samples_per_class < 100:
        dropout_val = 0.25
        label_smooth = 0.05
        print(f"📊 Dataset mediano ({avg_samples_per_class:.0f} imgs/clase promedio) → "
              f"regularización moderada (dropout={dropout_val}, label_smooth={label_smooth})")
    else:
        dropout_val = 0.4
        label_smooth = 0.1
        print(f"📊 Dataset grande ({avg_samples_per_class:.0f} imgs/clase promedio) → "
              f"regularización completa (dropout={dropout_val}, label_smooth={label_smooth})")

    model = build_model(num_classes=n_classes, pretrained=True, dropout=dropout_val).to(dev)

    # --- Class-weighted loss with label smoothing ---
    # Inverse-frequency weights, normalized
    class_weights = torch.tensor(
        [1.0 / class_counts.get(i, 1) for i in range(n_classes)],
        dtype=torch.float,
    )
    class_weights = class_weights / class_weights.sum() * n_classes
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(dev), label_smoothing=label_smooth)

    # --- Gradual fine-tuning schedule ---
    # Phase 1: freeze backbone, train only classifier head
    # Phase 2: unfreeze layer4 + head
    # Divide epochs: ~40% phase 1, ~60% phase 2
    phase1_epochs = max(1, int(args.epochs * 0.4))
    phase2_start = phase1_epochs + 1

    freeze_backbone(model)
    print(f"\n🧊 Fase 1 (épocas 1-{phase1_epochs}): backbone congelado, solo cabeza clasificadora")
    print(f"🔥 Fase 2 (épocas {phase2_start}-{args.epochs}): descongelando layer4 + cabeza\n")

    # Phase 1 optimizer: only fc params, higher LR for new head
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr * 5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=phase1_epochs, eta_min=args.lr)

    best_val_acc = -1.0
    best_state = None
    early_stop_patience = 5
    epochs_without_improvement = 0

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
    
    final_epoch = args.epochs
    tr_loss = 0.0
    tr_acc = 0.0
    va_loss = 0.0
    va_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Phase transition: unfreeze layer4 at phase2_start ---
        if epoch == phase2_start:
            unfreeze_last_block(model)
            # New optimizer with all trainable params and lower LR
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optim = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
            remaining_epochs = args.epochs - phase1_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=remaining_epochs, eta_min=args.lr * 0.1,
            )
            epochs_without_improvement = 0  # Reset early stopping for phase 2
            print(f"\n🔥 Fase 2: layer4 descongelado (lr={args.lr})\n")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, loss_fn, dev)
        va_loss, va_acc = eval_one_epoch(model, val_loader, loss_fn, dev)
        scheduler.step()
        
        elapsed = time.time() - t0
        progress_percent = (epoch / args.epochs) * 100
        phase_str = "fase1-head" if epoch <= phase1_epochs else "fase2-finetune"
        current_lr = optim.param_groups[0]["lr"]
        
        print(
            f"epoch={epoch:02d}/{args.epochs} [{phase_str}] lr={current_lr:.2e} "
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
            "phase": phase_str,
            "learning_rate": float(current_lr),
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping (per phase)
        if epochs_without_improvement >= early_stop_patience:
            if epoch <= phase1_epochs:
                # Skip to phase 2 early
                print(f"\n⏩ Early stop fase 1 en época {epoch} (sin mejora en {early_stop_patience} épocas). Pasando a fase 2...")
                phase1_epochs = epoch  # Adjust so phase 2 starts next
                # phase2_start logic handled by the condition at top of loop
                # Force phase2_start for next iteration
                phase2_start = epoch + 1
                epochs_without_improvement = 0
            else:
                print(f"\n⏹️  Early stopping en época {epoch} (sin mejora en {early_stop_patience} épocas)")
                final_epoch = epoch
                break

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
            "epochs_trained": final_epoch,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": 1e-4,
            "label_smoothing": label_smooth,
            "dropout": dropout_val,
            "gradual_unfreeze": True,
            "early_stopping_patience": early_stop_patience,
            "best_val_acc": float(best_val_acc),
            "device_used": str(dev),
            "train_seconds": float(elapsed),
        },
    )
    
    # Marcar entrenamiento como completado
    write_json(progress_file, {
        "status": "completed",
        "current_epoch": final_epoch,
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

    # Generar centroides para open-set recognition
    try:
        print("\nGenerando centroides de embeddings...")
        from .infer import compute_centroids, save_centroids
        # Load best model for centroid computation
        best_model = build_model(num_classes=n_classes, pretrained=False, dropout=dropout_val)
        best_model.load_state_dict(torch.load(artifacts_dir / "model.pt", map_location="cpu"))
        best_model.eval().to(dev)
        centroids = compute_centroids(best_model, val_tfm, dev, Path(args.data_dir), base_ds.classes)
        centroid_path = save_centroids(centroids, artifacts_dir)
        print(f"Centroides guardados: {centroid_path} ({len(centroids)} clases)")
    except Exception as e:
        print(f"Advertencia: No se pudieron generar centroides: {e}")
        import traceback
        traceback.print_exc()

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


