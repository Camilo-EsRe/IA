"""Script de entrenamiento del clasificador de daños vehiculares."""

from pathlib import Path
import argparse
import json
import random
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Asegurar que se puede importar src como paquete al ejecutar src/train.py directamente
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import set_seed, get_device, validate_data_folders, VALID_EXTENSIONS
from model import build_model, save_model, CLASS_NAMES

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
MODEL_PATH = Path("models/best_model.pth")
METRICS_PATH = Path("outputs/metrics.json")
CURVES_PATH = Path("outputs/training_curves.png")


def get_transforms(img_size: int = 224):
    """Devuelve transformaciones para train y val."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tf, val_tf


def generate_synthetic_demo(base_dir: Path = Path("data_demo")) -> None:
    """Genera imágenes sintéticas temporales para verificar el pipeline sin tocar data/."""
    if base_dir.exists():
        shutil.rmtree(base_dir)
    for split in ["train", "val"]:
        for cls in CLASS_NAMES:
            (base_dir / split / cls).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_per_split_cls = 40

    for split in ["train", "val"]:
        for cls in CLASS_NAMES:
            for i in range(n_per_split_cls):
                # Imagen aleatoria con un patrón sencillo para diferenciar clases
                img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
                # clase "chocado" tiende a tener más bordes oscuros (simulado)
                if cls == "chocado":
                    img[40:180, 40:180] = np.maximum(img[40:180, 40:180] - 60, 0)
                from PIL import Image
                Image.fromarray(img).save(base_dir / split / cls / f"{cls}_{i}.jpg")
    print(f"[demo] Imágenes sintéticas generadas en '{base_dir}'")


def train(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    img_size: int = 224,
    arch: str = "mobilenet_v3_small",
    fine_tune: bool = False,
    seed: int = 42,
    demo_synthetic: bool = False,
) -> None:
    set_seed(seed)
    device = get_device()
    print(f"[info] Dispositivo: {device}")

    train_dir = TRAIN_DIR
    val_dir = VAL_DIR

    if demo_synthetic:
        generate_synthetic_demo()
        train_dir = Path("data_demo/train")
        val_dir = Path("data_demo/val")

    if not validate_data_folders(train_dir, val_dir):
        print("[abort] Entrenamiento cancelado por falta de datos.")
        return

    train_tf, val_tf = get_transforms(img_size)

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=val_tf)

    class_names = train_ds.classes
    print(f"[info] Clases detectadas: {class_names}")
    print(f"[info] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(arch=arch, num_classes=len(class_names), pretrained=True, fine_tune=fine_tune)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, class_names, arch, MODEL_PATH)
            print(f"  -> Mejor modelo guardado en {MODEL_PATH} (val_acc={val_acc:.4f})")

    # Guardar historial
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[info] Métricas guardadas en {METRICS_PATH}")

    # Graficar curvas de entrenamiento
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Pérdida por época")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", marker="o")
    plt.plot(history["val_acc"], label="Val Acc", marker="o")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Exactitud por época")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVES_PATH, dpi=150)
    plt.close()
    print(f"[info] Curvas de entrenamiento guardadas en {CURVES_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar clasificador de daños vehiculares")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small",
                        choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--fine-tune", action="store_true",
                        help="Activar fine-tuning de las últimas capas del backbone")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-synthetic", action="store_true",
                        help="Generar imágenes sintéticas temporales en data_demo/ para probar el pipeline")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        arch=args.arch,
        fine_tune=args.fine_tune,
        seed=args.seed,
        demo_synthetic=args.demo_synthetic,
    )


if __name__ == "__main__":
    main()
