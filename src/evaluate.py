"""Script de evaluación del clasificador de daños vehiculares."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import get_device, validate_data_folders
from model import load_model, build_model, count_parameters, CLASS_NAMES

VAL_DIR = Path("data/val")
MODEL_PATH = Path("models/best_model.pth")
METRICS_PATH = Path("outputs/metrics.json")
CURVES_PATH = Path("outputs/training_curves.png")
CONFUSION_PATH = Path("outputs/confusion_matrix.png")


def get_val_transform(img_size: int = 224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


def plot_training_curves(history: dict, save_path: Path) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", marker="o", color="#1f77b4")
    plt.plot(history["val_loss"], label="Val Loss", marker="o", color="#ff7f0e")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Pérdida por época")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", marker="o", color="#1f77b4")
    plt.plot(history["val_acc"], label="Val Acc", marker="o", color="#ff7f0e")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Exactitud por época")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.replace("_", " ").title() for c in class_names],
        yticklabels=[c.replace("_", " ").title() for c in class_names],
        cbar=False,
        linewidths=1,
        linecolor="gray",
    )
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate() -> None:
    device = get_device()
    print(f"[info] Dispositivo: {device}")

    if not MODEL_PATH.exists():
        print(f"[ERROR] No se encontró el modelo en {MODEL_PATH}. Ejecuta primero: python src/train.py")
        return

    if not validate_data_folders(VAL_DIR, VAL_DIR):
        print("[abort] Evaluación cancelada por falta de datos de validación.")
        return

    # Cargar modelo
    model, class_names, arch = load_model(MODEL_PATH, device=device)
    print(f"[info] Modelo cargado: {arch} | clases: {class_names}")

    # Dataloader de validación
    val_tf = get_val_transform()
    val_ds = datasets.ImageFolder(root=str(VAL_DIR), transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Asegurar que class_names coincidan con ImageFolder
    if sorted(val_ds.classes) != sorted(class_names):
        print(f"[advertencia] Clases del modelo {class_names} difieren de las del dataset {val_ds.classes}")

    # Inferencia
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    target_names = [c.replace("_", " ").title() for c in class_names]
    report = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    print("\n=== Reporte de Clasificación ===")
    print(report)
    print(f"Accuracy global: {acc:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, CONFUSION_PATH)
    print(f"[info] Matriz de confusión guardada en {CONFUSION_PATH}")

    # Curvas de entrenamiento (si existe historial)
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
        plot_training_curves(history, CURVES_PATH)
        print(f"[info] Curvas de entrenamiento guardadas en {CURVES_PATH}")
    else:
        print(f"[aviso] No se encontró {METRICS_PATH}; no se generaron curvas de entrenamiento.")

    # Resumen de parámetros
    total, trainable = count_parameters(model)
    print(f"\n=== Arquitectura ===")
    print(f"Backbone: {arch}")
    print(f"Parámetros totales: {total:,}")
    print(f"Parámetros entrenables: {trainable:,}")


if __name__ == "__main__":
    evaluate()
