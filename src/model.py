"""Arquitectura de Transfer Learning para clasificación de daños vehiculares."""

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models


CLASS_NAMES = ["buen_estado", "chocado"]


def build_model(
    arch: str = "mobilenet_v3_small",
    num_classes: int = 2,
    pretrained: bool = True,
    fine_tune: bool = False,
) -> nn.Module:
    """Construye un modelo de Transfer Learning.

    Args:
        arch: 'mobilenet_v3_small' o 'resnet18'.
        num_classes: número de clases de salida (2 por defecto).
        pretrained: si usar pesos preentrenados de ImageNet.
        fine_tune: si True, descongela las últimas capas del backbone para
            fine-tuning; si False, el backbone queda congelado.

    Returns:
        modelo nn.Module listo para entrenamiento.
    """
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        # Congelar backbone
        for param in model.parameters():
            param.requires_grad = False
        # Reemplazar clasificador final usando la entrada real de features (576)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        if fine_tune:
            # Descongelar las últimas capas de features
            for param in model.features[-2:].parameters():
                param.requires_grad = True
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        if fine_tune:
            for param in model.layer4.parameters():
                param.requires_grad = True
    else:
        raise ValueError(
            f"Arquitectura no soportada: {arch}. Usa 'mobilenet_v3_small' o 'resnet18'."
        )

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Retorna (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_model(
    model: nn.Module,
    class_names: list[str],
    arch: str,
    path: Path | str,
    fine_tune: bool = False,
) -> None:
    """Guarda state_dict, class_names, arch y fine_tune en un checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "arch": arch,
        "fine_tune": fine_tune,
    }
    torch.save(checkpoint, path)


def load_model(
    path: Path | str,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[str], str]:
    """Carga un modelo desde checkpoint y lo devuelve listo para inferencia."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(path)
    checkpoint = torch.load(path, map_location=device)
    arch = checkpoint["arch"]
    class_names = checkpoint["class_names"]
    fine_tune = checkpoint.get("fine_tune", False)
    model = build_model(
        arch=arch,
        num_classes=len(class_names),
        pretrained=False,
        fine_tune=fine_tune,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, arch