"""Funciones auxiliares para el proyecto car_damage_classifier."""

from pathlib import Path
import random
import numpy as np
import torch


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def set_seed(seed: int = 42) -> None:
    """Fija semillas aleatorias para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detecta y devuelve cuda si está disponible, de lo contrario cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_data_folders(train_dir: Path, val_dir: Path) -> bool:
    """Valida que existen imágenes válidas en las carpetas de train y val.

    Retorna True si al menos hay una imagen válida en cada conjunto;
    de lo contrario imprime un mensaje claro y retorna False.
    """
    def _count_images(folder: Path) -> int:
        if not folder.exists():
            return 0
        return sum(1 for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)

    train_count = _count_images(train_dir)
    val_count = _count_images(val_dir)

    ok = True
    if train_count == 0:
        print(
            f"[ERROR] No se encontraron imágenes válidas en '{train_dir}'. "
            f"Agrega fotos (.jpg, .jpeg, .png, .webp) en las subcarpetas "
            f"'buen_estado' y 'chocado' antes de entrenar."
        )
        ok = False
    if val_count == 0:
        print(
            f"[ERROR] No se encontraron imágenes válidas en '{val_dir}'. "
            f"Agrega fotos de validación (.jpg, .jpeg, .png, .webp) en las "
            f"subcarpetas 'buen_estado' y 'chocado'."
        )
        ok = False
    return ok
