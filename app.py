"""Interfaz web Streamlit para el clasificador de daños vehiculares."""

from pathlib import Path
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import CLASS_NAMES, count_parameters, load_model
from src.utils import get_device

MODEL_PATH = Path("models/best_model.pth")
CURVES_PATH = Path("outputs/training_curves.png")
CONFUSION_PATH = Path("outputs/confusion_matrix.png")
METRICS_PATH = Path("outputs/metrics.json")

st.set_page_config(
    page_title="Detector de Daños Vehiculares",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Detector de Daños Vehiculares (CNN + Transfer Learning)")
st.markdown(
    "Clasificador binario de imágenes que distingue entre carros en **buen estado** y carros **chocados**. "
    "Basado en Transfer Learning con MobileNet-V3 Small (preentrenado en ImageNet)."
)

device = get_device()


@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        return None, None, None
    model, class_names, arch = load_model(MODEL_PATH, device=device)
    return model, class_names, arch


def predict_image(model, class_names, image: Image.Image):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    input_tensor = tf(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        outputs = model(input_tensor)
        t1 = time.perf_counter()
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    inference_ms = (t1 - t0) * 1000
    return pred_class, confidence, probs, inference_ms


# --- Pestañas ---
tab1, tab2 = st.tabs(["🔍 Clasificar Imagen", "📊 Métricas y Arquitectura"])

with tab1:
    st.subheader("Clasificar una foto de vehículo")
    uploaded_file = st.file_uploader(
        "Sube una imagen del vehículo",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida", use_column_width=True)

        model, class_names, arch = load_trained_model()
        if model is None:
            st.warning(
                "⚠️ Aún no existe un modelo entrenado. "
                "Ejecuta primero `python src/train.py` para generar `models/best_model.pth`."
            )
        else:
            pred_class, confidence, probs, inference_ms = predict_image(model, class_names, image)

            label_map = {"buen_estado": "Carro en Buen Estado", "chocado": "Carro Chocado"}
            diag = label_map.get(pred_class, pred_class)

            if pred_class == "buen_estado":
                st.success(f"**Diagnóstico:** {diag}")
            else:
                st.error(f"**Diagnóstico:** {diag}")

            st.metric("Confianza", f"{confidence * 100:.2f}%")
            st.metric("Tiempo de inferencia", f"{inference_ms:.1f} ms")

            st.subheader("Probabilidades por clase")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            labels_disp = [c.replace("_", " ").title() for c in class_names]
            colors = ["#2ca02c" if c == "buen_estado" else "#d62728" for c in class_names]
            bars = ax.bar(labels_disp, probs, color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probabilidad")
            for bar, p in zip(bars, probs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{p:.2%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            st.pyplot(fig)

with tab2:
    st.subheader("Métricas y Arquitectura del modelo")

    col_curves, col_cm = st.columns(2)
    with col_curves:
        st.markdown("#### Curvas de Entrenamiento")
        if CURVES_PATH.exists():
            st.image(str(CURVES_PATH), use_column_width=True)
        else:
            st.info("No se encontró `outputs/training_curves.png`. Entrena el modelo primero.")

    with col_cm:
        st.markdown("#### Matriz de Confusión")
        if CONFUSION_PATH.exists():
            st.image(str(CONFUSION_PATH), use_column_width=True)
        else:
            st.info("No se encontró `outputs/confusion_matrix.png`. Ejecuta `python src/evaluate.py`.")

    st.markdown("#### Hiperparámetros y Parámetros del Modelo")
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
        st.write("Historial de métricas (últimas épocas):")
        st.json({k: v[-3:] if isinstance(v, list) else v for k, v in history.items()})
    else:
        st.info("No se encontró `outputs/metrics.json`. Entrena el modelo primero.")

    model, class_names, arch = load_trained_model()
    if model is not None:
        total, trainable = count_parameters(model)
        st.write(f"**Arquitectura:** {arch}")
        st.write(f"**Clases:** {', '.join(c.replace('_', ' ').title() for c in class_names)}")
        st.write(f"**Parámetros totales:** {total:,}")
        st.write(f"**Parámetros entrenables:** {trainable:,}")
    else:
        st.info("Carga un modelo entrenado para ver el resumen de parámetros.")

st.divider()
st.caption("Proyecto educativo - Clasificación de daños vehiculares con Transfer Learning en PyTorch.")