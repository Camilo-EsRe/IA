# Car Damage Classifier

Clasificador binario de imágenes de vehículos (**buen estado** vs. **chocado**) usando Transfer Learning con PyTorch e interfaz web en Streamlit.

## 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/car_damage_classifier.git
cd car_damage_classifier
```

## 2. Crear y activar el entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 4. Agregar tus fotos de entrenamiento

Organiza las imágenes en estas carpetas (formatos: `.jpg`, `.jpeg`, `.png`, `.webp`):

- `data/train/buen_estado/` — carros en buen estado (entrenamiento)
- `data/train/chocado/` — carros chocados (entrenamiento)
- `data/val/buen_estado/` — carros en buen estado (validación)
- `data/val/chocado/` — carros chocados (validación)

Se recomienda mínimo 20–30 imágenes por clase en cada conjunto.

## 5. Entrenar el modelo

```bash
python src/train.py
```

Opción rápida con imágenes sintéticas (para probar sin subir fotos propias):

```bash
python src/train.py --demo-synthetic --epochs 5
```

Al terminar se genera `models/best_model.pth`, `outputs/metrics.json` y `outputs/training_curves.png`.

## 6. Evaluar el modelo

```bash
python src/evaluate.py
```

## 7. Ejecutar la aplicación web

```bash
streamlit run app.py
```

La app se abre en tu navegador, por defecto en `http://localhost:8501`.

## Comandos resumen (Windows PowerShell)

```powershell
git clone https://github.com/TU_USUARIO/car_damage_classifier.git
cd car_damage_classifier
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## Requisitos

- Python 3.9 o superior
- GPU NVIDIA recomendada (opcional, funciona en CPU)
