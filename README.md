## Cats vs Dogs Classifier (PyTorch)

End-to-end project to train and deploy a CNN/Transfer Learning model that classifies images as cat or dog. Includes data preprocessing, training, evaluation, inference, Streamlit UI, and Docker.

### 1) Setup

```bash
# Create venv (Windows PowerShell)
python -m venv .venv
. .venv/Scripts/Activate.ps1

# Install deps
pip install -r requirements.txt
```

### 2) Prepare Data

Organize images in `data/` using `ImageFolder` format:
```
data/
  cats/
    cat001.jpg
  dogs/
    dog001.jpg
```
You can download the Kaggle Dogs vs Cats dataset and extract into this structure.

### 3) Train

<img width="1547" height="502" alt="image" src="https://github.com/user-attachments/assets/2d3b89d8-9628-4010-9c23-6613f6673938" />

```bash
python -m src.train \
  --data-dir data \
  --out-dir checkpoints \
  --model resnet18 \
  --epochs 5 \
  --batch-size 32 \
  --tensorboard \
  --metrics-average macro
```
Artifacts: `checkpoints/model_best.pt`, `checkpoints/label_map.json`, `checkpoints/confusion_matrix.png`, `checkpoints/train_log.csv`.

- Metrics averaging: default to macro to be safer on class imbalance; override with `--metrics-average binary|weighted` as needed.
- TensorBoard: add `--tensorboard` and run `tensorboard --logdir checkpoints`.
<img width="495" height="498" alt="image" src="https://github.com/user-attachments/assets/2fe83f61-6d62-4cee-8a74-fe61490364a7" /> <img width="697" height="545" alt="image" src="https://github.com/user-attachments/assets/9c64f7bb-8c02-4a5c-9b3c-2c2c0b576a9a" />


### 4) Inference (CLI)

```bash
python -m src.infer --image path/to/test.jpg --arch resnet18 --label-map checkpoints/label_map.json --weights checkpoints/model_best.pt
```

### 5) Streamlit App

```bash
streamlit run app_streamlit.py
```
- Supports multi-image upload
- Shows per-class confidence bar chart (averaged) and downloadable CSV of predictions
- Guard for max total upload size (configurable in sidebar)

### 6) Docker

CPU base (default):
```bash
docker build -t cats-vs-dogs .
docker run -p 8501:8501 cats-vs-dogs
```

GPU base (for CUDA deployments):
```bash
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime -t cats-vs-dogs-gpu .
docker run --gpus all -p 8501:8501 cats-vs-dogs-gpu
```

### 7) HuggingFace Spaces

- Set `SDK: Streamlit`
- Entry point: `app.py` (runs Streamlit on port 7860)
- Dependencies are pinned in `requirements.txt` for stability; consider using specific torch/torchvision versions compatible with Spaces runtime

### Notes
- Default transforms assume 128x128 resize and ImageNet normalization.
- Choose architectures: `baseline`, `resnet18`, `mobilenet_v2`, `efficientnet_b0`.
- For GPU training, install CUDA-enabled PyTorch per `pytorch.org` instructions.
