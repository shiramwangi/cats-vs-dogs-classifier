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
<img width="431" height="341" alt="image" src="https://github.com/user-attachments/assets/c049d8f7-0f30-410b-8d23-7067cdfbca42" /> <img width="475" height="475" alt="image" src="https://github.com/user-attachments/assets/513fd864-b3e8-4baa-960c-ed80022dd2b9" />


Organize images in `data/` using `ImageFolder` format, or pre-split as shown:
```
data/
  train/
    cats/
    dogs/
  val/
    cats/
    dogs/
```
You can download the Kaggle Dogs vs Cats dataset and extract into this structure.

### 3) Train

**Basic training (frozen backbone):**
```bash
python -m src.train \
  --data-dir data \
  --out-dir checkpoints \
  --model resnet18 \
  --epochs 10 \
  --batch-size 32 \
  --tensorboard \
  --metrics-average macro
```
<img width="1540" height="501" alt="image" src="https://github.com/user-attachments/assets/a4b82eb4-d75b-4e88-ae15-5e8aad96b873" />
<img width="1542" height="493" alt="image" src="https://github.com/user-attachments/assets/ec18113f-6f99-4ca4-9308-91ef2e1bbcb1" />
<img width="1542" height="501" alt="image" src="https://github.com/user-attachments/assets/c4aa583f-f60b-43b3-9c42-2afe9de13bc3" />

**Fine-tuning (unfreeze all layers):**
```bash
python -m src.train \
  --data-dir data \
  --out-dir checkpoints \
  --model resnet18 \
  --epochs 10 \
  --batch-size 32 \
  --unfreeze \
  --tensorboard \
  --metrics-average macro
```

**Artifacts:**
- `checkpoints/model_best.pt` - Best model weights
- `checkpoints/label_map.json` - Class mappings
- `checkpoints/confusion_matrix.png` - Validation confusion matrix
- `checkpoints/train_log.csv` - Training metrics
- `checkpoints/training_curves.png` - Loss/accuracy plots

**Features:**
- Data Augmentation: RandomResizedCrop, ColorJitter, RandomAffine, RandomRotation
- Transfer Learning: Freeze pretrained backbone by default (`--freeze-backbone`), use `--unfreeze` to fine-tune all layers
- Metrics: Default to macro averaging for class imbalance safety
- TensorBoard: add `--tensorboard` and run `tensorboard --logdir checkpoints`

### 4) Inference (CLI)

```bash
python -m src.infer --image path/to/test.jpg --arch resnet18 --label-map checkpoints/label_map.json --weights checkpoints/model_best.pt
```

### 5) Streamlit App

```bash
streamlit run app_streamlit.py
```
- Supports multi-image upload
- Shows per-image probability bar charts and downloadable CSV of predictions
- Sidebar shows best accuracy/F1 if `train_log.csv` exists
- Guard for max total upload size (configurable in sidebar)

Exported-model demo (TorchScript/ONNX):
```bash
python -m src.export --arch resnet18 --weights checkpoints/model_best.pt --label-map checkpoints/label_map.json --out-dir exports
streamlit run app_streamlit_export.py
```

### 6) Docker

CPU base (default):
```bash
docker build -t cats-vs-dogs .
docker run -p 8501:8501 -v %cd%/checkpoints_corrected:/app/checkpoints_corrected cats-vs-dogs
```

GPU base (for CUDA deployments):
```bash
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime -t cats-vs-dogs-gpu .
docker run --gpus all -p 8501:8501 cats-vs-dogs-gpu
```

### 7) HuggingFace Spaces

- Set `SDK: Streamlit`
- Entry point: `app.py` (runs Streamlit on port 7860)
- Dependencies are pinned in `requirements.txt`; choose torch/torchvision versions compatible with Spaces runtime

### Notes
- Default transforms assume 128x128 resize and ImageNet normalization.
- Choose architectures: `baseline`, `resnet18`, `mobilenet_v2`, `efficientnet_b0`.
- For GPU training, install CUDA-enabled PyTorch per `pytorch.org` instructions.
