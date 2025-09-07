from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import io
import torch
from torchvision import transforms

from .models import create_transfer_model, BaselineCNN
from .utils import load_label_map

app = FastAPI(title="Cats vs Dogs Classifier API")

MODEL = None
IDX_TO_CLASS = None
PREPROCESS = None


def build_preprocess(image_size: int = 128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(arch: str, num_classes: int, weights_path: Path):
    global MODEL
    if arch == "baseline":
        model = BaselineCNN(num_classes=num_classes)
    else:
        model = create_transfer_model(arch, num_classes=num_classes, pretrained=False, freeze_backbone=False)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()
    MODEL = model


@app.on_event("startup")
async def startup_event():
    global IDX_TO_CLASS, PREPROCESS
    label_map = load_label_map(Path("checkpoints_corrected/label_map.json"))
    IDX_TO_CLASS = {v: k for k, v in label_map.items()}
    PREPROCESS = build_preprocess(128)
    load_model("resnet18", num_classes=len(IDX_TO_CLASS), weights_path=Path("checkpoints_corrected/model_best.pt"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None or PREPROCESS is None or IDX_TO_CLASS is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    x = PREPROCESS(img).unsqueeze(0)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
    return JSONResponse({
        "prediction": IDX_TO_CLASS[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "probabilities": {IDX_TO_CLASS[i]: float(probs[i].item()) for i in range(len(IDX_TO_CLASS))}
    })
