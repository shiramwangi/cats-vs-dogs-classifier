import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from .models import BaselineCNN, create_transfer_model
from .utils import load_label_map


def build_preprocess(image_size: int = 128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(arch: str, num_classes: int, weights_path: Path) -> torch.nn.Module:
    if arch == "baseline":
        model = BaselineCNN(num_classes=num_classes)
    else:
        model = create_transfer_model(arch, num_classes=num_classes, pretrained=False, freeze_backbone=False)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()
    return model


def predict(image_path: Path, arch: str, label_map_path: Path, weights_path: Path, image_size: int = 128):
    label_map = load_label_map(label_map_path)
    idx_to_class = {v: k for k, v in label_map.items()}
    preprocess = build_preprocess(image_size)

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)

    model = load_model(arch, num_classes=len(label_map), weights_path=weights_path)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
    return idx_to_class[pred_idx], float(probs[pred_idx].item())


def main():
    parser = argparse.ArgumentParser(description="Single-image inference")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--label-map", type=str, default="checkpoints_corrected/label_map.json")
    parser.add_argument("--weights", type=str, default="checkpoints_corrected/model_best.pt")
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    pred, conf = predict(Path(args.image), args.arch, Path(args.label_map), Path(args.weights), args.image_size)
    print(f"Prediction: {pred} (confidence={conf:.3f})")


if __name__ == "__main__":
    main()
