import argparse
from pathlib import Path
import torch
import torch.nn as nn
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


def load_model(arch: str, num_classes: int, weights_path: Path) -> nn.Module:
    if arch == "baseline":
        model = BaselineCNN(num_classes=num_classes)
    else:
        model = create_transfer_model(arch, num_classes=num_classes, pretrained=False, freeze_backbone=False)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()
    return model


def export_torchscript(model: nn.Module, example: torch.Tensor, out_path: Path):
    traced = torch.jit.trace(model, example)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))


def export_onnx(model: nn.Module, example: torch.Tensor, out_path: Path, opset: int = 12):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )


def main():
    parser = argparse.ArgumentParser(description="Export model to TorchScript and ONNX")
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--weights", type=str, default="checkpoints_corrected/model_best.pt")
    parser.add_argument("--label-map", type=str, default="checkpoints_corrected/label_map.json")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--out-dir", type=str, default="exports")
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    label_map = load_label_map(Path(args.label_map))
    num_classes = len(label_map)

    model = load_model(args.arch, num_classes, Path(args.weights))
    example = torch.randn(1, 3, args.image_size, args.image_size)

    out_dir = Path(args.out_dir)
    export_torchscript(model, example, out_dir / f"{args.arch}.ts.pt")
    export_onnx(model, example, out_dir / f"{args.arch}.onnx", opset=args.opset)
    print(f"Exported TorchScript and ONNX to: {out_dir}")


if __name__ == "__main__":
    main()
