import argparse
from pathlib import Path
import time
import csv
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter  # optional
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from .data import create_dataloaders
from .models import BaselineCNN, create_transfer_model
from .utils import compute_metrics, plot_confusion, save_checkpoint, save_label_map, plot_training_curves


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name: str, num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    return create_transfer_model(model_name=model_name, num_classes=num_classes, pretrained=True, freeze_backbone=freeze_backbone)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, y_true, y_pred


def train(args):
    device = get_device()
    train_loader, val_loader, full_dataset = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    num_classes = len(full_dataset.classes)
    model = build_model(args.model, num_classes, freeze_backbone=args.freeze_backbone).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save label map
    class_to_idx = full_dataset.class_to_idx
    save_label_map(class_to_idx, out_dir / "label_map.json")

    # CSV logger (write header if file is missing or empty)
    csv_path = out_dir / "train_log.csv"
    write_header = True
    if csv_path.exists():
        try:
            if os.path.getsize(csv_path) > 0:
                write_header = False
        except OSError:
            write_header = True

    # TensorBoard
    writer = SummaryWriter(log_dir=str(out_dir)) if (args.tensorboard and SummaryWriter is not None) else None

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_loss = 0.0
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            epoch_loss += loss.item() * batch_size
            global_step += 1

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)

            pbar.set_postfix(loss=loss.item())

        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss, y_true, y_pred = evaluate(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred, average=args.metrics_average)
        val_acc = metrics["accuracy"]

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} acc={val_acc:.4f} f1={metrics['f1']:.4f}")

        # log epoch to CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer_csv = csv.writer(f)
            if write_header:
                writer_csv.writerow(["epoch", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1"]) 
                write_header = False
            writer_csv.writerow([epoch, train_loss, val_loss, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]])

        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            writer.add_scalar("epoch/val_accuracy", metrics["accuracy"], epoch)
            writer.add_scalar("epoch/val_f1", metrics["f1"], epoch)

        # Save latest
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "timestamp": time.time(),
        }, out_dir / "model_last.pt")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
                "args": vars(args),
                "timestamp": time.time(),
            }, out_dir / "model_best.pt")
            # Confusion matrix
            plot_confusion(y_true, y_pred, tuple(full_dataset.classes), out_path=out_dir / "confusion_matrix.png")

    if writer is not None:
        writer.close()
    
    # Generate training curves plot
    plot_training_curves(csv_path, out_dir / "training_curves.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Train cats vs dogs classifier")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory with class subfolders")
    parser.add_argument("--out-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--model", type=str, default="resnet18", choices=["baseline", "resnet18", "mobilenet_v2", "efficientnet_b0"], help="Model architecture")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--metrics-average", type=str, default=None, choices=[None, "binary", "macro", "weighted"], help="Averaging for PRF1 metrics")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze pretrained backbone layers (transfer learning)")
    parser.add_argument("--unfreeze", action="store_true", help="Unfreeze all layers for fine-tuning")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Handle unfreeze flag
    if args.unfreeze:
        args.freeze_backbone = False
    train(args)
