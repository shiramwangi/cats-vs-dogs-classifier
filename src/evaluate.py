import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .data import create_dataloaders
from .models import BaselineCNN, create_transfer_model
from .utils import load_label_map, compute_metrics


def load_model(arch: str, num_classes: int, weights_path: Path) -> nn.Module:
    """Load model from checkpoint."""
    if arch == "baseline":
        model = BaselineCNN(num_classes=num_classes)
    else:
        model = create_transfer_model(arch, num_classes=num_classes, pretrained=False, freeze_backbone=False)
    
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device, class_names: list):
    """Evaluate model and return predictions and metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_preds


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyze_class_balance(y_true, y_pred, class_names):
    """Analyze class balance and bias."""
    from collections import Counter
    
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    print("\n" + "="*50)
    print("CLASS BALANCE ANALYSIS")
    print("="*50)
    
    for i, class_name in enumerate(class_names):
        true_count = true_counts.get(i, 0)
        pred_count = pred_counts.get(i, 0)
        total = len(y_true)
        
        print(f"{class_name}:")
        print(f"  True samples: {true_count} ({true_count/total*100:.1f}%)")
        print(f"  Predicted: {pred_count} ({pred_count/total*100:.1f}%)")
        print(f"  Bias: {pred_count - true_count:+d} samples")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pt", help="Path to model checkpoint")
    parser.add_argument("--arch", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--image-size", type=int, default=128, help="Image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--out-dir", type=str, default="checkpoints", help="Output directory")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading validation data...")
    _, val_loader, full_dataset = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=0.2,
        num_workers=2,
    )
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.arch, num_classes, Path(args.checkpoint))
    model = model.to(device)
    
    # Evaluate
    print("Running evaluation...")
    y_true, y_pred = evaluate_model(model, val_loader, device, class_names)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, average=None)
    macro_metrics = compute_metrics(y_true, y_pred, average="macro")
    micro_metrics = compute_metrics(y_true, y_pred, average="micro")
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {macro_metrics['f1']:.4f}")
    print(f"Micro F1: {micro_metrics['f1']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Analyze class balance
    analyze_class_balance(y_true, y_pred, class_names)
    
    # Save confusion matrix
    out_dir = Path(args.out_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir / "evaluation_confusion_matrix.png")
    print(f"Confusion matrix saved to: {out_dir / 'evaluation_confusion_matrix.png'}")
    
    # Save detailed results
    results = {
        'accuracy': metrics['accuracy'],
        'macro_f1': macro_metrics['f1'],
        'micro_f1': micro_metrics['f1'],
        'precision_per_class': metrics['precision'].tolist() if hasattr(metrics['precision'], 'tolist') else list(metrics['precision']),
        'recall_per_class': metrics['recall'].tolist() if hasattr(metrics['recall'], 'tolist') else list(metrics['recall']),
        'f1_per_class': metrics['f1'].tolist() if hasattr(metrics['f1'], 'tolist') else list(metrics['f1']),
    }
    
    import json
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {out_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    import numpy as np
    main()
