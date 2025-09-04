import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true, y_pred, average: Optional[str] = None) -> Dict[str, float]:
    """Compute accuracy, precision, recall, f1.

    If average is None: auto-select 'binary' for 2 classes, else 'macro'.
    Accepts: 'binary', 'macro', 'weighted'.
    """
    acc = accuracy_score(y_true, y_pred)
    if average is None:
        num_classes = len(set(list(y_true) + list(y_pred)))
        average = "binary" if num_classes == 2 else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def plot_confusion(y_true, y_pred, class_names: Tuple[str, ...], out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_checkpoint(state: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def save_label_map(class_to_idx: Dict[str, int], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)


def load_label_map(path: Path) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
