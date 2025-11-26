import logging
import sys
import numpy as np
import torch
import json
import os
import transformers
import warnings
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Dict


def setup_logging():
    """
    Basic configuration of the logging service
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity_error()


def compute_metrics(eval_pred):
    """
    Compute all useful metrics for evalutaing predictions
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

    return {"accuracy": accuracy, "f1": f1_weighted, "f1_macro": f1_macro}


def compute_loss_weights(
    labels: List[int],
    json_path: str,
    id2label: Dict[int, str],
    num_labels: int,
    use_balance: bool = True,
    use_priority: bool = True,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Computes combined weights for loss function.

    Args:
        alpha: 0=priority only, 1=balance only, 0.5=equal mix
    """
    balance_weights = np.ones(num_labels, dtype=np.float32)
    priority_weights = np.ones(num_labels, dtype=np.float32)

    # Class balance weights
    if use_balance:
        counts = np.bincount(labels, minlength=num_labels)
        safe_counts = np.where(counts == 0, 1, counts)
        total = len(labels)
        balance_weights = total / (num_labels * safe_counts.astype(np.float32))
        balance_weights = normalize_weights(balance_weights, min_val=0.5, max_val=2.0)
        logging.info(
            f"Class balance weights: {balance_weights.min():.3f} - {balance_weights.max():.3f}"
        )

    # Priority weights
    if use_priority and os.path.exists(json_path):
        with open(json_path, "r") as f:
            descriptions = json.load(f)
        priority_map = {item["intent"]: item["priority"] for item in descriptions}
        priorities = np.array(
            [priority_map.get(id2label[i], 50) for i in range(num_labels)]
        )
        priority_weights = 0.5 + 1.5 * (priorities / 100.0)
        logging.info(
            f"Priority weights: {priority_weights.min():.3f} - {priority_weights.max():.3f}"
        )

    # Combine weights
    if use_balance and use_priority:
        final_weights = alpha * balance_weights + (1 - alpha) * priority_weights
        logging.info(f"Combined weights (α={alpha:.2f})")
    elif use_balance:
        final_weights = balance_weights
    elif use_priority:
        final_weights = priority_weights
    else:
        final_weights = np.ones(num_labels, dtype=np.float32)

    # Normalize to mean=1.0 and clip extremes
    final_weights = final_weights / final_weights.mean()
    final_weights = np.clip(final_weights, 0.1, 10.0)

    logging.info(
        f"Final weights: {final_weights.min():.3f} - {final_weights.max():.3f}"
    )

    return torch.tensor(final_weights, dtype=torch.float32)


def normalize_weights(
    weights: np.ndarray, min_val: float = 0.5, max_val: float = 2.0
) -> np.ndarray:
    """
    Min-max normalization to [min_val, max_val]
    """
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min < 1e-6:
        return np.ones_like(weights) * ((min_val + max_val) / 2)
    normalized = (weights - w_min) / (w_max - w_min)

    return min_val + (max_val - min_val) * normalized
