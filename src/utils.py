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


def compute_loss_weights(labels: List[int], num_labels: int) -> torch.Tensor:
    """
    Computes weights for loss function based solely on class balance (Inverse Frequency).
    """
    counts = np.bincount(labels, minlength=num_labels)

    # Handle zero counts to avoid division by zero
    safe_counts = np.where(counts == 0, 1, counts)

    # Compute Inverse Class Frequency
    total = len(labels)
    weights = total / (num_labels * safe_counts.astype(np.float32))

    # Normalize to prevent exploding gradients for rare classes
    weights = normalize_weights(weights, min_val=0.5, max_val=2.0)

    # Ensure mean is 1.0 to maintain loss scale
    weights = weights / weights.mean()

    logging.info(
        f"Class balance weights: Min={weights.min():.3f}, Max={weights.max():.3f}"
    )

    return torch.tensor(weights, dtype=torch.float32)


def normalize_weights(
    weights: np.ndarray, min_val: float = 0.5, max_val: float = 2.0
) -> np.ndarray:
    """Min-max normalization helper"""
    w_min, w_max = weights.min(), weights.max()

    if w_max - w_min < 1e-6:
        return np.ones_like(weights) * ((min_val + max_val) / 2)

    normalized = (weights - w_min) / (w_max - w_min)
    return min_val + (max_val - min_val) * normalized
