"""
Utility functions for the intent classification system
Now with centralized logging configuration
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Optional
from .logging_config import LoggingConfig


# Module logger
logger = LoggingConfig.get_logger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup project-wide logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    LoggingConfig.setup_logging(
        level=level, log_file=log_file, use_colors=True, suppress_warnings=True
    )
    logger.info(f"Logging initialized: level={level}")


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for model predictions.

    Args:
        eval_pred: Tuple of (logits, labels) from Trainer

    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

    logger.debug(
        f"Metrics computed - Acc: {accuracy:.4f}, F1-weighted: {f1_weighted:.4f}, "
        f"F1-macro: {f1_macro:.4f}"
    )

    return {"accuracy": accuracy, "f1": f1_weighted, "f1_macro": f1_macro}


def compute_loss_weights(labels: List[int], num_labels: int) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.
    Uses inverse frequency weighting with normalization.

    Args:
        labels: List of integer labels
        num_labels: Total number of classes

    Returns:
        Tensor of class weights
    """
    # Count occurrences of each label
    counts = np.bincount(labels, minlength=num_labels)

    # Log class distribution
    logger.info("Class distribution:")
    for i, count in enumerate(counts):
        percentage = (count / len(labels)) * 100
        logger.info(f"  Class {i}: {count:5d} samples ({percentage:5.2f}%)")

    # Handle zero counts to avoid division by zero
    safe_counts = np.where(counts == 0, 1, counts)

    # Compute Inverse Class Frequency
    total = len(labels)
    weights = total / (num_labels * safe_counts.astype(np.float32))

    # Normalize to prevent exploding gradients for rare classes
    weights = normalize_weights(weights, min_val=0.5, max_val=2.0)

    # Ensure mean is 1.0 to maintain loss scale
    weights = weights / weights.mean()

    logger.info(
        f"Computed class weights - Min: {weights.min():.3f}, "
        f"Max: {weights.max():.3f}, Mean: {weights.mean():.3f}"
    )

    return torch.tensor(weights, dtype=torch.float32)


def normalize_weights(
    weights: np.ndarray, min_val: float = 0.5, max_val: float = 2.0
) -> np.ndarray:
    """
    Apply min-max normalization to weights.

    Args:
        weights: Array of weights to normalize
        min_val: Minimum value after normalization
        max_val: Maximum value after normalization

    Returns:
        Normalized weights array
    """
    w_min, w_max = weights.min(), weights.max()

    # Handle edge case where all weights are equal
    if w_max - w_min < 1e-6:
        normalized = np.ones_like(weights) * ((min_val + max_val) / 2)
        logger.debug(f"All weights equal, using uniform value: {normalized[0]:.3f}")
        return normalized

    # Min-max normalization
    normalized = (weights - w_min) / (w_max - w_min)
    normalized = min_val + (max_val - min_val) * normalized

    logger.debug(
        f"Normalized weights - Range: [{normalized.min():.3f}, {normalized.max():.3f}]"
    )

    return normalized


def validate_input_data(texts: List[str], max_length: int = 512) -> List[str]:
    """
    Validate and sanitize input text data.

    Args:
        texts: List of input text strings
        max_length: Maximum allowed length per text

    Returns:
        List of validated and truncated texts
    """
    validated = []
    truncated_count = 0

    for i, text in enumerate(texts):
        # Convert to string and strip whitespace
        text_str = str(text).strip()

        # Truncate if too long
        if len(text_str) > max_length:
            text_str = text_str[:max_length]
            truncated_count += 1

        validated.append(text_str)

    if truncated_count > 0:
        logger.warning(
            f"Truncated {truncated_count} texts exceeding {max_length} characters"
        )

    logger.debug(f"Validated {len(validated)} input texts")
    return validated


def log_model_info(model, tokenizer=None):
    """
    Log information about the model and tokenizer.

    Args:
        model: The model instance
        tokenizer: Optional tokenizer instance
    """
    logger.info("=" * 70)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 70)

    # Model architecture
    if hasattr(model, "config"):
        config = model.config
        logger.info(
            f"Model Type: {config.model_type if hasattr(config, 'model_type') else 'Unknown'}"
        )
        logger.info(
            f"Hidden Size: {config.hidden_size if hasattr(config, 'hidden_size') else 'Unknown'}"
        )
        logger.info(
            f"Num Labels: {config.num_labels if hasattr(config, 'num_labels') else 'Unknown'}"
        )
        logger.info(
            f"Max Position Embeddings: {config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 'Unknown'}"
        )

    # Model parameters
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Trainable Ratio: {trainable_params/total_params*100:.2f}%")
    except Exception as e:
        logger.warning(f"Could not compute parameter counts: {e}")

    # Tokenizer info
    if tokenizer:
        logger.info(f"Tokenizer Vocab Size: {len(tokenizer)}")
        logger.info(f"Max Length: {tokenizer.model_max_length}")
        logger.info(f"Padding Side: {tokenizer.padding_side}")

    logger.info("=" * 70)


def log_training_config(config):
    """
    Log training configuration in a readable format.

    Args:
        config: TrainingConfig instance
    """
    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)

    # Data configuration
    logger.info("Data Configuration:")
    logger.info(f"  Input File: {config.MAIN_DATA_FILE}")
    logger.info(f"  Test Size: {config.TEST_SIZE * 100:.1f}%")
    logger.info(f"  Random Seed: {config.SEED}")

    # Model configuration
    logger.info("Model Configuration:")
    logger.info(f"  Architecture: {config.MODEL_NAME}")
    logger.info(f"  Max Length: {config.MAX_LEN}")

    # Training hyperparameters
    logger.info("Training Hyperparameters:")
    logger.info(f"  Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  Epochs: {config.EPOCHS}")
    logger.info(f"  Learning Rate: {config.LEARNING_RATE:.2e}")
    logger.info(f"  Weight Decay: {config.WEIGHT_DECAY}")
    logger.info(f"  Warmup Ratio: {config.WARMUP_RATIO}")
    logger.info(f"  Gradient Clip: {config.GRADIENT_CLIP}")

    # Advanced features
    logger.info("Advanced Features:")
    logger.info(f"  Class Weights: {config.USE_CLASS_WEIGHTS}")
    logger.info(f"  Mixed Precision: {config.USE_MIXED_PRECISION}")
    logger.info(f"  Label Smoothing: {config.LABEL_SMOOTHING}")
    logger.info(f"  Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")

    # Cross-validation
    if hasattr(config, "K_FOLDS"):
        logger.info("Cross-Validation:")
        logger.info(f"  K-Folds: {config.K_FOLDS}")
        logger.info(f"  Optuna Trials: {config.OPTUNA_N_TRIALS}")

    # Output paths
    logger.info("Output Paths:")
    logger.info(f"  Model Save Path: {config.MODEL_SAVE_PATH}")
    logger.info(f"  Checkpoint Dir: {config.CHECKPOINT_DIR}")

    logger.info("=" * 70)


def log_inference_config(config):
    """
    Log inference configuration in a readable format.

    Args:
        config: InferenceConfig instance
    """
    logger.info("=" * 70)
    logger.info("INFERENCE CONFIGURATION")
    logger.info("=" * 70)

    logger.info("Model Paths:")
    logger.info(f"  Classifier: {config.MODEL_PATH}")
    logger.info(f"  LLM: {config.LLM_MODEL_ID}")
    logger.info(f"  Intent Descriptions: {config.INTENT_DESCRIPTION_PATH}")

    logger.info("Classification Parameters:")
    logger.info(f"  Temperature: {config.TEMPERATURE}")
    logger.info(f"  Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
    logger.info(f"  Batch Size: {config.BATCH_SIZE}")

    logger.info("Normalization Settings:")
    logger.info(f"  Max New Tokens: {config.LLM_MAX_NEW_TOKENS}")
    logger.info(f"  Temperature: {config.LLM_TEMPERATURE}")

    logger.info("Output Settings:")
    logger.info(f"  Max Text Length: {config.MAX_TEXT_LENGTH}")

    logger.info("=" * 70)


def create_progress_logger(total: int, description: str = "Processing"):
    """
    Create a progress logger for long-running operations.

    Args:
        total: Total number of items to process
        description: Description of the operation

    Returns:
        ProgressLogger instance
    """
    from logging_config import ProgressLogger

    return ProgressLogger(logger, total, description)
