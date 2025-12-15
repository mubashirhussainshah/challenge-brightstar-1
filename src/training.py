import copy
import gc
import logging
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    IntervalStrategy,
)
from sklearn.model_selection import train_test_split
from .data import IntentDataset
from .utils import compute_loss_weights, compute_metrics
from .trainer import CustomTrainer
from .config import TrainingConfig
from typing import List, Dict, Optional


def finetune_model(
    config: TrainingConfig,
    tokenizer: AutoTokenizer,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    label2id: Dict,
    id2label: Dict,
    num_labels: int,
    main_model: Optional[AutoModelForSequenceClassification] = None,
    fold_id: str = "0",
    save_model: bool = False,
    model_save_path: Optional[str] = None,
) -> float:
    """Train model on a single fold"""
    logging.info(f"Training Fold {fold_id}...")

    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, config.MAX_LEN)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer, config.MAX_LEN)

    if main_model is not None:
        model = copy.deepcopy(main_model)
    else:
        # Load from disk (fallback)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
    model.gradient_checkpointing_enable()

    # Compute loss weights
    final_weights = None
    if config.USE_CLASS_WEIGHTS:
        final_weights = compute_loss_weights(labels=train_labels, num_labels=num_labels)

    # Training arguments
    args = TrainingArguments(
        output_dir=f"{config.CHECKPOINT_DIR}/fold_{fold_id}",
        overwrite_output_dir=True,
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        greater_is_better=True,
        fp16=config.USE_MIXED_PRECISION,
        logging_steps=50,
        report_to="none",
        disable_tqdm=False,
    )

    # Select trainer class
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE
            )
        ],
        loss_weights=final_weights,
        label_smoothing=config.LABEL_SMOOTHING,
    )

    # Train
    trainer.train()

    # Get best F1
    log_history = trainer.state.log_history
    eval_f1_scores = [entry["eval_f1"] for entry in log_history if "eval_f1" in entry]
    best_f1 = max(eval_f1_scores) if eval_f1_scores else trainer.evaluate()["eval_f1"]

    logging.info(f"Fold {fold_id}: Best F1 = {best_f1:.4f}")

    if save_model and model_save_path:
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        logging.info(f"Model saved to {model_save_path}")

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return best_f1


def train_final_model(
    config: TrainingConfig,
    df_trainval: pd.DataFrame,
    label2id: Dict,
    id2label: Dict,
    num_labels: int,
    text_column: str,
    model_save_suffix: str = "",
) -> str:
    """Train final model on full trainval set with best hyperparameters"""
    model_save_path = f"{config.MODEL_SAVE_PATH}{model_save_suffix}"

    logging.info("=" * 70)
    logging.info("TRAINING FINAL MODEL")
    logging.info("=" * 70)
    logging.info(f"Using optimized hyperparameters:")
    logging.info(
        f"  LR={config.LEARNING_RATE:.2e}, BS={config.BATCH_SIZE}, WD={config.WEIGHT_DECAY:.3f}"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Internal train/val split for early stopping
    train_df, val_df = train_test_split(
        df_trainval, test_size=0.1, random_state=42, stratify=df_trainval["labels"]
    )

    train_texts = train_df[text_column].tolist()
    train_labels = train_df["labels"].tolist()
    val_texts = val_df[text_column].tolist()
    val_labels = val_df["labels"].tolist()

    best_f1 = finetune_model(
        config=config,
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        label2id=label2id,
        id2label=id2label,
        num_labels=num_labels,
        fold_id="FINAL",
        save_model=True,
        model_save_path=model_save_path,
    )

    logging.info(f"Final model training complete. Val F1: {best_f1:.4f}")
    logging.info(f"Model saved to: {model_save_path}")

    # Save configuration
    config.to_json(f"{model_save_path}/training_config.json")

    return model_save_path
