import optuna
import copy
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold
import gc
import torch

from .config import Config
from .training import finetune_model


def run_hp_search_optuna(
    config: Config,
    df_trainval: pd.DataFrame,
    label2id: Dict,
    id2label: Dict,
    num_labels: int,
    text_column: str,
) -> Tuple[Config, pd.DataFrame]:
    """
    Optuna hyperparameter search with:
    - Pruning for efficiency
    - Better search space
    """
    logging.info("=" * 70)
    logging.info(f"OPTUNA HYPERPARAMETER SEARCH ({config.OPTUNA_N_TRIALS} trials)")
    logging.info("=" * 70)
    logging.info("Loading main Model and tokenizer into CPU memory...")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    main_model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    main_model.cpu()

    # Create cached datasets
    X = df_trainval[text_column].tolist()
    y = df_trainval["labels"].tolist()

    logging.info("Assets loaded. Starting Optimization.")

    def objective(trial: optuna.Trial) -> float:
        """Objective function with expanded search space"""

        # Sample hyperparameters
        trial_config = copy.deepcopy(config)

        # Core hyperparameters
        trial_config.LEARNING_RATE = trial.suggest_float(
            "learning_rate", 5e-6, 5e-5, log=True
        )
        trial_config.BATCH_SIZE = trial.suggest_categorical("batch_size", [16, 24, 32])
        trial_config.WEIGHT_DECAY = trial.suggest_float("weight_decay", 0.0, 0.15)
        trial_config.WARMUP_RATIO = trial.suggest_float("warmup_ratio", 0.05, 0.2)

        # Loss function parameters
        if config.USE_CLASS_WEIGHTS and config.USE_PRIORITY_SCORES:
            trial_config.ALPHA = trial.suggest_float("alpha", 0.2, 0.8)

        if config.USE_FOCAL_LOSS:
            trial_config.FOCAL_GAMMA = trial.suggest_float("focal_gamma", 0.5, 3.0)
            trial_config.LABEL_SMOOTHING = trial.suggest_float(
                "label_smoothing", 0.0, 0.1
            )

        # K-Fold cross-validation
        skf = StratifiedKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=42)
        fold_f1_scores = []

        try:
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                train_texts = [X[i] for i in train_idx]
                train_labels = [y[i] for i in train_idx]
                val_texts = [X[i] for i in val_idx]
                val_labels = [y[i] for i in val_idx]

                fold_id = f"T{trial.number}_F{fold+1}"

                # Train fold
                fold_f1 = finetune_model(
                    config=trial_config,
                    tokenizer=tokenizer,
                    train_texts=train_texts,
                    train_labels=train_labels,
                    val_texts=val_texts,
                    val_labels=val_labels,
                    label2id=label2id,
                    id2label=id2label,
                    num_labels=num_labels,
                    main_model=main_model,
                    fold_id=fold_id,
                    save_model=False,
                )
                fold_f1_scores.append(fold_f1)

                # Cleanup
                gc.collect()
                torch.cuda.empty_cache()

                # Optuna pruning (stop bad trials early)
                if config.USE_PRUNING:
                    current_avg_f1 = np.mean(fold_f1_scores)
                    trial.report(current_avg_f1, step=fold)
                    if trial.should_prune():
                        logging.info(f"Pruning Trial {trial.number} at Fold {fold+1}")
                        raise optuna.exceptions.TrialPruned()

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logging.error(f"Trial {trial.number} failed: {e}")
            return 0.0

        # Calculate average F1
        avg_f1 = float(np.mean(fold_f1_scores))
        std_f1 = np.std(fold_f1_scores)

        logging.info(f"Trial {trial.number}: F1={avg_f1:.4f} ± {std_f1:.4f}")

        return avg_f1

    # Create Optuna study with pruning
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=2, interval_steps=1
        )
        if config.USE_PRUNING
        else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        direction="maximize", pruner=pruner, study_name="bert_intent_classification"
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=config.OPTUNA_N_TRIALS,
        timeout=config.OPTUNA_TIMEOUT,
        show_progress_bar=True,
    )

    # Report results
    logging.info(f"\n{'=' * 70}")
    logging.info("OPTIMIZATION COMPLETE")
    logging.info(f"{'=' * 70}")
    logging.info(f"Best Trial: {study.best_trial.number}")
    logging.info(f"Best F1: {study.best_value:.4f}")
    logging.info(f"Best Params:")
    for key, value in study.best_params.items():
        logging.info(f"  {key}: {value}")
    logging.info(f"{'=' * 70}\n")

    # Update config with best params
    best_config = copy.deepcopy(config)
    for param, value in study.best_params.items():
        if hasattr(best_config, param.upper()):
            setattr(best_config, param.upper(), value)
        else:
            # Handle nested parameters
            param_map = {
                "learning_rate": "LEARNING_RATE",
                "batch_size": "BATCH_SIZE",
                "weight_decay": "WEIGHT_DECAY",
                "warmup_ratio": "WARMUP_RATIO",
                "alpha": "ALPHA",
                "focal_gamma": "FOCAL_GAMMA",
                "label_smoothing": "LABEL_SMOOTHING",
            }
            if param in param_map:
                setattr(best_config, param_map[param], value)

    # Save best hyperparameters
    with open(config.BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    logging.info(f"Best params saved to {config.BEST_PARAMS_PATH}")

    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_trials.csv", index=False)
    logging.info(f"Trials saved to optuna_trials.csv")

    return best_config, trials_df
