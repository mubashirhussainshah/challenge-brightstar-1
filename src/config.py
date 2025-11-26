import os
import json


class Config:
    """
    Configuration class for the training pipeline
    """

    # Paths (relative to project root)
    DATA_DIR = "data"
    OUTPUT_DIR = "models"
    MAIN_DATA_FILE = os.path.join(DATA_DIR, "match_sentences.csv")
    PRIORITY_SCORES_PATH = os.path.join(DATA_DIR, "intent_description.json")
    TEST_SET_PATH = os.path.join(DATA_DIR, "test_set.csv")
    TRAINVAL_SET_PATH = os.path.join(DATA_DIR, "trainval_set.csv")
    PLOTS_SAVE_PATH = os.path.join(DATA_DIR, "plots")
    BEST_PARAMS_PATH = os.path.join(DATA_DIR, "best_hyperparameters.json")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "italian_intent_model")

    # Dataset parameters
    RAW_TEXT_COLUMN = "text"
    LABEL_COLUMN = "intent"
    TEST_SIZE = 0.15
    SEED = 42

    # Model
    MODEL_NAME = "dbmdz/bert-base-italian-xxl-cased"
    MAX_LEN = 128

    # Training hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 4.75e-5
    WEIGHT_DECAY = 1.75e-4
    WARMUP_RATIO = 1.75e-4
    GRADIENT_CLIP = 1.0

    # Advanced feature toggles
    USE_CLASS_WEIGHTS = True
    USE_PRIORITY_SCORES = False
    USE_MIXED_PRECISION = True
    USE_FOCAL_LOSS = True
    USE_LABEL_SMOOTHING = True
    USE_RDROP = False

    # Loss params
    EARLY_STOPPING_PATIENCE = 10
    ALPHA = 0.776
    FOCAL_GAMMA = 1.38
    LABEL_SMOOTHING = 0.092

    # Cross-validation and hyperparameter tuning
    K_FOLDS = 5
    OPTUNA_N_TRIALS = 30
    OPTUNA_TIMEOUT = None  # Optional timeout in seconds
    USE_PRUNING = True

    # Inference params
    TEMPERATURE = 0.8
    CONFIDENCE_THRESHOLD = 0.5

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
