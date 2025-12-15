import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class TrainingConfig:
    """Central configuration for the training pipeline"""

    # Data paths
    MAIN_DATA_FILE: str = "data/match_sentences.csv"
    RAW_TEXT_COLUMN: str = "text"
    LABEL_COLUMN: str = "intent"

    TEST_SET_PATH: str = "data/processed/final_test_set.csv"
    TRAINVAL_SET_PATH: str = "data/processed/final_trainval_set.csv"

    TEST_SIZE: float = 0.15
    SEED: int = 42

    # Model
    MODEL_NAME: str = "dbmdz/bert-base-italian-xxl-cased"
    MAX_LEN: int = 128

    # Training hyperparameters
    BATCH_SIZE: int = 16
    EPOCHS: int = 10
    LEARNING_RATE: float = 3e-5
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.1
    GRADIENT_CLIP: float = 1.0

    # Advanced features
    USE_CLASS_WEIGHTS: bool = True
    USE_MIXED_PRECISION: bool = True
    LABEL_SMOOTHING: float = 0.1
    EARLY_STOPPING_PATIENCE: int = 5

    # Cross-validation
    K_FOLDS: int = 5
    OPTUNA_N_TRIALS: int = 30

    # Output paths
    MODEL_SAVE_PATH: str = "models/italian_intent_model"
    CHECKPOINT_DIR: str = "checkpoints"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, filepath: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, filepath: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def override(self, **kwargs) -> "TrainingConfig":
        """Return a new config instance with updated values."""
        current_dict = asdict(self)
        current_dict.update(kwargs)
        return TrainingConfig.from_dict(current_dict)


@dataclass
class InferenceConfig:
    """Central configuration for the inference pipeline"""

    # Model paths
    MODEL_PATH: str = "models/italian_intent_model"
    LLM_MODEL_ID: str = "meta-llama/Llama-3.2-3B-Instruct"
    INTENT_DESCRIPTION_PATH: str = "data/intent_description.json"

    # Inference parameters
    TEMPERATURE: float = 1.0
    CONFIDENCE_THRESHOLD: float = 0.65
    MAX_LEN: int = 128
    BATCH_SIZE: int = 32

    # LLM normalization settings
    LLM_MAX_NEW_TOKENS: int = 64
    LLM_TEMPERATURE: float = 0.1
    NORMALIZE_BATCH_SIZE: int = 1

    # Output settings
    OUTPUT_FILENAME: str = "predictions_output.csv"
    SAVE_CHECKPOINTS: bool = True
    CHECKPOINT_DIR: str = "checkpoints"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "InferenceConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, filepath: str) -> "InferenceConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, filepath: str) -> "InferenceConfig":
        """Load configuration from YAML file."""
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def override(self, **kwargs) -> "InferenceConfig":
        """Create a new config with overridden values."""
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return InferenceConfig.from_dict(current_dict)
