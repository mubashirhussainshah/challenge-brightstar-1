import json
import yaml
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class TrainingConfig:
    """Central configuration for the training pipeline"""

    # Data paths
    MAIN_DATA_FILE: str = "data/match_sentences.csv"
    TEXT_COLUMN: str = "text"
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
    OPTUNA_TIMEOUT: Optional[int] = None
    BEST_PARAMS_PATH: str = "checkpoints/best_params.json"

    # Output paths
    MODEL_SAVE_PATH: str = "models/italian_intent_model"
    CHECKPOINT_DIR: str = "checkpoints"

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate()

    def _validate(self):
        """Comprehensive validation of configuration parameters"""
        errors = []

        # Data validation
        if not 0 < self.TEST_SIZE < 1:
            errors.append(f"TEST_SIZE must be between 0 and 1, got {self.TEST_SIZE}")

        if self.SEED < 0:
            errors.append(f"SEED must be non-negative, got {self.SEED}")

        # Model validation
        if self.MAX_LEN < 1 or self.MAX_LEN > 512:
            errors.append(f"MAX_LEN must be between 1 and 512, got {self.MAX_LEN}")

        # Training hyperparameters validation
        if self.BATCH_SIZE < 1:
            errors.append(f"BATCH_SIZE must be positive, got {self.BATCH_SIZE}")

        if self.EPOCHS < 1:
            errors.append(f"EPOCHS must be positive, got {self.EPOCHS}")

        if self.LEARNING_RATE <= 0:
            errors.append(f"LEARNING_RATE must be positive, got {self.LEARNING_RATE}")

        if not 0 <= self.WEIGHT_DECAY <= 1:
            errors.append(
                f"WEIGHT_DECAY must be between 0 and 1, got {self.WEIGHT_DECAY}"
            )

        if not 0 <= self.WARMUP_RATIO <= 1:
            errors.append(
                f"WARMUP_RATIO must be between 0 and 1, got {self.WARMUP_RATIO}"
            )

        if self.GRADIENT_CLIP <= 0:
            errors.append(f"GRADIENT_CLIP must be positive, got {self.GRADIENT_CLIP}")

        # Advanced features validation
        if not 0 <= self.LABEL_SMOOTHING < 0.5:
            errors.append(
                f"LABEL_SMOOTHING must be between 0 and 0.5, got {self.LABEL_SMOOTHING}"
            )

        if self.EARLY_STOPPING_PATIENCE < 1:
            errors.append(
                f"EARLY_STOPPING_PATIENCE must be positive, got {self.EARLY_STOPPING_PATIENCE}"
            )

        # Cross-validation validation
        if self.K_FOLDS < 2:
            errors.append(f"K_FOLDS must be at least 2, got {self.K_FOLDS}")

        if self.OPTUNA_N_TRIALS < 1:
            errors.append(
                f"OPTUNA_N_TRIALS must be positive, got {self.OPTUNA_N_TRIALS}"
            )

        if self.OPTUNA_TIMEOUT is not None and self.OPTUNA_TIMEOUT < 1:
            errors.append(
                f"OPTUNA_TIMEOUT must be positive or None, got {self.OPTUNA_TIMEOUT}"
            )

        # Raise all errors at once for better user experience
        if errors:
            raise ValueError(
                f"Configuration validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {err}" for err in errors)
            )

        logging.debug("Configuration validation passed")

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
    INTENT_DESCRIPTION_PATH: str = "data/intent_description.json"
    EXAMPLES_PATH: str = "data/examples.json"

    # Inference parameters
    TEMPERATURE: float = 1.0
    CONFIDENCE_THRESHOLD: float = 0.65
    MAX_LEN: int = 128
    BATCH_SIZE: int = 32

    # LLM normalization settings
    LLM_MODEL_ID: str = "meta-llama/Llama-3.2-3B-Instruct"
    LLM_MAX_NEW_TOKENS: int = 64
    LLM_TEMPERATURE: float = 0.1
    NORMALIZE_BATCH_SIZE: int = 1

    # Output settings
    OUTPUT_FILENAME: str = "predictions_output.csv"
    SAVE_CHECKPOINTS: bool = True
    CHECKPOINT_DIR: str = "checkpoints"

    # Input validation
    MAX_TEXT_LENGTH: int = 512  # Maximum allowed input text length

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate()

    def _validate(self):
        """Comprehensive validation of configuration parameters"""
        errors = []

        # Inference parameters validation
        if self.TEMPERATURE <= 0:
            errors.append(f"TEMPERATURE must be positive, got {self.TEMPERATURE}")

        if not 0 <= self.CONFIDENCE_THRESHOLD <= 1:
            errors.append(
                f"CONFIDENCE_THRESHOLD must be between 0 and 1, got {self.CONFIDENCE_THRESHOLD}"
            )

        if self.MAX_LEN < 1 or self.MAX_LEN > 512:
            errors.append(f"MAX_LEN must be between 1 and 512, got {self.MAX_LEN}")

        if self.BATCH_SIZE < 1:
            errors.append(f"BATCH_SIZE must be positive, got {self.BATCH_SIZE}")

        # LLM parameters validation
        if self.LLM_MAX_NEW_TOKENS < 1:
            errors.append(
                f"LLM_MAX_NEW_TOKENS must be positive, got {self.LLM_MAX_NEW_TOKENS}"
            )

        if self.LLM_TEMPERATURE < 0:
            errors.append(
                f"LLM_TEMPERATURE must be non-negative, got {self.LLM_TEMPERATURE}"
            )

        if self.NORMALIZE_BATCH_SIZE < 1:
            errors.append(
                f"NORMALIZE_BATCH_SIZE must be positive, got {self.NORMALIZE_BATCH_SIZE}"
            )

        # Input validation
        if self.MAX_TEXT_LENGTH < 1:
            errors.append(
                f"MAX_TEXT_LENGTH must be positive, got {self.MAX_TEXT_LENGTH}"
            )

        # Logging validation
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.LOG_LEVEL.upper() not in valid_log_levels:
            errors.append(
                f"LOG_LEVEL must be one of {valid_log_levels}, got {self.LOG_LEVEL}"
            )

        # Raise all errors at once
        if errors:
            raise ValueError(
                f"Configuration validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {err}" for err in errors)
            )

        logging.debug("Configuration validation passed")

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


@dataclass
class DiscoveryConfig:
    """Configuration for intent discovery from UNLISTED sentences"""

    # Data
    INPUT_FILE: str = "data/unlisted_sentences.csv"
    TEXT_COLUMN: str = "text"

    OUTPUT_DIR: str = "reports"
    OUTPUT_CSV: str = "discovered_intents.csv"

    # Model selection
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    LLM_MODEL_ID: str = "meta-llama/Llama-3.2-3B-Instruct"

    # UMAP parameters (dimensionality reduction)
    UMAP_N_NEIGHBORS: int = 15  # Smaller = more local structure
    UMAP_N_COMPONENTS: int = 15  # Dimensions to reduce to
    UMAP_MIN_DIST: float = 0.0
    UMAP_METRIC: str = "cosine"

    # HDBSCAN parameters (clustering)
    MIN_CLUSTER_SIZE: int = 30  # Minimum sentences per cluster
    MIN_SAMPLES: int = 5  # Core point threshold
    HDBSCAN_METRIC: str = "euclidean"

    # LLM labeling parameters
    LLM_MAX_NEW_TOKENS: int = 30
    LLM_TEMPERATURE: float = 0.1

    # Text processing
    TOP_N_WORDS: int = 10
    MIN_DF: int = 2  # Minimum document frequency
    N_GRAM_RANGE: Tuple[int, int] = (1, 2)

    # Preprocessing
    MIN_TEXT_LENGTH: int = 5
    MAX_TEXT_LENGTH: int = 500
    MIN_WORDS_AFTER_CLEANING: int = 3

    # Minimum sentences needed for discovery
    MIN_SENTENCES_FOR_DISCOVERY: int = 100

    # General
    SEED: int = 42

    # Visualization
    CREATE_VISUALIZATIONS: bool = True
    VIZ_OUTPUT_DIR: str = "reports"

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate()

    def _validate(self):
        """Validate configuration parameters"""
        errors = []

        # UMAP validation
        if self.UMAP_N_NEIGHBORS < 2:
            errors.append(
                f"UMAP_N_NEIGHBORS must be at least 2, got {self.UMAP_N_NEIGHBORS}"
            )

        if self.UMAP_N_COMPONENTS < 2:
            errors.append(
                f"UMAP_N_COMPONENTS must be at least 2, got {self.UMAP_N_COMPONENTS}"
            )

        if not 0 <= self.UMAP_MIN_DIST <= 1:
            errors.append(
                f"UMAP_MIN_DIST must be between 0 and 1, got {self.UMAP_MIN_DIST}"
            )

        # HDBSCAN validation
        if self.MIN_CLUSTER_SIZE < 2:
            errors.append(
                f"MIN_CLUSTER_SIZE must be at least 2, got {self.MIN_CLUSTER_SIZE}"
            )

        if self.MIN_SAMPLES < 1:
            errors.append(f"MIN_SAMPLES must be at least 1, got {self.MIN_SAMPLES}")

        # LLM validation
        if self.LLM_MAX_NEW_TOKENS < 1:
            errors.append(
                f"LLM_MAX_NEW_TOKENS must be positive, got {self.LLM_MAX_NEW_TOKENS}"
            )

        if self.LLM_TEMPERATURE < 0:
            errors.append(
                f"LLM_TEMPERATURE must be non-negative, got {self.LLM_TEMPERATURE}"
            )

        # Text processing validation
        if self.TOP_N_WORDS < 1:
            errors.append(f"TOP_N_WORDS must be positive, got {self.TOP_N_WORDS}")

        if self.MIN_DF < 1:
            errors.append(f"MIN_DF must be positive, got {self.MIN_DF}")

        if (
            not isinstance(self.N_GRAM_RANGE, tuple)
            or len(self.N_GRAM_RANGE) != 2
            or self.N_GRAM_RANGE[0] > self.N_GRAM_RANGE[1]
        ):
            errors.append(f"Invalid N_GRAM_RANGE: {self.N_GRAM_RANGE}")

        # General validation
        if self.MIN_SENTENCES_FOR_DISCOVERY < self.MIN_CLUSTER_SIZE:
            errors.append(
                f"MIN_SENTENCES_FOR_DISCOVERY ({self.MIN_SENTENCES_FOR_DISCOVERY}) "
                f"must be >= MIN_CLUSTER_SIZE ({self.MIN_CLUSTER_SIZE})"
            )

        if errors:
            raise ValueError(
                f"Configuration validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {err}" for err in errors)
            )

        logging.debug("DiscoveryConfig validation passed")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DiscoveryConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, filepath: str) -> "DiscoveryConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, filepath: str) -> "DiscoveryConfig":
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
