import argparse
import os
import sys
from typing import Optional

# Add project root to sys.path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import TrainingConfig
from src.utils import setup_logging, log_training_config
from src.logging_config import LoggingConfig
from src.data import load_and_prep_data
from src.training import train_final_model


def run_training_pipeline(
    config_path: str,
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    Orchestrates the training process.
    """

    setup_logging(level=log_level, log_file=log_file)
    logger = LoggingConfig.get_logger(__name__)

    # Load configuration
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        return 1

    logger.info(f"Loading configuration from {config_path}...")

    try:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            config = TrainingConfig.from_yaml(config_path)
        else:
            config = TrainingConfig.from_json(config_path)
        logger.info("Configuration loaded and validated successfully")
    except ValueError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return 1

    # Apply CLI overrides
    if input_file:
        logger.info(f"Overriding input dataset from CLI: {input_file}")
        config.MAIN_DATA_FILE = input_file

    if output_dir:
        logger.info(f"Overriding model output dir from CLI: {output_dir}")
        config.MODEL_SAVE_PATH = output_dir

    log_training_config(config)

    # Load and prepare data
    try:
        logger.info("Step 1/2: Preparing Dataset...")

        if not os.path.exists(config.MAIN_DATA_FILE):
            raise FileNotFoundError(
                f"Training data file not found at: {config.MAIN_DATA_FILE}"
            )

        train_df, test_df, label2id, id2label, num_labels = load_and_prep_data(
            config, logger
        )

        logger.info(f"Data prepared successfully.")
        logger.info(f"  Train samples: {len(train_df)}")
        logger.info(f"  Test samples:  {len(test_df)}")
        logger.info(f"  Num Labels:    {num_labels}")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed during data preparation: {e}", exc_info=True)
        return 1

    # Train model
    try:
        logger.info("-" * 60)
        logger.info("Step 2/2: Training Model...")

        model_path = train_final_model(
            config=config,
            df_trainval=train_df,
            label2id=label2id,
            id2label=id2label,
            num_labels=num_labels,
            text_column=config.RAW_TEXT_COLUMN,
            logger=logger,
        )

        logger.info("-" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Model saved to: {model_path}")
        logger.info("-" * 60)

        return 0

    except Exception as e:
        logger.error(f"Failed during model training: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BERT Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default config
  python scripts/train.py
  
  # With custom config
  python scripts/train.py --config configs/my_config.yaml
  
  # Override input data
  python scripts/train.py --input data/my_data.csv
  
  # Debug mode with file logging
  python scripts/train.py --log-level DEBUG --log-file logs/debug.log
        """,
    )

    # Config is still required as the "base"
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration file (YAML or JSON)",
    )

    # Optional override for input
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input CSV training data. Overrides MAIN_DATA_FILE in config.",
    )

    # Optional override for output
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Path to save the trained model. Overrides MODEL_SAVE_PATH in config.",
    )

    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. If not provided, logs only to console.",
    )

    args = parser.parse_args()

    # Run the pipeline and exit with appropriate code
    exit_code = run_training_pipeline(
        config_path=args.config,
        input_file=args.input,
        output_dir=args.output_dir,
        log_level=args.log_level,
        log_file=args.log_file,
    )

    sys.exit(exit_code)
