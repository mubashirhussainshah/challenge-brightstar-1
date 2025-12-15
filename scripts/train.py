import argparse
import logging
import os
import sys

# Add project root to sys.path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import TrainingConfig
from src.data import load_and_prep_data
from src.training import train_final_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_training_pipeline(
    config_path: str, input_file: str = None, output_dir: str = None
):
    """
    Orchestrates the training process.
    """

    # Load configuration
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        return

    logger.info(f"Loading configuration from {config_path}...")
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        config = TrainingConfig.from_yaml(config_path)
    else:
        config = TrainingConfig.from_json(config_path)

    # Apply CLI overrides
    if input_file:
        logger.info(f"Overriding input dataset from CLI: {input_file}")
        config.MAIN_DATA_FILE = input_file

    if output_dir:
        logger.info(f"Overriding model output dir from CLI: {output_dir}")
        config.MODEL_SAVE_PATH = output_dir

    logger.info("-" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info(f"Model Architecture: {config.MODEL_NAME}")
    logger.info(f"Input Data:         {config.MAIN_DATA_FILE}")
    logger.info(f"Output Directory:   {config.MODEL_SAVE_PATH}")
    logger.info("-" * 60)

    # Load and prepare data
    try:
        logger.info("Step 1/2: Preparing Dataset...")

        if not os.path.exists(config.MAIN_DATA_FILE):
            raise FileNotFoundError(
                f"Training data file not found at: {config.MAIN_DATA_FILE}"
            )

        train_df, test_df, label2id, id2label, num_labels = load_and_prep_data(config)

        logger.info(f"Data prepared successfully.")
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Test samples:  {len(test_df)}")
        logger.info(f"Num Labels:    {num_labels}")
    except Exception as e:
        logger.error(f"Failed during data preparation: {e}", exc_info=True)
        return

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
        )

        logger.info("-" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Model saved to: {model_path}")
        logger.info("-" * 60)

    except Exception as e:
        logger.error(f"Failed during model training: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BERT Training Pipeline")

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

    args = parser.parse_args()

    run_training_pipeline(args.config, args.input, args.output_dir)
