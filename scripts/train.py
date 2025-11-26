import sys
import os
import logging
import warnings

# Add the project root to python path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.utils import setup_logging
from src.data import load_and_prep_data, debug_dataset
from src.contrastive import train_contrastive_embeddings
from src.training import train_final_model
from evaluation import evaluate_on_test_set


def main():
    setup_logging()
    warnings.filterwarnings("ignore")

    config = Config()

    # Update config paths for script execution
    config.MODEL_SAVE_PATH = os.path.join(config.OUTPUT_DIR, "italian_intent_model")

    logging.info("Starting Training Pipeline")

    # Load Data
    train_df, test_df, l2id, id2l, n_labels = load_and_prep_data(config)

    DEBUG_MODE = True

    if DEBUG_MODE:
        logging.info("=" * 70)
        logging.info("DEBUG MODE")
        logging.info("=" * 70)
        train_df = debug_dataset(
            train_df,
            label_col=config.LABEL_COLUMN,
            n_per_class=10,
        )
        test_df = debug_dataset(test_df, config.LABEL_COLUMN, n_per_class=5)
        config.EPOCHS = 2

    # Contrastive learning
    # contrastive_path = os.path.join(config.OUTPUT_DIR, "contrastive_bert")
    # if not os.path.exists(contrastive_path):
    #     train_contrastive_embeddings(
    #         config,
    #         train_df,
    #         config.RAW_TEXT_COLUMN,
    #         config.LABEL_COLUMN,
    #         output_path=contrastive_path,
    #     )
    #
    # config.MODEL_NAME = contrastive_path

    # Final Training
    model_path = train_final_model(
        config,
        train_df,
        l2id,
        id2l,
        n_labels,
        config.RAW_TEXT_COLUMN,
        model_save_suffix="_final",
    )

    # Calibration & Evaluation
    evaluate_on_test_set(
        config,
        test_df,
        model_path,
        l2id,
        id2l,
        n_labels,
        config.LABEL_COLUMN,
        report_suffix="_final",
    )


if __name__ == "__main__":
    main()
