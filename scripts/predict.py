import sys
import os
import gc
import json
import logging
import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import InferenceConfig
from src.predictor import IntentPredictor
from src.normalizer import ItalianSentenceNormalizer
from src.utils import setup_logging, log_inference_config, validate_input_data
from src.logging_config import LoggingConfig


def prepare_input_dataframe(filepath: str) -> pd.DataFrame:
    """Handles CSV loading, column mapping, and filling missing values."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except pd.errors.ParserError:
        df = pd.read_csv(filepath, sep=";")

    col_map = {
        "Phrase": "Phrase",
        "phrase": "Phrase",
        "Testo": "Phrase",
        "text": "Phrase",
        "UserUtterance": "Phrase",
        "Interaction ID": "Interaction ID",
        "ID": "Interaction ID",
        "Topic title": "Topic bot",
        "Bot Topic": "Topic bot",
    }
    df = df.rename(columns=col_map)

    if "Phrase" not in df.columns:
        raise ValueError(
            f"Input CSV {filepath} must contain a column named 'Phrase'."
            f"Found columns: {list(df.columns)}"
        )

    # Optional columns filling
    if "Interaction ID" not in df.columns:
        df["Interaction ID"] = [f"sim_{i+1}" for i in range(len(df))]
    if "Date/time" not in df.columns:
        df["Date/time"] = [f"timestamp_{i+1}" for i in range(len(df))]
    if "Topic bot" not in df.columns:
        df["Topic bot"] = ""

    # Fill NaNs
    df["Interaction ID"] = df["Interaction ID"].fillna("unknown")
    df["Topic bot"] = df["Topic bot"].fillna("")
    df["Date/time"] = df["Date/time"].fillna("unknown")

    return df


def calculate_summary_status(
    prediction_status: str, bot_topic: str, tool_label: str
) -> str:
    """Determines the final MATCH/MISMATCH status."""
    if prediction_status == "VALID":
        if not bot_topic:
            return "MATCH"
        elif bot_topic == tool_label:
            return "MATCH_BOT"
        else:
            return "MISMATCH"
    return prediction_status


def run_inference_pipeline(
    config: InferenceConfig, input_path: str, output_path: str, logger
):
    """
    Runs the pipeline using the provided config and explicit file paths.
    """
    logger.info("-" * 60)
    logger.info("STARTING INFERENCE PIPELINE")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info("-" * 60)

    # Load data
    try:
        logger.info("")
        logger.info("STEP 1/3: LOADING INPUT DATA")
        logger.info("-" * 70)

        df = prepare_input_dataframe(input_path)
        phrases = df["Phrase"].astype(str).tolist()

        # Validate and sanitize input
        phrases = validate_input_data(phrases, max_length=config.MAX_TEXT_LENGTH)

        logger.info(f"Loaded {len(phrases)} phrases")

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load input data: {e}", exc_info=True)
        raise

    # Run classification
    try:
        logger.info("")
        logger.info("STEP 2/3: INTENT CLASSIFICATION")
        logger.info("-" * 70)
        logger.info(f"Loading BERT model from: {config.MODEL_PATH}")

        classifier = IntentPredictor(config.MODEL_PATH)

        logger.info(f"Running classification (batch_size={config.BATCH_SIZE})...")
        predictions = classifier.predict(
            phrases,
            batch_size=config.BATCH_SIZE,
            threshold=config.CONFIDENCE_THRESHOLD,
            temperature=config.TEMPERATURE,
        )

        predicted_labels = [p[0] for p in predictions]

        # Log classification statistics
        unique_labels, counts = np.unique(predicted_labels, return_counts=True)
        logger.info(f"Classification complete")
        logger.info("Predicted intent distribution:")
        for label, count in sorted(zip(unique_labels, counts), key=lambda x: -x[1])[:5]:
            percentage = (count / len(predictions)) * 100
            logger.info(f"  {label:30s}: {count:4d} ({percentage:5.1f}%)")

    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise

    # Normalization
    try:
        logger.info("")
        logger.info("STEP 3/3: TEXT NORMALIZATION")
        logger.info("-" * 70)

        # Memory management: Unload classifier before loading LLM
        logger.info("Unloading classifier to free VRAM...")
        del classifier
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.debug(
            f"VRAM freed: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated"
        )

        # Load intent descriptions
        intents_context = []
        if os.path.exists(config.INTENT_DESCRIPTION_PATH):
            logger.info(
                f"Loading intent descriptions from: {config.INTENT_DESCRIPTION_PATH}"
            )
            with open(config.INTENT_DESCRIPTION_PATH, "r", encoding="utf-8") as f:
                intents_context = json.load(f)
            logger.info(f"✓ Loaded {len(intents_context)} intent descriptions")
        else:
            logger.warning(
                f"Intent description file not found: {config.INTENT_DESCRIPTION_PATH}"
            )
            logger.warning("Normalizing without intent context")

        # Initialize normalizer
        logger.info(f"Loading normalizer LLM: {config.LLM_MODEL_ID}")

        # Load prompt config if specified
        prompt_config_path = getattr(config, "NORMALIZER_PROMPT_CONFIG", None)
        if prompt_config_path and os.path.exists(prompt_config_path):
            logger.info(f"Using prompt config: {prompt_config_path}")
            normalizer = ItalianSentenceNormalizer(
                config.LLM_MODEL_ID, intent_list=intents_context
            )
        else:
            if prompt_config_path:
                logger.warning(
                    f"Prompt config not found: {prompt_config_path}, using defaults"
                )
            normalizer = ItalianSentenceNormalizer(
                config.LLM_MODEL_ID, intent_list=intents_context
            )

        logger.info("Normalizer loaded")

        # Run normalization
        logger.info("Starting normalization (this may take a while)...")
        normalized_phrases = []

        for phrase, intent in tqdm(
            zip(phrases, predicted_labels),
            total=len(phrases),
            desc="Normalizing",
            disable=False,
        ):
            # Only pass intent context for valid intents
            intent_context = intent if intent not in ["UNLISTED", "NOMATCH"] else None

            norm_text = normalizer.normalize_sentence(
                phrase,
                predicted_intent=intent_context,
                temperature=config.LLM_TEMPERATURE,
                max_new_tokens=config.LLM_MAX_NEW_TOKENS,
            )
            normalized_phrases.append(norm_text)

        logger.info("Normalization complete")

    except Exception as e:
        logger.error(f"Normalization failed: {e}", exc_info=True)
        raise

    # Construct final output
    try:
        logger.info("")
        logger.info("FINALIZING OUTPUT")
        logger.info("-" * 70)

        output_rows = []
        for i, (tool_label, status, conf) in enumerate(predictions):
            row = df.iloc[i]
            bot_topic = str(row["Topic bot"]).strip()
            final_summary = calculate_summary_status(status, bot_topic, tool_label)

            # Apply confidence scaling from config
            scaled_confidence = round(conf * 10, 2)

            output_rows.append(
                {
                    "Interaction ID": row["Interaction ID"],
                    "Date/time": row["Date/time"],
                    "Phrase": row["Phrase"],
                    "Topic bot": bot_topic,
                    "Phrase normalized": normalized_phrases[i],
                    "Topic tool": tool_label,
                    "Summary": final_summary,
                    "Confidence": scaled_confidence,
                }
            )

        result_df = pd.DataFrame(output_rows)

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)

        logger.info(f"✓ Results saved to: {output_path}")

        # Summary statistics
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE - SUMMARY STATISTICS")
        logger.info("=" * 70)

        summary_counts = result_df["Summary"].value_counts()
        for status, count in summary_counts.items():
            percentage = (count / len(result_df)) * 100
            logger.info(f"  {status:15s}: {count:4d} ({percentage:5.1f}%)")

        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to save output: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Intent Classification & Normalization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/predict.py --input data/test.csv
  
  # With custom config
  python scripts/predict.py -i data/test.csv -c configs/my_config.yaml
  
  # With custom output path
  python scripts/predict.py -i data/test.csv -o results/predictions.csv
  
  # Debug mode with file logging
  python scripts/predict.py -i data/test.csv --log-level DEBUG --log-file logs/debug.log
        """,
    )

    # Required argument: input file
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the input CSV file containing phrases.",
    )

    # Optional: config file
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/inference_config.yaml",
        help="Path to inference configuration YAML (default: configs/inference_config.yaml)",
    )

    # Optional: output file
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save the results CSV. If not provided, auto-generated from input name.",
    )

    # Logging configuration options
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

    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger = LoggingConfig.get_logger(__name__)

    # Load and validate configuration
    try:
        if not os.path.exists(args.config):
            logger.warning(f"Config file not found: {args.config}")
            logger.info("Using default configuration")
            config = InferenceConfig()
        else:
            logger.info(f"Loading configuration from: {args.config}")
            config = InferenceConfig.from_yaml(args.config)
            logger.info("Configuration loaded and validated")

    except ValueError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)

    # Log the configuration
    log_inference_config(config)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate: "data/input.csv" -> "data/input_predictions.csv"
        input_p = Path(args.input)
        output_path = str(input_p.with_name(f"{input_p.stem}_predictions.csv"))

    logger.info(f"Output will be saved to: {output_path}")

    # Run pipeline
    try:
        run_inference_pipeline(config, args.input, output_path, logger)
        logger.info("")
        logger.info("Pipeline completed successfully!")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
