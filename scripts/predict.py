import sys
import os
import gc
import json
import logging
import torch
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import InferenceConfig
from src.predictor import IntentPredictor
from src.normalizer import ItalianSentenceNormalizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def prepare_input_dataframe(filepath: str) -> pd.DataFrame:
    """Handles CSV loading, column mapping, and filling missing values."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    try:
        df = pd.read_csv(filepath)
    except:
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
        raise ValueError(f"Input CSV {filepath} must contain a column named 'Phrase'.")

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


def run_inference_pipeline(config: InferenceConfig, input_path: str, output_path: str):
    """
    Runs the pipeline using the provided config and explicit file paths.
    """
    logger.info("-" * 60)
    logger.info("STARTING INTEGRATED PIPELINE")
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info("-" * 60)

    # Load data
    try:
        df = prepare_input_dataframe(input_path)
        phrases = df["Phrase"].astype(str).tolist()
        logger.info(f"Loaded {len(phrases)} phrases.")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    # Run classification
    logger.info(f"Loading BERT model from {config.MODEL_PATH}...")
    classifier = IntentPredictor(config.MODEL_PATH)

    predictions = classifier.predict(
        phrases, threshold=config.CONFIDENCE_THRESHOLD, temperature=config.TEMPERATURE
    )

    predicted_labels = [p[0] for p in predictions]

    # Memory management
    logger.info("Unloading Classifier to free VRAM for LLM...")
    del classifier
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

    # Run normalization
    logger.info("Loading Intent Descriptions...")
    intents_context = []
    if os.path.exists(config.INTENT_DESCRIPTION_PATH):
        with open(config.INTENT_DESCRIPTION_PATH, "r", encoding="utf-8") as f:
            intents_context = json.load(f)
    else:
        logger.warning(
            "Intent description file not found. Normalizing without context."
        )

    logger.info(f"Loading Normalizer LLM ({config.LLM_MODEL_ID})...")
    normalizer = ItalianSentenceNormalizer(
        config.LLM_MODEL_ID, intent_list=intents_context
    )

    normalized_phrases = []
    logger.info("Starting Normalization...")

    for phrase, intent in tqdm(
        zip(phrases, predicted_labels), total=len(phrases), desc="Normalizing"
    ):
        intent_context = intent if intent not in ["UNLISTED", "NOMATCH"] else None

        norm_text = normalizer.normalize_sentence(
            phrase,
            predicted_intent=intent_context,
            temperature=config.LLM_TEMPERATURE,
            max_new_tokens=config.LLM_MAX_NEW_TOKENS,
        )
        normalized_phrases.append(norm_text)

    # Construct final output
    output_rows = []
    for i, (tool_label, status, conf) in enumerate(predictions):
        row = df.iloc[i]
        bot_topic = str(row["Topic bot"]).strip()
        final_summary = calculate_summary_status(status, bot_topic, tool_label)

        output_rows.append(
            {
                "Interaction ID": row["Interaction ID"],
                "Date/time": row["Date/time"],
                "Phrase": row["Phrase"],
                "Topic bot": bot_topic,
                "Phrase normalized": normalized_phrases[i],
                "Topic tool": tool_label,
                "Summary": final_summary,
                "Confidence": round(conf * 10, 2),
            }
        )

    result_df = pd.DataFrame(output_rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    logger.info("=" * 30)
    logger.info(f"Report saved to: {output_path}")
    logger.info("\nSUMMARY STATISTICS:")
    print(result_df["Summary"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference Pipeline")

    # Required argument: the input file
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the input CSV file containing phrases.",
    )

    # Optional argument: the config file (defaults to standard location)
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/inference_config.yaml",
        help="Path to model configuration YAML.",
    )

    # Optional argument: the output file
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save the results CSV. If not provided, it will be auto-generated based on input name.",
    )

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = InferenceConfig.from_yaml(args.config)
    else:
        logger.warning(f"Config file {args.config} not found. Using defaults.")
        config = InferenceConfig()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate: "data/raw_file.csv" -> "data/raw_file_predictions.csv"
        input_p = Path(args.input)
        output_path = str(input_p.with_name(f"{input_p.stem}_predictions.csv"))

    run_inference_pipeline(config, args.input, output_path)
