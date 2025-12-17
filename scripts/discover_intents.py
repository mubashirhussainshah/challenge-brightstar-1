import sys
import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
from bertopic import BERTopic

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DiscoveryConfig
from src.intent_discovery import IntentDiscoveryPipeline
from src.utils import setup_logging
from src.logging_config import LoggingConfig
from src.data import filter_unlisted_sentences


def print_discovery_report(
    pipeline: IntentDiscoveryPipeline,
    topics: list,
    probs,
    cleaned_texts: list,
    logger,
):
    """Print detailed discovery report"""
    topic_info = pipeline.get_topic_info()

    # Overview
    total = len(topics)
    n_topics = len([t for t in set(topics) if t != -1])
    n_outliers = sum(1 for t in topics if t == -1)

    logger.info("-" * 70)
    logger.info("INTENT DISCOVERY REPORT")
    logger.info("-" * 70)
    logger.info(f"Total sentences analyzed: {total}")
    logger.info(f"Discovered intents: {n_topics}")
    logger.info(f"Outliers (noise): {n_outliers} ({n_outliers/total*100:.1f}%)")
    logger.info(f"Coverage: {(total-n_outliers)/total*100:.1f}%")
    logger.info("-" * 70)

    # Show each discovered intent
    logger.info("\nDISCOVERED INTENTS:")
    logger.info("-" * 70)

    for _, row in topic_info[topic_info.Topic != -1].iterrows():
        topic_id = row["Topic"]
        count = row["Count"]
        percentage = (count / total) * 100

        # Get label (from LLM or keywords)
        label = row.get("Main", row.get("Name", f"Topic {topic_id}"))
        if isinstance(label, list):
            label = " ".join(label[:5])

        logger.info(f"\nIntent {topic_id}: {label}")
        logger.info(f"  Size: {count} sentences ({percentage:.1f}%)")

        # Get top keywords
        keywords = pipeline.topic_model.get_topic(topic_id)
        if keywords:
            keyword_str = ", ".join([word for word, score in keywords[:5]])
            logger.info(f"  Keywords: {keyword_str}")

        # Get representative sentences
        rep_sentences = pipeline.get_representative_sentences(
            topic_id, cleaned_texts, topics, probs, n=3
        )
        if rep_sentences:
            logger.info("  Examples:")
            for i, (sent, score) in enumerate(rep_sentences, 1):
                logger.info(f"    {i}. [{score:.3f}] {sent[:80]}...")

    logger.info("=" * 70)


def run_discovery_pipeline(
    config_path: str,
    input_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    text_column: str = "Phrase",
    enable_llm_labeling: bool = False,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    Main discovery pipeline orchestrator.

    Args:
        config_path: Path to discovery config YAML
        input_file: Optional override for input file
        output_dir: Optional override for output directory
        text_column: Name of text column in input CSV
        log_level: Logging level
        log_file: Optional log file path
    """
    # Setup logging
    setup_logging(level=log_level, log_file=log_file)
    logger = LoggingConfig.get_logger(__name__)

    logger.info("=" * 70)
    logger.info("INTENT DISCOVERY PIPELINE")
    logger.info("=" * 70)

    # Load configuration
    try:
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}")
            logger.info("Using default configuration")
            config = DiscoveryConfig()
        else:
            logger.info(f"Loading configuration from: {config_path}")
            config = DiscoveryConfig.from_yaml(config_path)
            logger.info("Configuration loaded and validated")
    except ValueError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        return 1

    # Apply CLI overrides
    if input_file:
        config.INPUT_FILE = input_file
    if output_dir:
        config.OUTPUT_DIR = output_dir

    # Load and filter data
    try:
        logger.info("")
        logger.info("STEP 1/4: LOADING DATA")
        logger.info("-" * 70)

        if not os.path.exists(config.INPUT_FILE):
            raise FileNotFoundError(f"Input file not found: {config.INPUT_FILE}")

        df_unlisted = filter_unlisted_sentences(
            config.INPUT_FILE, text_column, config.TEXT_COLUMN
        )

        if len(df_unlisted) < config.MIN_SENTENCES_FOR_DISCOVERY:
            logger.warning(
                f"Only {len(df_unlisted)} UNLISTED sentences found. "
                f"Need at least {config.MIN_SENTENCES_FOR_DISCOVERY} for reliable discovery."
            )
            logger.info("Consider lowering MIN_SENTENCES_FOR_DISCOVERY in config.")
            return 1

        texts = df_unlisted["text"].tolist()
        logger.info(f"Loaded {len(texts)} UNLISTED sentences for discovery")

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        return 1

    # Run discovery
    try:
        logger.info("")
        logger.info("STEP 2/4: DISCOVERING INTENTS")
        logger.info("-" * 70)

        pipeline = IntentDiscoveryPipeline(
            config, enable_llm_labeling=enable_llm_labeling
        )
        topics, probs, cleaned_texts, valid_indices = pipeline.discover_intents(texts)

    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=True)
        return 1

    # Evaluate clustering
    try:
        logger.info("")
        logger.info("STEP 3/4: EVALUATING CLUSTERS")
        logger.info("-" * 70)

        metrics = pipeline.evaluate_clustering(cleaned_texts, topics)

        metrics_path = os.path.join(config.OUTPUT_DIR, "clustering_metrics.csv")
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        metrics = {}

    # Save results
    try:
        logger.info("")
        logger.info("STEP 4/4: SAVING RESULTS")
        logger.info("-" * 70)

        output_path = f"{config.OUTPUT_DIR}/{config.OUTPUT_CSV}"
        df_results = pipeline.save_results(
            df_unlisted, valid_indices, topics, output_path
        )

        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}", exc_info=True)
        return 1

    # Print report
    print_discovery_report(pipeline, topics, probs, cleaned_texts, logger)

    # Cleanup
    pipeline.cleanup()

    logger.info("")
    logger.info("=" * 70)
    logger.info("INTENT DISCOVERY COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Results: {output_path}")
    logger.info(f"Discovered {len(set(topics)) - 1} new intent clusters")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover new intents from UNLISTED sentences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (discovers intents from predictions file)
  python scripts/discover_intents.py --input data/input_predictions.csv
  
  # With custom config
  python scripts/discover_intents.py -i data/predictions.csv -c configs/my_discovery_config.yaml
  
  # Debug mode
  python scripts/discover_intents.py -i data/predictions.csv --log-level DEBUG
  
  # Save to custom output directory
  python scripts/discover_intents.py -i data/predictions.csv -o results/discovery
        """,
    )

    # Required: input file
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to CSV with UNLISTED sentences (typically output from predict.py)",
    )

    # Optional: config
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/discovery_config.yaml",
        help="Path to discovery configuration YAML (default: configs/discovery_config.yaml)",
    )

    # Optional: output directory
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for results (overrides config)",
    )

    # Optional: text column name
    parser.add_argument(
        "--text-column",
        type=str,
        default="Phrase",
        help="Name of text column in input CSV (default: Phrase)",
    )

    # Logging options
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
        help="Path to log file (optional)",
    )

    parser.add_argument(
        "--enable-llm-labeling",
        action="store_true",
        help="Enable Local LLM for topic labeling (requires GPU/VRAM). Default: False (KeyBERT only).",
    )

    args = parser.parse_args()

    # Run pipeline
    exit_code = run_discovery_pipeline(
        config_path=args.config,
        input_file=args.input,
        output_dir=args.output,
        text_column=args.text_column,
        log_level=args.log_level,
        log_file=args.log_file,
        enable_llm_labeling=args.enable_llm_labeling,
    )

    sys.exit(exit_code)
