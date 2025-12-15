import sys
import logging
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import InferenceConfig
from src.predictor import IntentPredictor


def evaluate(
    config: InferenceConfig,
    df_test: pd.DataFrame,
    model_path: str,
    label_column: str,
    label2id: Dict,
    id2label: Dict,
    num_labels: int,
    text_column: str,
    report_suffix: str = "",
):
    """
    Evaluate using the InferencePipeline class (batch mode).
    """
    logging.info("\n" + "=" * 70)
    logging.info("FINAL EVALUATION ON TEST SET")
    logging.info("=" * 70)

    pipeline = IntentPredictor(model_path=model_path)

    test_texts = df_test[text_column].tolist()
    true_labels = df_test[label_column].tolist()

    logging.info(f"Running inference on {len(test_texts)} samples...")

    # Get predictions with confidences
    results = pipeline.predict(
        test_texts,
        batch_size=config.BATCH_SIZE,
        temperature=config.TEMPERATURE,
        threshold=config.CONFIDENCE_THRESHOLD,
    )

    pred_labels, _, confidences = zip(*results)
    pred_labels = list(pred_labels)
    confidences = np.array(confidences)

    unique_labels = sorted(list(set(true_labels + pred_labels)))

    # Calculate metrics
    test_acc = accuracy_score(true_labels, pred_labels)
    test_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    logging.info(f"\nTest Accuracy: {test_acc:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")

    # Classification report
    target_names = [id2label[i] for i in range(num_labels)]
    logging.info("\n" + "=" * 70)
    logging.info("DETAILED CLASSIFICATION REPORT")
    logging.info("=" * 70)
    logging.info(classification_report(true_labels, pred_labels, zero_division=0))

    # ============================================================
    # CONFIDENCE STATISTICS
    # ============================================================
    logging.info("\n" + "=" * 70)
    logging.info("CONFIDENCE STATISTICS")
    logging.info("=" * 70)
    logging.info(f"Mean Confidence:   {np.mean(confidences):.4f}")
    logging.info(f"Median Confidence: {np.median(confidences):.4f}")
    logging.info(f"Std Confidence:    {np.std(confidences):.4f}")
    logging.info(f"Min Confidence:    {np.min(confidences):.4f}")
    logging.info(f"Max Confidence:    {np.max(confidences):.4f}")

    # Confidence distribution
    logging.info("\nConfidence Distribution:")
    conf_ranges = [
        ("Very High (>0.90)", np.sum(confidences > 0.90)),
        ("High (0.80-0.90)", np.sum((confidences >= 0.80) & (confidences <= 0.90))),
        ("Medium (0.70-0.80)", np.sum((confidences >= 0.70) & (confidences < 0.80))),
        ("Low (0.50-0.70)", np.sum((confidences >= 0.50) & (confidences < 0.70))),
        ("Very Low (<0.50)", np.sum(confidences < 0.50)),
    ]

    for range_name, count in conf_ranges:
        percentage = (count / len(confidences)) * 100
        logging.info(f"  {range_name:25s}: {count:4d} ({percentage:5.1f}%)")

    # Confidence by correctness
    correct_mask = np.array([t == p for t, p in zip(true_labels, pred_labels)])
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]

    logging.info("\nConfidence by Prediction Correctness:")
    logging.info(
        f"  Correct predictions:   Mean={np.mean(correct_confidences):.4f}, "
        f"Median={np.median(correct_confidences):.4f}"
    )
    if len(incorrect_confidences) > 0:
        logging.info(
            f"  Incorrect predictions: Mean={np.mean(incorrect_confidences):.4f}, "
            f"Median={np.median(incorrect_confidences):.4f}"
        )
    else:
        logging.info(f"  Incorrect predictions: None (Perfect accuracy!)")

    logging.info("=" * 70 + "\n")

    # ============================================================
    # SAVE DETAILED RESULTS
    # ============================================================
    results_df = pd.DataFrame(
        {
            "text": test_texts,
            "true_intent": true_labels,
            "predicted_intent": pred_labels,
            "confidence": confidences,
            "correct": correct_mask,
        }
    )

    # Save results
    results_path = f"evaluation_results{report_suffix}.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8")
    logging.info(f"✓ Evaluation results saved to: {results_path}")

    # ============================================================
    # ERROR ANALYSIS
    # ============================================================
    errors = results_df[results_df["correct"] == False].copy()

    if len(errors) > 0:
        logging.info("\n" + "=" * 70)
        logging.info("ERROR ANALYSIS")
        logging.info("=" * 70)
        logging.info(
            f"Total Errors: {len(errors)} ({len(errors)/len(results_df)*100:.2f}%)"
        )

        # Error patterns
        error_patterns = (
            errors.groupby(["true_intent", "predicted_intent"])
            .size()
            .sort_values(ascending=False)
            .head(10)
        )

        logging.info("\nTop 10 Error Patterns:")
        logging.info("-" * 70)
        for (true_int, pred_int), count in error_patterns.items():
            percentage = (count / len(errors)) * 100
            logging.info(
                f"  {true_int:40s} → {pred_int:40s}: {count:3d} ({percentage:4.1f}%)"
            )

        # Low confidence errors
        low_conf_errors = errors[errors["confidence"] < 0.5]
        if len(low_conf_errors) > 0:
            logging.info(f"\nLow Confidence Errors (<0.5): {len(low_conf_errors)}")
            logging.info("  (These are expected errors - model was uncertain)")

        # High confidence errors (more concerning)
        high_conf_errors = errors[errors["confidence"] >= 0.8]
        if len(high_conf_errors) > 0:
            logging.info(f"\n High Confidence Errors (≥0.8): {len(high_conf_errors)}")
            logging.info("  (These are concerning - model was confident but wrong)")
            logging.info("\n  Top 5 High-Confidence Errors:")
            for idx, row in high_conf_errors.nlargest(5, "confidence").iterrows():
                logging.info(f"    Confidence: {row['confidence']:.3f}")
                logging.info(f"    Text: {row['text'][:80]}...")
                logging.info(f"    True: {row['true_intent']}")
                logging.info(f"    Predicted: {row['predicted_intent']}")

        # Save error details
        errors = errors.sort_values("confidence")
        errors_path = f"classification_errors{report_suffix}.csv"
        errors.to_csv(errors_path, index=False, encoding="utf-8")
        logging.info(f"✓ Error details saved to: {errors_path}")

        logging.info("=" * 70 + "\n")
    else:
        logging.info("\n" + "=" * 70)
        logging.info("NO CLASSIFICATION ERRORS - PERFECT ACCURACY!")
        logging.info("=" * 70 + "\n")

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

    plt.figure(figsize=(20, 20))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        cbar_kws={"label": "Count"},
    )
    plt.title(f"Confusion Matrix (Accuracy: {test_acc:.3f}, F1: {test_f1:.3f})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs("training_plots", exist_ok=True)
    cm_path = f"training_plots/confusion_matrix{report_suffix}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Confusion matrix saved to: {cm_path}")

    # ============================================================
    # CONFIDENCE HISTOGRAM
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall confidence distribution
    axes[0].hist(confidences, bins=50, alpha=0.7, color="blue", edgecolor="black")
    axes[0].axvline(
        np.mean(confidences),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(confidences):.3f}",
    )
    axes[0].axvline(
        np.median(confidences),
        color="green",
        linestyle="--",
        label=f"Median: {np.median(confidences):.3f}",
    )
    axes[0].set_xlabel("Confidence Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Overall Confidence Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Confidence by correctness
    if len(incorrect_confidences) > 0:
        axes[1].hist(
            correct_confidences,
            bins=30,
            alpha=0.6,
            color="green",
            label=f"Correct (n={len(correct_confidences)})",
            edgecolor="black",
        )
        axes[1].hist(
            incorrect_confidences,
            bins=30,
            alpha=0.6,
            color="red",
            label=f"Incorrect (n={len(incorrect_confidences)})",
            edgecolor="black",
        )
    else:
        axes[1].hist(
            correct_confidences,
            bins=30,
            alpha=0.7,
            color="green",
            label=f"All Correct (n={len(correct_confidences)})",
            edgecolor="black",
        )

    axes[1].set_xlabel("Confidence Score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Confidence by Prediction Correctness")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    conf_path = f"training_plots/confidence_distribution{report_suffix}.png"
    plt.savefig(conf_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Confidence distribution saved to: {conf_path}")

    return test_f1
