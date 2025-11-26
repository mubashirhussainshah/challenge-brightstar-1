import torch
import numpy as np
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Optional


class InferencePipeline:
    """
    Optimized inference class that loads the model once and stays in memory.
    """

    def __init__(self, model_path: str, device: str = ""):
        self.model_path = model_path
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logging.info(f"Loading inference model from {model_path} to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()
        logging.info("Model loaded successfully.")

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        temperature: float = 1.0,
        threshold: float = 0.7,
        return_confidences: bool = False,
    ) -> Tuple[List[str], List[int], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Batch prediction optimized for test set evaluation.

        Args:
            texts: List of input texts
            batch_size: Batch size for inference
            return_confidences: If True, also return confidence scores

        Returns:
            predicted_labels: List of predicted intent labels
            all_preds_ids: List of predicted class IDs
            confidences: (Optional) Array of confidence scores
        """
        all_preds_ids = []
        all_confidences = []
        all_probs = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Inference Batch"):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                scaled_logits = outputs.logits / temperature
                probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                confidences, predictions = torch.max(probs, dim=-1)

                all_preds_ids.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        predicted_labels = []
        for idx, conf in zip(all_preds_ids, all_confidences):
            if conf < threshold:
                predicted_labels.append("NOMATCH")
            else:
                predicted_labels.append(self.model.config.id2label[idx])

        if return_confidences:
            return (
                predicted_labels,
                all_preds_ids,
                np.array(all_confidences),
                np.array(all_probs),
            )
        else:
            return (predicted_labels, all_preds_ids)
