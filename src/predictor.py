import torch
import numpy as np
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Optional


class IntentPredictor:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[Classifier] Initializing Engine on {self.device.upper()}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.labels = self.model.config.id2label
        except Exception as e:
            raise ValueError(
                f"CRITICAL: Could not load model from '{model_path}'.\nError: {e}"
            )

    def predict(
        self,
        texts: List[str],
        batch_size: int = 32,
        temperature: float = 1.0,
        threshold: float = 0.7,
    ):
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(
                    outputs.logits / temperature, dim=-1
                )
                confidences, pred_ids = torch.max(probs, dim=-1)

            confidences = confidences.cpu().numpy()
            pred_ids = pred_ids.cpu().numpy()

            for _, conf, pid in zip(batch, confidences, pred_ids):
                label = self.labels[pid]
                if label == "NOMATCH":
                    final_label = "NOMATCH"
                    status = "NO_MATCH"
                elif conf < threshold:
                    final_label = "UNLISTED"
                    status = "UNLISTED"
                else:
                    final_label = label
                    status = "VALID"
                results.append((final_label, status, conf))

        return results
