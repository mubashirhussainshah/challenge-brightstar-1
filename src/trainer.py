import torch
import torch.nn.functional as F
from transformers import Trainer
import logging
from typing import Optional


class CustomTrainer(Trainer):
    """
    Trainer with:
    - Weighted CrossEntropy
    - Label smoothing
    """

    def __init__(
        self,
        loss_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights
        self.label_smoothing = label_smoothing

        # Log configuration
        loss_type = "Weighted" if loss_weights is not None else "Standard"
        logging.info(f"Using {loss_type} CrossEntropy (smoothing={label_smoothing})")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the weighted cross-entropy loss with label smoothing.
        """
        labels = inputs.get("labels")

        # Standard forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Ensure weights are on the correct device (GPU/CPU)
        if self.loss_weights is not None and self.loss_weights.device != logits.device:
            self.loss_weights = self.loss_weights.to(logits.device)

        # define loss function
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.loss_weights, label_smoothing=self.label_smoothing
        )

        # Calculate loss
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
