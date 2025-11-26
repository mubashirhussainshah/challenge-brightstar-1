import torch
import torch.nn.functional as F
from transformers import Trainer
from .loss import FocalLoss
import logging
from typing import Optional


class FocalTrainer(Trainer):
    """
    Enhanced trainer with:
    - Focal loss
    - Priority weighting
    - Label smoothing
    """

    def __init__(
        self,
        loss_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        use_focal_loss: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_weights = loss_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss

        # Initialize loss functions
        if self.use_focal_loss:
            self.loss_fn = FocalLoss(
                alpha=loss_weights,
                gamma=gamma,
                reduction="mean",
                label_smoothing=label_smoothing,
            )
            logging.info(f"Using Focal Loss (γ={gamma}, smoothing={label_smoothing})")
        else:
            self.loss_fn = None
            logging.info(f"Using Weighted CrossEntropy")

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")

        # Standard forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute base loss
        if self.use_focal_loss and self.loss_fn is not None:
            loss = self.loss_fn(logits, labels)
        else:
            if self.loss_weights is not None:
                if self.loss_weights.device != logits.device:
                    self.loss_weights = self.loss_weights.to(logits.device)
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=self.loss_weights, label_smoothing=self.label_smoothing
                )
            else:
                loss_fct = torch.nn.CrossEntropyLoss(
                    label_smoothing=self.label_smoothing
                )
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )

        return (loss, outputs) if return_outputs else loss


class RDropTrainer(FocalTrainer):
    """
    Trainer with R-Drop regularization (Simultaneously minimizes CE and KL-Divergence).
    Paper: https://arxiv.org/abs/2106.14448
    """

    def __init__(self, rdrop_alpha: float = 4.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = rdrop_alpha

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            # During evaluation/inference, we don't need R-Drop
            return super().compute_loss(model, inputs, return_outputs)

        labels = inputs.get("labels")

        # Forward pass 1
        outputs1 = model(**inputs)
        logits1 = outputs1.get("logits")

        # Forward pass 2
        outputs2 = model(**inputs)
        logits2 = outputs2.get("logits")

        # We use the internal loss_fn (Focal) if available, otherwise standard CE
        if self.use_focal_loss and self.loss_fn is not None:
            loss1 = self.loss_fn(logits1, labels)
            loss2 = self.loss_fn(logits2, labels)
            ce_loss = (loss1 + loss2) / 2
        else:
            # Reconstruct standard loss function if Focal is off
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.loss_weights, label_smoothing=self.label_smoothing
            )
            ce_loss = (loss_fct(logits1, labels) + loss_fct(logits2, labels)) / 2

        # KL Divergence (Consistency loss)
        # We want the predictions of Pass 1 to be close to Pass 2, and vice versa.
        # No .detach() ensures gradients flow through both to align them.
        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)

        # kl_div(input, target) -> divergence of input FROM target
        kl_loss = (
            F.kl_div(p1, p2, reduction="batchmean", log_target=True)
            + F.kl_div(p2, p1, reduction="batchmean", log_target=True)
        ) / 2

        # Final loss
        loss = ce_loss + (self.rdrop_alpha * kl_loss)

        return loss
