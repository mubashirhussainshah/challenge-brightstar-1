import torch
import torch.nn.functional as F
from typing import Optional


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for multi-class classification with priority weighting.

    Features:
    - Focuses on hard examples (reduces loss for easy examples)
    - Supports class weighting (alpha)
    - Optional label smoothing for regularization

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: True labels [batch_size]
        """
        # Compute cross-entropy with label smoothing
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=self.label_smoothing
        )

        # Get probabilities
        p = torch.nn.functional.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
