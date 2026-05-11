"""
src/training/losses.py
----------------------
Loss functions for partially labeled multilabel classification.

Main idea:
    Labels may contain NaN in the original dataset.
    During training, NaN labels are replaced by 0 in the label tensor,
    but a separate label_mask indicates which labels are actually annotated.

    loss is computed only where label_mask == 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Binary cross entropy loss for partially labeled multilabel data.

    Expected shapes:
        logits:     (batch_size, num_labels)
        labels:     (batch_size, num_labels)
        label_mask: (batch_size, num_labels)

    label_mask:
        1 = label is annotated, include in loss
        0 = label is NaN / missing, ignore in loss
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.float()
        label_mask = label_mask.float()

        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )

        masked_loss = loss * label_mask

        denominator = label_mask.sum().clamp(min=1.0)
        return masked_loss.sum() / denominator


class MaskedFocalLoss(nn.Module):
    """
    Optional focal loss for imbalanced multilabel classification.

    This is useful when positive labels are very rare,
    e.g. the threat label.

    Formula:
        BCE loss is weighted by (1 - pt) ** gamma

    Args:
        alpha:
            Optional positive-class weight.
            If None, no alpha weighting is applied.
        gamma:
            Focusing parameter. Common value: 2.0.
    """

    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.float()
        label_mask = label_mask.float()

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt).pow(self.gamma)

        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha = torch.as_tensor(
                self.alpha,
                dtype=loss.dtype,
                device=loss.device,
            )

            if alpha.ndim == 0:
                alpha_weight = torch.where(labels == 1, alpha, 1 - alpha)
            else:
                alpha_weight = torch.where(labels == 1, alpha, 1 - alpha)

            loss = alpha_weight * loss

        masked_loss = loss * label_mask

        denominator = label_mask.sum().clamp(min=1.0)
        return masked_loss.sum() / denominator