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
from typing import Optional


def _make_mask(labels: torch.Tensor, label_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Return label_mask; if None, infer it from NaN positions in labels."""
    if label_mask is not None:
        return label_mask.float()
    return (~torch.isnan(labels)).float()


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Binary cross entropy loss for partially labeled multilabel data.

    Expected shapes:
        logits:     (batch_size, num_labels)
        labels:     (batch_size, num_labels)  — may contain NaN
        label_mask: (batch_size, num_labels)  — optional; inferred from NaN if None

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
        label_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = _make_mask(labels, label_mask)
        safe_labels = torch.nan_to_num(labels, nan=0.0).float()

        loss = F.binary_cross_entropy_with_logits(
            logits,
            safe_labels,
            reduction="none",
        )

        masked_loss = loss * mask

        denominator = mask.sum().clamp(min=1.0)
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
            Scalar float: same weight for all labels.
            1-D tensor of shape (num_labels,): per-label weights.
            If None, no alpha weighting is applied.
        gamma:
            Focusing parameter. Common value: 2.0.
    """

    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = _make_mask(labels, label_mask)
        safe_labels = torch.nan_to_num(labels, nan=0.0).float()

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            safe_labels,
            reduction="none",
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(safe_labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt).pow(self.gamma)

        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(loss.device)

            if alpha.ndim == 0:
                # scalar alpha: same weight for all labels
                alpha_weight = torch.where(safe_labels == 1, alpha, 1 - alpha)
            else:
                # per-label alpha vector: shape (num_labels,) → broadcast over batch
                alpha = alpha.unsqueeze(0)  # shape: (1, num_labels)
                alpha_weight = torch.where(safe_labels == 1, alpha, 1 - alpha)

            loss = alpha_weight * loss

        masked_loss = loss * mask

        denominator = mask.sum().clamp(min=1.0)
        return masked_loss.sum() / denominator
