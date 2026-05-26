from torch import nn
from torch.nn import functional as F
import torch
from .base import BaseModule


class StyleContrastiveEnhanceCLIP(BaseModule):
    def __init__(self, loss_weight, clip_params) -> None:
        super().__init__(loss_weight)
        if clip_params is None:
            clip_params = {}
        self.clip_objective = CLIP_loss(**clip_params)

    def forward(self, target_feats, style_feats):
        clip_loss = self.clip_objective(style_feats, target_feats.detach()) + self.clip_objective(
            target_feats, style_feats.detach()
        )
        clip_loss /= 2
        return self._ouput_loss(clip_loss)


class CLIP_loss(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, reconstruct_features, target_features):
        """
        target_features shape: BxD
        reconstruct_features shape BxD
        """
        target_features = F.normalize(target_features, p=2, dim=-1)
        reconstruct_features = F.normalize(reconstruct_features, p=2, dim=-1)

        logits = target_features @ reconstruct_features.t()
        loss = self.contrastive_loss(logits)

        return loss

    def contrastive_loss(self, logits):
        targets = torch.arange(logits.size(0)).to(logits.device)
        loss = F.cross_entropy(logits, targets, reduction=self.reduction)

        return loss
