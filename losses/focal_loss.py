import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(
            self,
            class_weights: torch.Tensor,
            gamma: int = 0,
            reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        self.class_weights = class_weights

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        batch_size, num_classes = targets.shape

        p = torch.sigmoid(preds)
        p = torch.where(targets >= 0.5, p, 1 - p)
        logp = -torch.log(torch.clamp(p, 1e-6, 1 - 1e-6))
        loss = logp * ((1 - p) ** self.gamma)

        class_weights = self.class_weights.to(loss.device)
        if self.reduction == 'mean':
            return (loss.mean(dim=0) * class_weights).mean()
        elif self.reduction == 'sum':
            return (loss.sum(dim=0) * class_weights) / batch_size
        else:
            raise ValueError()
