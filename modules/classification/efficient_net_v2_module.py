from __future__ import annotations

import torch
from omegaconf import DictConfig
from timm.models.efficientnet import tf_efficientnetv2_s, tf_efficientnetv2_m, tf_efficientnetv2_l
from torch.nn import Sigmoid

from modules.base_modules import LoadableModule


class EfficientNetV2Module(LoadableModule):
    _VARIANTS = {
        's': tf_efficientnetv2_s,
        'm': tf_efficientnetv2_m,
        'l': tf_efficientnetv2_l
    }

    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)
        assert self.hparams.net_type in self._VARIANTS

        num_classes = len(self.hparams.dynamic.classes)

        self.model = self._VARIANTS[hparams.net_type](drop_path_rate=self.hparams.drop_path_rate,
                                                      num_classes=num_classes,
                                                      pretrained=self.hparams.pretrained)
        self.last_act = Sigmoid()

        # update batch norm momentum to 0.99 (1 - 0.99 = 0.01)
        for name, child in self.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                child.momentum = 0.01

    def forward(self, x: torch.Tensor, apply_act=True) -> torch.Tensor:
        outputs = self.model(x)
        if apply_act:
            outputs = self.last_act(outputs)

        return outputs

    def set_phase(self, phase: DictConfig):
        self.model.drop_rate = phase.dropout_rate
