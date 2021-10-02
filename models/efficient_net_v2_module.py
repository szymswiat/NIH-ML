from __future__ import annotations

import torch
from omegaconf import DictConfig
from timm.models.efficientnet import tf_efficientnetv2_s, tf_efficientnetv2_m, tf_efficientnetv2_l

from models.nih_training_module import NIHTrainingModule


class EfficientNetV2Module(NIHTrainingModule):
    _VARIANTS = {
        's': tf_efficientnetv2_s,
        'm': tf_efficientnetv2_m,
        'l': tf_efficientnetv2_l
    }

    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        self.model = self._VARIANTS[self.hparams.net_type](drop_path_rate=self.hparams.drop_path_rate,
                                                           num_classes=self._num_classes,
                                                           pretrained=self.hparams.pretrained)

        # update batch norm momentum to 0.99 (1 - 0.99 = 0.01)
        for name, child in self.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                child.momentum = 0.01

    def on_train_epoch_start(self) -> None:
        for phase in reversed(self.hparams.phases):
            if self.current_epoch >= phase.epoch_milestone:
                self._set_phase(phase)
                break

    def _set_phase(self, phase: DictConfig):
        self.model.drop_rate = phase.dropout_rate

    def forward_derived(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
