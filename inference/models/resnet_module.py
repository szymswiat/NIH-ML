import torch
from omegaconf import DictConfig
from timm.models.resnet import resnet18, resnet34, resnet50, resnet101
from torch.nn import Sigmoid

from inference.models.base.loadable_module import LoadableModule


class ResNetModule(LoadableModule):
    _VARIANTS = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101
    }

    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)
        assert self.hparams.net_type in self._VARIANTS

        num_classes = len(self.hparams.dynamic.classes)

        self.model = self._VARIANTS[hparams.net_type](num_classes=num_classes,
                                                      pretrained=hparams.pretrained)

        self.last_act = Sigmoid()

    def forward(self, x: torch.Tensor, apply_act=True) -> torch.Tensor:
        outputs = self.model(x)
        if apply_act:
            outputs = self.last_act(outputs)

        return outputs
