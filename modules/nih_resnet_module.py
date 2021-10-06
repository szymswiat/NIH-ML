import torch
from omegaconf import DictConfig
from timm.models.resnet import resnet18, resnet34, resnet50, resnet101

from modules.nih_classification_module import NIHClassificationModule


class NIHResNetModule(NIHClassificationModule):
    _VARIANTS = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101
    }

    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        self.model = self._VARIANTS[self.hparams.net_type](num_classes=self._num_classes,
                                                           pretrained=self.hparams.pretrained)

    def forward_derived(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
