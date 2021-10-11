from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from clearml import Logger, Task
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningModule


class ClearMLModule(LightningModule):

    def __init__(self, hparams: DictConfig):
        super().__init__()

        self.save_hyperparameters(hparams)

    @property
    def cml_logger(self) -> Logger:
        return self.logger.experiment

    @property
    def cml_task(self) -> Task:
        return self.logger.task

    @classmethod
    def load_state(cls, state: Dict):
        hparams = state['hparams']
        state_dict = state['state_dict']

        model = cls(OmegaConf.create(hparams))
        model.load_state_dict(state_dict)

        return model

    @classmethod
    def load_state_from_file(cls, path: Path):
        state = torch.load(path.as_posix())

        return cls.load_state(state)

    def save_state_to_file(self, path: Path, **kwargs):
        state = {
            'hparams': OmegaConf.to_object(self.hparams),
            'state_dict': self.state_dict(),
            **kwargs
        }
        torch.save(state, path)
