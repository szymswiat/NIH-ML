from __future__ import annotations

from clearml import Logger, Task
from omegaconf import DictConfig
from pytorch_lightning import LightningModule


class CommonTrainingModule(LightningModule):

    def __init__(self, hparams: DictConfig):
        super().__init__()

        self.save_hyperparameters(hparams)

    @property
    def cml_logger(self) -> Logger:
        return self.logger.experiment

    @property
    def cml_task(self) -> Task:
        return self.logger.task