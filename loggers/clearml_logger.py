from typing import Union, Dict, Optional, Any

from clearml import Task, Logger
from omegaconf import OmegaConf
from pytorch_lightning.loggers import LightningLoggerBase


class ClearMLLogger(LightningLoggerBase):

    def __init__(
            self,
            task: Task,
            log_hyperparams: bool = True
    ):
        super().__init__()

        self.task = task
        self._log_hyperparams = log_hyperparams

    @property
    def experiment(self) -> Logger:
        return self.task.logger

    @property
    def name(self) -> str:
        return self.task.name

    @property
    def version(self) -> Union[int, str]:
        return self.task.task_id

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for metric_name, metric_val in metrics.items():
            name_parts = metric_name.split('/')
            self.experiment.report_scalar(title=name_parts[0],
                                          series='.'.join(name_parts[1:]),
                                          value=metric_val,
                                          iteration=step)

    def log_hyperparams(self, params: Union[Dict[str, Any]], *args, **kwargs):
        if self._log_hyperparams:
            params = self._convert_params(params)
            params = OmegaConf.create(params)

            self.task.set_parameters_as_dict(OmegaConf.to_object(params))
