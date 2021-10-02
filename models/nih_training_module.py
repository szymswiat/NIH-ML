from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import List, Any

import numpy as np
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
from clearml import Logger, Task
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch import nn
from torchmetrics import AUROC
from torchmetrics.utilities.data import dim_zero_cat

from losses.focal_loss import FocalLoss
from optimizers.over9000 import RangerLars
from utils.pred_zarr_io import PredZarrWriter


class NIHTrainingModule(pl.LightningModule):

    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)

        assert self.hparams.net_type in self._VARIANTS

        self._classes = OmegaConf.to_object(self.hparams.dynamic.classes)
        self._num_classes = len(self._classes)

        self.last_activation = nn.Sigmoid()

        class_weights = self._compute_class_weights(*self.hparams.dynamic.class_freq)

        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        self.criterion = FocalLoss(class_weights=class_weights,
                                   gamma=self.hparams.focal_loss.gamma,
                                   reduction='mean')

        self.val_auroc = AUROC(num_classes=self._num_classes, compute_on_step=False)
        self.test_auroc = AUROC(num_classes=self._num_classes, compute_on_step=False)

    @abstractmethod
    def forward_derived(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def cml_logger(self) -> Logger:
        return self.logger.experiment

    @property
    def cml_task(self) -> Task:
        return self.logger.task

    def forward(self, x):
        return self.last_activation(self.forward_derived(x))

    def training_step(self, batch, batch_idx):
        if self.trainer.is_global_zero:
            optimizer = self.optimizers()
            for i, group in enumerate(optimizer.param_groups):
                self.log(f'lr/param_group_{i}', group['lr'], on_step=True, on_epoch=False,
                         logger=True, prog_bar=True, rank_zero_only=True)

        x, y_true = batch
        y_pred = self.forward_derived(x)

        loss = self.criterion(y_pred, y_true)

        self.log('loss/train', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward_derived(x)

        loss = self.criterion(y_pred, y_true)

        self.log('loss/val', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.val_auroc(self.last_activation(y_pred), y_true.to(dtype=torch.int))

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.val_auroc.sync()
        preds = dim_zero_cat(self.val_auroc.preds).detach().cpu().numpy()
        targets = dim_zero_cat(self.val_auroc.target).detach().cpu().numpy()
        self.val_auroc.unsync()

        self.val_auroc.reset()

        auroc_val = roc_auc_score(targets, preds)

        if self.trainer.is_global_zero:
            self.cml_logger.report_text(msg=f'Val samples count: {len(targets)}.')

            fig = self._create_auroc_fig(preds, targets)
            self.cml_logger.report_plotly(title='roc_plots',
                                          series='val',
                                          figure=fig,
                                          iteration=self.trainer.current_epoch)
            self.cml_logger.report_scalar(title='auroc_avg',
                                          series='val',
                                          value=auroc_val,
                                          iteration=self.trainer.current_epoch)

        self.log('auroc_avg/val', value=auroc_val, on_epoch=True, on_step=False,
                 logger=False, prog_bar=False)

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward_derived(x)

        loss = self.criterion(y_pred, y_true)

        self.log('loss/test', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.test_auroc(self.last_activation(y_pred), y_true.to(dtype=torch.int))

        return loss

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.test_auroc.sync()
        preds = dim_zero_cat(self.test_auroc.preds).detach().cpu().numpy()
        targets = dim_zero_cat(self.test_auroc.target).detach().cpu().numpy()
        self.test_auroc.unsync()

        if self.trainer.is_global_zero:
            self._write_and_upload_epoch_output_h5(preds, targets)

            self.cml_logger.report_text(msg=f'\nTest samples count: {len(targets)}.')

            fig = self._create_auroc_fig(preds, targets)

            self.cml_logger.report_plotly(title='roc_plots',
                                          series='test',
                                          figure=fig,
                                          iteration=self.trainer.current_epoch)

            self.cml_logger.report_scalar(title='auroc_avg',
                                          series='test',
                                          value=roc_auc_score(targets, preds),
                                          iteration=self.trainer.current_epoch)
            self.cml_logger.flush()

    def configure_optimizers(self):
        opt: DictConfig = self.hparams.optimizer

        common_params = dict(
            params=self.parameters(),
            lr=opt.get('lr', 0),
            weight_decay=opt.get('weight_decay', 0)
        )
        if opt.type == 'rmsprop':
            optimizer = torch.optim.RMSprop(**common_params,
                                            momentum=opt.get('momentum', 0))
        elif opt.type == 'adam':
            optimizer = torch.optim.Adam(**common_params)
        elif opt.type == 'ranger_lars':
            optimizer = RangerLars(**common_params)
        elif opt.type == 'sgd':
            optimizer = torch.optim.SGD(**common_params,
                                        momentum=opt.get('momentum', 0),
                                        nesterov=opt.get('nesterov', False))
        else:
            raise ValueError('Invalid optimizer type in train_config.yaml.')

        return optimizer

    def _write_and_upload_epoch_output_h5(self, preds: np.ndarray, targets: np.ndarray):
        log_dir = Path(self.trainer._default_root_dir)
        log_dir.mkdir(exist_ok=True, parents=True)

        with PredZarrWriter(log_dir / 'test_output.zarr') as pzw:
            pzw.write_pred_output(preds, targets, self._classes)

        self.cml_task.upload_artifact(name='test_prediction_output',
                                      artifact_object=log_dir / 'test_output.zarr')

    def _create_auroc_fig(self, preds: np.ndarray, targets: np.ndarray) -> go.Figure:
        fig = go.Figure()

        fig.add_shape(type='line', line=dict(dash='dash'),
                      x0=0, x1=1,
                      y0=0, y1=1)

        roc_output = []
        for i in range(targets.shape[1]):
            roc_output.append(roc_curve(targets[..., i], preds[..., i]))

        for i, ((fpr, tpr, thresholds), cls) in enumerate(zip(roc_output, self._classes)):
            cls = cls.replace('_', ' ')
            thresholds = [f'threshold: {th:.3f}' for th in thresholds]
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, text=thresholds,
                           name=f'{cls:20} AUC: {auc(fpr, tpr):.3f}', mode='lines'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=800, height=800,
            font=dict(family='Courier New', size=10),
            legend=dict(
                xanchor='right',
                yanchor='bottom',
                x=0.928, y=0.01,
                traceorder='normal',
                font=dict(size=9)
            )
        )

        return fig

    @staticmethod
    def _compute_class_weights(samples_per_class: List[float], samples_count: int) -> torch.Tensor:
        class_freq = torch.tensor(samples_per_class, dtype=torch.float)

        class_weights = samples_count / (len(class_freq) * class_freq)

        return torch.sqrt(class_weights / torch.max(class_weights))

    @classmethod
    def load_from_file(cls, path: Path) -> NIHTrainingModule:
        state = torch.load(path.as_posix())

        hparams = state['hparams']
        state_dict = state['state_dict']

        model = cls(OmegaConf.create(hparams))
        model.load_state_dict(state_dict)

        return model

    def save_to_file(self, path: Path):
        state = {
            'hparams': OmegaConf.to_object(self.hparams),
            'state_dict': self.state_dict()
        }
        torch.save(state, path)
