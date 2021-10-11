from abc import abstractmethod
from pathlib import Path
from typing import List, Any, Dict

import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from torch import nn
from torchmetrics import AUROC
from torchmetrics.utilities.data import dim_zero_cat

from losses.focal_loss import FocalLoss
from metrics import plot_gen
from metrics.helpers import precision_recall_auc_scores
from modules.clearml_module import ClearMLModule
from optimizers.over9000 import RangerLars
from utils.pred_zarr_io import PredZarrWriter


class NIHClassificationModule(ClearMLModule):

    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        assert self.hparams.net_type in self._VARIANTS

        self._classes = OmegaConf.to_object(self.hparams.dynamic.classes)
        self._num_classes = len(self._classes)

        self.last_activation = nn.Sigmoid()

        class_weights = self._compute_class_weights(*self.hparams.dynamic.class_freq)

        self.criterion = self.get_loss(class_weights)

        self.val_auroc = AUROC(num_classes=self._num_classes, compute_on_step=False)
        self.test_auroc = AUROC(num_classes=self._num_classes, compute_on_step=False)

    @abstractmethod
    def forward_derived(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, x):
        if self.training:
            self.eval()
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
        targets = dim_zero_cat(self.val_auroc.target).detach().cpu().numpy()
        preds = dim_zero_cat(self.val_auroc.preds).detach().cpu().numpy()
        self.val_auroc.unsync()

        self.val_auroc.reset()

        self._report_epoch_metrics(targets, preds, 'val')

        auroc_val = roc_auc_score(targets, preds)

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
        targets = dim_zero_cat(self.test_auroc.target).detach().cpu().numpy()
        preds = dim_zero_cat(self.test_auroc.preds).detach().cpu().numpy()
        self.test_auroc.unsync()

        self._report_epoch_metrics(targets, preds, 'test')

    def configure_optimizers(self):
        opt: DictConfig = self.hparams.optimizer

        common_params = dict(
            params=self.parameters(),
            lr=opt.get('lr_initial', 0),
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

    def get_loss(self, class_weights: torch.Tensor) -> nn.Module:
        loss_cfg: DictConfig = self.hparams.loss
        if loss_cfg.type == 'bce':
            return nn.BCEWithLogitsLoss(pos_weight=class_weights)
        elif loss_cfg.type == 'focal':
            return FocalLoss(class_weights=class_weights,
                             gamma=loss_cfg.get('gamma', 2),
                             reduction='mean')
        elif loss_cfg.type == 'ml_soft_margin':
            return nn.MultiLabelSoftMarginLoss(weight=class_weights)
        else:
            raise ValueError('Invalid loss type in train_config.yaml.')

    def _report_epoch_metrics(self, targets: np.ndarray, preds: np.ndarray, epoch_type: str):
        if self.trainer.is_global_zero:
            self._write_and_upload_epoch_output_to_zarr(targets, preds)

            self.cml_logger.report_text(msg=f'\n{epoch_type.capitalize()} samples count: {len(targets)}.')

            for name, fig in self._create_metric_figures(targets, preds).items():
                self.cml_logger.report_plotly(title=f'{name}_{epoch_type}',
                                              series=name,
                                              figure=fig,
                                              iteration=self.trainer.current_epoch)

            self.cml_logger.report_scalar(title='auroc_avg',
                                          series=epoch_type,
                                          value=roc_auc_score(targets, preds),
                                          iteration=self.trainer.current_epoch)
            self.cml_logger.report_scalar(title='aupr_avg',
                                          series=epoch_type,
                                          value=precision_recall_auc_scores(targets, preds).mean(),
                                          iteration=self.trainer.current_epoch)

            self.cml_logger.flush()

    def _write_and_upload_epoch_output_to_zarr(self, targets: np.ndarray, preds: np.ndarray):
        log_dir = Path(self.trainer._default_root_dir)
        log_dir.mkdir(exist_ok=True, parents=True)

        with PredZarrWriter(log_dir / 'test_output.zarr') as pzw:
            pzw.write_pred_output(targets, preds, self._classes)

        self.cml_task.upload_artifact(name='test_prediction_output',
                                      artifact_object=log_dir / 'test_output.zarr')

    def _create_metric_figures(self, targets: np.ndarray, preds: np.ndarray) -> Dict[str, go.Figure]:
        roc_curves = []
        for i in range(targets.shape[1]):
            roc_curves.append(roc_curve(targets[..., i], preds[..., i]))

        roc_fig = plot_gen.create_fig(roc_curves, self._classes, ('FPR', 'TPR'), 'AUC', auc, line_mode=0)

        pr_curves = []
        for i in range(targets.shape[1]):
            precision, recall, thresholds = precision_recall_curve(targets[..., i], preds[..., i])
            pr_curves.append((recall, precision, thresholds))

        pr_fig = plot_gen.create_fig(pr_curves, self._classes, ('Recall', 'Precision'), 'AUC', auc, line_mode=1)

        return {
            'roc': roc_fig,
            'pr': pr_fig
        }

    @staticmethod
    def _compute_class_weights(samples_per_class: List[float], samples_count: int) -> torch.Tensor:
        class_freq = torch.tensor(samples_per_class, dtype=torch.float)

        class_weights = samples_count / (len(class_freq) * class_freq)

        return torch.sqrt(class_weights / torch.max(class_weights))
