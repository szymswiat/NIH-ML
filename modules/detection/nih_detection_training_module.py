from typing import Any, List, Tuple, Dict

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule

from metrics.detection_metric import DetectionMetric
from modules.base_modules import CommonTrainingModule
from optimizers.over9000 import RangerLars


class NIHDetectionTrainingModule(CommonTrainingModule):

    LOG_METRICS_CLASS = ['f_score', 'AP', 'total TP', 'total FP']

    def __init__(
            self,
            hparams: DictConfig,
            model: LightningModule
    ):
        super().__init__(hparams)

        self.classes = OmegaConf.to_object(hparams.dynamic.classes)

        self.detection_metric_val = DetectionMetric(self.classes)
        self.detection_metric_test = DetectionMetric(self.classes)

        self.model = model

    def forward(self, batch) -> Any:
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        if self.trainer.is_global_zero:
            optimizer = self.optimizers()
            for i, group in enumerate(optimizer.param_groups):
                self.log(f'lr/param_group_{i}', group['lr'], on_step=True, on_epoch=False,
                         logger=True, prog_bar=True, rank_zero_only=True)

        images, targets = self.target_to_frcnn_input(batch)

        loss_dict = self.model(images, targets)
        loss_dict['total'] = sum(loss for loss in loss_dict.values())
        # for loss_name, value in loss_dict.items():
        #     self.log(f'loss/{loss_name}_train', value, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss_dict['total']

    def validation_step(self, batch, batch_idx):
        images, targets = self.target_to_frcnn_input(batch)

        outputs = self.model(images)

        self.detection_metric_val(targets, outputs)

    def validation_epoch_end(self, outputs):
        metrics_all, metrics_per_class = self.detection_metric_val.compute()

        m_ap_avg = metrics_all["total_map"]

        if self.trainer.is_global_zero:
            self._report_epoch_metrics(metrics_all, metrics_per_class, 'val')
        self.log(f'm_ap/all_val', m_ap_avg, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        images, targets = self.target_to_frcnn_input(batch)

        preds = self.model(images)

        self.detection_metric_test(targets, preds)

    def test_epoch_end(self, outputs):
        metrics_all, metrics_per_class = self.detection_metric_test.compute()

        if self.trainer.is_global_zero:
            self._report_epoch_metrics(metrics_all, metrics_per_class, 'test')

    def configure_optimizers(self):
        opt: DictConfig = self.hparams.optimizer

        common_params = dict(
            params=self.parameters(),
            lr=opt.get('lr_initial', 0),
            weight_decay=opt.get('weight_decay', 0)
        )

        if opt.type == 'ranger_lars':
            optimizer = RangerLars(**common_params)
        else:
            raise ValueError()

        return optimizer

    def target_to_frcnn_input(self, batch: List[Tuple[List, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
        batch_image_list, target_list = zip(*batch)

        targets = []
        for target in target_list:
            target = torch.tensor(target).long().to(self.device)
            if len(target) > 0:
                labels = target[..., -1]
                boxes = target[..., :-1]
            else:
                labels = torch.empty((0,)).long()
                boxes = torch.empty((0, 4)).long().to(self.device)
            targets.append(dict(labels=labels, boxes=boxes))

        batch_image_list = [img.float().to(self.device) for img in batch_image_list]

        return batch_image_list, targets

    def _report_epoch_metrics(self, metrics_all: Dict, metrics_per_class: Dict, epoch_type: str):
        # report average metrics across all classes
        for m_name, value in metrics_all.items():
            self.cml_logger.report_scalar(title=m_name,
                                          series=f'average_{epoch_type}',
                                          value=value,
                                          iteration=self.trainer.current_epoch)
        # report metrics for subsequent classes
        for mc in metrics_per_class:
            cls_name = mc.pop('class')
            for m_name in self.LOG_METRICS_CLASS:
                value = mc[m_name]
                self.cml_logger.report_scalar(title=m_name,
                                              series=f'{cls_name}_{epoch_type}',
                                              value=value,
                                              iteration=self.trainer.current_epoch)

        self.cml_logger.flush()
