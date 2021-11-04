from typing import Any, List, Tuple, Dict

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torchmetrics import MAP

from training.common_training_module import CommonTrainingModule
from optimizers.over9000 import RangerLars


class NIHDetectionTrainingModule(CommonTrainingModule):

    def __init__(
            self,
            hparams: DictConfig,
            model: LightningModule
    ):
        super().__init__(hparams)

        self.classes = OmegaConf.to_object(hparams.dynamic.classes)

        self.model = model

        self.val_map = MAP(class_metrics=False, compute_on_step=False)
        self.test_map = MAP(class_metrics=False, compute_on_step=False)

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
        for loss_name, value in loss_dict.items():
            self.log(f'loss/{loss_name}_train', value, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        loss_dict['total'] = sum(loss for loss in loss_dict.values())
        self.log(f'loss/total_train', loss_dict['total'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss_dict['total']

    def validation_step(self, batch, batch_idx):
        images, targets = self.target_to_frcnn_input(batch)

        preds = self.model(images)

        self.val_map(preds, targets)

    def validation_epoch_end(self, outputs):
        map_metric = self.val_map.compute()

        self.log(f'map/total_avg_val', map_metric['map'],
                 prog_bar=False, on_step=False,
                 on_epoch=True, sync_dist=True, logger=False)

        self._report_epoch_metrics(map_metric, 'val')

    def test_step(self, batch, batch_idx: int):
        images, targets = self.target_to_frcnn_input(batch)

        preds = self.model(images)
        self.test_map.update(preds, targets)

    def test_epoch_end(self, outputs):
        map_metric = self.test_map.compute()

        self._report_epoch_metrics(map_metric, 'test')

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
                labels = torch.empty((0,)).long().to(self.device)
                boxes = torch.empty((0, 4)).long().to(self.device)
            targets.append(dict(labels=labels, boxes=boxes))

        batch_image_list = [img.float().to(self.device) for img in batch_image_list]

        return batch_image_list, targets

    def _report_epoch_metrics(self, map_metric: Dict, epoch_type: str):
        # report average metrics across all classes
        for m_name, value in map_metric.items():
            self.cml_logger.report_scalar(title=f'map_{epoch_type}',
                                          series=m_name,
                                          value=value,
                                          iteration=self.trainer.current_epoch)

        self.cml_logger.flush()
