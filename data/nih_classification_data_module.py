import logging
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import ListConfig, DictConfig
from torch.utils.data import DataLoader

from data import nih_const
from data.nih_dataset import NIHDataset

logger = logging.getLogger(__name__)


class NIHClassificationDataModule(pl.LightningDataModule):

    def __init__(
            self,
            dataset_path: str,
            split_type: str,
            phases: ListConfig,
            num_workers: int = 2,
            classes: List[str] = None
    ):
        super().__init__()
        self._dataset_path = Path(dataset_path)
        self._phases = phases
        self._num_workers = num_workers

        logger.info('Parsing dataset files ...')
        self._metadata = NIHDataset.parse_dataset_meta(
            dataset_path=self._dataset_path,
            split_type=split_type,
            classes=classes
        )
        logger.info('Dataset files parsed!')

    @property
    def classes(self) -> List[str]:
        return self._metadata['classes']

    @property
    def _current_phase(self) -> DictConfig:
        assert self._phases
        for phase in reversed(self._phases):
            if self.trainer.current_epoch >= phase.epoch_milestone:
                return phase

    @property
    def _transforms_train(self) -> A.Compose:
        size = self._current_phase.image_size
        pre_crop_size = int(size * 1.1)
        p = self._current_phase.augment_rate

        return A.Compose([
            A.OneOrOther(
                first=A.Compose([
                    A.Resize(pre_crop_size, pre_crop_size),
                    A.RandomCrop(size, size),
                ]),
                second=A.Resize(size, size),
                p=p
            ),
            A.MultiplicativeNoise(p=p),
            A.GridDropout(ratio=0.1, holes_number_x=20, holes_number_y=20, p=p),
            A.Rotate(limit=45, p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.Normalize(max_pixel_value=nih_const.MIN_MAX_VALUE[1],
                        mean=[nih_const.MEAN] * 3,
                        std=[nih_const.STD] * 3),
            ToTensorV2()
        ])

    @property
    def _transforms_val(self) -> A.Compose:
        size = self._current_phase.image_size
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(max_pixel_value=nih_const.MIN_MAX_VALUE[1],
                        mean=[nih_const.MEAN] * 3,
                        std=[nih_const.STD] * 3),
            ToTensorV2()
        ])

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader('val')

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader('test')

    def _create_dataloader(self, split: str) -> DataLoader:
        assert split in ['train', 'val', 'test']
        shuffle = split == 'train'
        transforms = self._transforms_train if split == 'train' else self._transforms_val
        split_set = NIHDataset(
            dataset_path=self._dataset_path,
            input_df=self._metadata[f'{split}_df'],
            transforms=transforms,
            mode=NIHDataset.CLASSIFICATION_MODE
        )
        return DataLoader(split_set, batch_size=self._current_phase.batch_size,
                          shuffle=shuffle, num_workers=self._num_workers,
                          collate_fn=NIHClassificationDataModule.collate_batch)

    def get_train_class_freq(self) -> Tuple[List[float], int]:
        labels = self._metadata['test_df']["Label"].tolist()
        labels = np.array(labels, dtype=float)

        samples_count = len(labels)
        samples_per_class = np.sum(labels, axis=0)

        return list(map(lambda x: float(x), samples_per_class)), samples_count

    @staticmethod
    def collate_batch(batch: List[Tuple[List, List]]) -> Tuple[torch.Tensor, ...]:
        batch_image_list, target_list = zip(*batch)

        labels = [torch.tensor(target) for target in target_list]

        return torch.stack(batch_image_list).float(), torch.stack(labels).float()
