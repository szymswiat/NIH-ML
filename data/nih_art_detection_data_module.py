import logging
from pathlib import Path
from typing import List
from data import nih_const

import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from omegaconf import ListConfig, DictConfig
from torch.utils.data import DataLoader

import transforms as tfm
from data.nih_dataset import NIHDataset

logger = logging.getLogger(__name__)


class NIHArtDetectionDataModule(pl.LightningDataModule):

    def __init__(
            self,
            dataset_path: str,
            split_type: str,
            phases: ListConfig,
            num_workers: int = 2
    ):
        super().__init__()
        self._dataset_path = Path(dataset_path)
        self._phases = phases
        self._num_workers = num_workers

        logger.info('Parsing dataset files ...')
        self._metadata = NIHDataset.parse_dataset_meta(
            dataset_path=self._dataset_path,
            split_type=split_type
        )
        logger.info('Dataset files parsed!')

    @property
    def classes(self) -> List[str]:
        return self._metadata['classes_art']

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
            A.Rotate(limit=12, p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            tfm.NormalizeAlb(nih_const.MIN_MAX_VALUE, mean=[0] * 3, std=[1] * 3),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4))

    @property
    def _transforms_val(self) -> A.Compose:
        size = self._current_phase.image_size
        return A.Compose([
            A.Resize(size, size),
            tfm.NormalizeAlb(nih_const.MIN_MAX_VALUE, mean=[0] * 3, std=[1] * 3),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.4))

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
            mode=NIHDataset.BBOX_ART_DETECTION_MODE
        )
        return DataLoader(split_set, batch_size=self._current_phase.batch_size,
                          shuffle=shuffle, num_workers=self._num_workers,
                          collate_fn=lambda x: x)
