import os
import shutil
import zipfile
from pathlib import Path
from time import time
from typing import List, Tuple, Any

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
import logging
from albumentations.pytorch import ToTensorV2
from omegaconf import ListConfig, DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from data.nih_dataset import NIHDataset
import transforms as tfm

logger = logging.getLogger(__name__)


class NIHDataModule(pl.LightningDataModule):

    def __init__(
            self,
            dataset_path: str,
            phases: ListConfig = None,
            num_workers: int = 2,
            copy_to_scratch: bool = False,
            merge_train_val: bool = False,
            classes: List[str] = None
    ):
        super().__init__()
        self._dataset_path = Path(dataset_path)
        self._phases = phases
        self._num_workers = num_workers
        self._copy_to_scratch = copy_to_scratch
        self._merge_train_val = merge_train_val

        logger.info('Parsing dataset files ...')
        self._metadata = NIHDataset.parse_dataset_meta(
            dataset_path=self._dataset_path,
            validation_df_size=0.0 if merge_train_val else 0.05,
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

    def prepare_data(self):
        if self._copy_to_scratch:
            # TODO update NIHDfGenerator.generate_and_save_dfs with custom df support
            assert False
            start = time()
            scratch_dataset_root = Path(os.environ['SCRATCH_LOCAL']) / 'NihDataset'
            scratch_dataset_root.mkdir()
            shutil.copy(self._dataset_path, scratch_dataset_root)

            scratch_dataset = scratch_dataset_root / self._dataset_path.name

            print('Extracting data ...')
            with zipfile.ZipFile(str(scratch_dataset), 'r') as zip_ref:
                zip_ref.extractall(str(scratch_dataset_root))
            print(f'Dataset copied and extracted. Time {time() - start}s.')
            self._dataset_path = scratch_dataset_root
            print('Generating dataframes ...')
            NIHDfGenerator.generate_and_save_dfs(self._dataset_path)
            print('Dataframes generated.')

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
            tfm.NormalizeAlb(NIHDataset.MIN_MAX_VALUE,
                             mean=[NIHDataset.MEAN] * 3,
                             std=[NIHDataset.STD] * 3),
            ToTensorV2()
        ])

    @property
    def _transforms_val(self) -> A.Compose:
        size = self._current_phase.image_size
        return A.Compose([
            A.Resize(size, size),
            tfm.NormalizeAlb(NIHDataset.MIN_MAX_VALUE,
                             mean=[NIHDataset.MEAN] * 3,
                             std=[NIHDataset.STD] * 3),
            ToTensorV2()
        ])

    def train_dataloader(self) -> DataLoader:
        train_set = NIHDataset(
            dataset_path=self._dataset_path,
            input_df=self._metadata['train_df'],
            transforms=self._transforms_train
        )
        return DataLoader(train_set, batch_size=self._current_phase.batch_size,
                          shuffle=True, num_workers=self._num_workers,
                          collate_fn=train_set.collate_fn())

    def val_dataloader(self) -> DataLoader:
        assert self._merge_train_val is False

        val_set = NIHDataset(
            dataset_path=self._dataset_path,
            input_df=self._metadata['val_df'],
            transforms=self._transforms_val
        )
        return DataLoader(val_set, batch_size=self._current_phase.batch_size,
                          shuffle=False, num_workers=self._num_workers,
                          collate_fn=val_set.collate_fn())

    def test_dataloader(self) -> DataLoader:
        test_set = NIHDataset(
            dataset_path=self._dataset_path,
            input_df=self._metadata['test_df'],
            transforms=self._transforms_val
        )
        return DataLoader(test_set, batch_size=self._current_phase.batch_size,
                          shuffle=False, num_workers=self._num_workers,
                          collate_fn=test_set.collate_fn())

    def get_train_class_freq(self) -> Tuple[List[float], int]:
        labels = self._metadata['test_df']["Label"].tolist()
        labels = np.array(labels, dtype=float)

        samples_count = len(labels)
        samples_per_class = np.sum(labels, axis=0)

        return list(map(lambda x: float(x), samples_per_class)), samples_count


class NIHInferenceModuleWrapper(pl.LightningModule):

    def __init__(
            self,
            model: pl.LightningModule,
            img_size: Tuple[int, int],
            min_max_value: Tuple[int, int],
            mean: float,
            std: float
    ):
        super().__init__()

        self.model = model
        self.model.eval()

        self._min_max_value = min_max_value
        self._range_value = abs(min_max_value[0]) + abs(min_max_value[1])

        self._pre_transforms = transforms.Compose([
            transforms.Resize(size=img_size),
            tfm.NormalizeTorch(min_max_value, mean=[mean] * 3, std=[std] * 3)
        ])

    def forward(self, x) -> Any:
        x2 = torch.stack([self._pre_transforms(img) for img in x])

        return self.model(x2.float())
