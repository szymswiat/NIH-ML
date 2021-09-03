import json
import os
import shutil
import zipfile
from pathlib import Path
from time import time
from typing import List, Dict, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from omegaconf import ListConfig, DictConfig
from torch.utils.data import DataLoader

from data.nih_dataset import NIHDataset
from data.nih_df_generator import NIHDfGenerator


class NIHDataModule(pl.LightningDataModule):
    NUM_CLASSES = 14

    def __init__(
            self,
            dataset_path: str,
            df_prefix='orig',
            phases: ListConfig = None,
            num_workers=2,
            copy_to_scratch=False,
            merge_train_val=False
    ):
        super().__init__()
        self._dataset_path = Path(dataset_path)
        self._phases = phases
        self._num_workers = num_workers
        self._copy_to_scratch = copy_to_scratch
        self._merge_train_val = merge_train_val

        df_files_root = self._dataset_path / 'df_split_files'
        self._train_df = pd.read_csv(str(df_files_root / f'{df_prefix}_train_df.csv'))
        self._val_df = pd.read_csv(str(df_files_root / f'{df_prefix}_val_df.csv'))
        self._test_df = pd.read_csv(str(df_files_root / f'{df_prefix}_test_df.csv'))

        if self._merge_train_val:
            self._train_df = self._train_df.append(self._val_df)

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
            A.Normalize(mean=[0.5055] * 3, std=[0.1712] * 3),
            ToTensorV2()
        ])

    @property
    def _transforms_val(self) -> A.Compose:
        size = self._current_phase.image_size
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=[0.5055] * 3, std=[0.1712] * 3),
            ToTensorV2()
        ])

    def train_dataloader(self) -> DataLoader:
        train_set = NIHDataset(
            dataset_path=self._dataset_path,
            df=self._train_df,
            transforms=self._transforms_train,
        )
        return DataLoader(train_set, batch_size=self._current_phase.batch_size,
                          shuffle=True, num_workers=self._num_workers)

    def val_dataloader(self) -> DataLoader:
        assert self._merge_train_val is False

        val_set = NIHDataset(
            dataset_path=self._dataset_path,
            df=self._val_df,
            transforms=self._transforms_val,
        )
        return DataLoader(val_set, batch_size=self._current_phase.batch_size,
                          shuffle=False, num_workers=self._num_workers)

    def test_dataloader(self) -> DataLoader:
        test_set = NIHDataset(
            dataset_path=self._dataset_path,
            df=self._test_df,
            transforms=self._transforms_val,
        )
        return DataLoader(test_set, batch_size=self._current_phase.batch_size,
                          shuffle=False, num_workers=self._num_workers)

    def get_train_class_freq(self) -> Tuple[List[float], int]:
        column = self._train_df["Label"].tolist()
        labels = list(map(lambda row: json.loads(row), column))
        labels = np.array(labels, dtype=float)

        samples_count = len(labels)
        samples_per_class = np.sum(labels, axis=0)

        return list(map(lambda x: float(x), samples_per_class)), samples_count
