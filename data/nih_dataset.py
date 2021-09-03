import json
from pathlib import Path

import cv2
import torch
from pandas import DataFrame
from torch.utils.data import Dataset
import albumentations as A


class NIHDataset(Dataset):
    def __init__(
            self,
            dataset_path: Path,
            df: DataFrame,
            transforms=A.Compose([])
    ):
        self.dataset_path = dataset_path
        self.df = df
        self.transforms = transforms

        self.paths = self.df["Path"].tolist()
        self.labels = self.df["Label"].tolist()
        self.labels = list(map(lambda row: json.loads(row), self.labels))
        self.labels = torch.tensor(self.labels, dtype=torch.float)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        label = self.labels[idx]

        image_path = self.dataset_path / self.paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transforms(image=image)
        image = augmented['image']
        return image, label
