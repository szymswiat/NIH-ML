import glob
import json
import logging
from itertools import chain
from pathlib import Path
from typing import List, Dict, Callable, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class NIHDataset(Dataset):
    CLASSIFICATION_MODE = 'cls'
    BBOX_MODE = 'bbox'
    BBOX_ONLY_MODE = 'bbox_only'

    # found with notebooks/data_mean_std.ipynb notebook
    MEAN = 0.4986
    STD = 0.2500
    MIN_MAX_VALUE = (0, 255)

    def __init__(
            self,
            dataset_path: Path,
            input_df: pd.DataFrame,
            transforms=A.Compose([]),
            mode=CLASSIFICATION_MODE
    ):
        self._dataset_path = dataset_path
        self._transforms = transforms
        self.mode = mode
        df = input_df.copy(deep=True)

        if type(df['Bboxes'].iloc[0]) == str:
            df['Bboxes'] = df['Bboxes'].map(lambda x: json.loads(x))
        if type(df['Label'].iloc[0]) == str:
            df['Label'] = df['Label'].map(lambda x: json.loads(x))

        if mode == self.BBOX_ONLY_MODE:
            df = df[df['Bboxes'] != {}]

        self._df = df

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        image_path = self._dataset_path / self._df['Image_Path'].iloc[idx]
        image = cv2.imread(image_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self._transforms(image=image)
        image = augmented['image']
        return image, dict(
            label=self._df['Label'].iloc[idx],
            bboxes=self._df['Bboxes'].iloc[idx]
        )

    @staticmethod
    def collate_labels(data: List[Dict]) -> Tuple[torch.Tensor, ...]:
        image_list, label_list = zip(*data)

        labels = [torch.tensor(labels['label']) for labels in label_list]

        return torch.stack(image_list).float(), torch.stack(labels).float()

    def collate_fn(self) -> Callable[[List[Dict]], Tuple[torch.Tensor, ...]]:
        if self.mode == self.CLASSIFICATION_MODE:
            return self.collate_labels
        elif self.mode == self.BBOX_MODE or self.mode == self.BBOX_ONLY_MODE:
            raise NotImplementedError()
        else:
            raise ValueError()

    @staticmethod
    def parse_dataset_meta(
            dataset_path: Path,
            validation_df_size=0.0,
            classes: List[str] = None,
            drop_no_findings_class=True
    ) -> Dict:
        data_entry_path = dataset_path / 'Data_Entry_2017.csv'
        bbox_list_path = dataset_path / 'BBox_List_2017.csv'

        data_entry_df = pd.read_csv(data_entry_path)
        bbox_list_df = pd.read_csv(bbox_list_path)

        logger.info('Scanning image tree...')
        all_image_paths = glob.glob(f'{dataset_path}/images_*/images/*.png', recursive=True)
        all_image_paths.sort()
        all_image_paths = {Path(x).name: Path(x).relative_to(dataset_path) for x in all_image_paths}
        logger.info(f'Total images found: {len(all_image_paths)}')

        data_entry_df['Finding Labels'] = data_entry_df['Finding Labels'].map(lambda x: x.replace('|', ','))

        if classes is None:
            classes = np.unique(list(chain(*data_entry_df['Finding Labels'].map(lambda x: x.split(',')).tolist())))
            classes = list(sorted([x for x in classes if len(x) > 0]))

            classes.remove('No Finding')
            if not drop_no_findings_class:
                classes.insert(0, 'No Finding')

        encoder = MultiLabelBinarizer(classes=classes)
        labels_all = encoder.fit_transform([c.split(',') for c in list(data_entry_df['Finding Labels'])])

        all_df = pd.DataFrame()

        all_df['Image_Index'] = data_entry_df['Image Index']
        all_df['Image_Path'] = all_df['Image_Index'].map(all_image_paths.get)
        all_df['Label'] = labels_all.tolist()
        all_df['Patient_ID'] = data_entry_df['Patient ID']

        grouped = bbox_list_df.groupby(by='Image Index').indices

        def extract_bboxes_for_img(img_index: str) -> List:
            if img_index not in grouped:
                return []
            indices = grouped[img_index]
            rows = bbox_list_df.iloc[indices]
            rows = rows[rows['Finding Label'].isin(classes)]
            labels = rows['Finding Label'].to_list()
            bboxes = rows[['Bbox [x', 'y', 'w', 'h]']].to_numpy()
            labels = encoder.fit_transform([[x] for x in labels])
            return [(label.tolist(), bbox.astype(int).tolist()) for label, bbox in zip(labels, bboxes)]

        all_df['Bboxes'] = all_df['Image_Index'].map(extract_bboxes_for_img)

        train_val_list = pd.read_csv(dataset_path / 'train_val_list.txt', header=None)
        train_val_list.columns = ['Image_Index']
        test_list = pd.read_csv(dataset_path / 'test_list.txt', header=None)
        test_list.columns = ['Image_Index']

        train_val_df = all_df.merge(train_val_list, on='Image_Index')
        test_df = all_df.merge(test_list, on='Image_Index')

        out_dct = {
            'classes': classes,
            'train_df': train_val_df,
            'test_df': test_df
        }

        # TODO: don't split dataset randomly
        if validation_df_size > 0.0:
            train_df, val_df = NIHDataset.split_df(train_val_df, 1.0 - validation_df_size)
            out_dct.update({
                'train_df': train_df,
                'val_df': val_df
            })

        logger.info('Dataframes generated!')

        return out_dct

    @staticmethod
    def split_df(df: pd.DataFrame, chunk1_frac=0.95, seed=0):
        chunk1 = df.sample(frac=chunk1_frac, random_state=seed)
        chunk2 = df.drop(chunk1.index)

        return chunk1, chunk2

    @staticmethod
    def save_dfs(
            dataset_path: Path,
            out_dir: Path,
            name_prefix: str,
            validation_df_size=0.0,
            drop_no_findings_class=True,
            classes: List[str] = None
    ) -> Dict:

        data = NIHDataset.parse_dataset_meta(dataset_path=dataset_path,
                                             validation_df_size=validation_df_size,
                                             classes=classes,
                                             drop_no_findings_class=drop_no_findings_class)

        out_dir.mkdir(parents=True, exist_ok=True)

        data['train_df'].to_csv(str(out_dir / f'{name_prefix}_train_df.csv'), index=False)
        data['test_df'].to_csv(str(out_dir / f'{name_prefix}_test_df.csv'), index=False)

        if 'val_df' in data:
            data['val_df'].to_csv(str(out_dir / f'{name_prefix}_val_df.csv'), index=False)

        with open(out_dir / f'{name_prefix}_classes.yaml', 'w') as f:
            yaml.dump(list(map(lambda x: str(x), data['classes'])), f)

        return data
