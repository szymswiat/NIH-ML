import glob
import json
import logging
from itertools import chain
from pathlib import Path
from typing import List, Dict, Callable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class NIHDataset(Dataset):
    CLASSIFICATION_MODE = 'cls'
    BBOX_CLASSIFICATION_MODE = 'bbox'
    BBOX_ART_DETECTION_MODE = 'bbox_art'

    # found with notebooks/data_mean_std.ipynb notebook
    MEAN = 0.4986
    STD = 0.2500
    MIN_MAX_VALUE = (0, 255)

    SPLIT_OFFICIAL_WITH_VAL = 'official_with_val'
    SPLIT_OFFICIAL_VAL_FROM_TEST = 'official_val_from_test'
    SPLIT_STRATIFIED = 'stratified'

    def __init__(
            self,
            dataset_path: Path,
            input_df: pd.DataFrame,
            transforms=A.Compose([]),
            mode=CLASSIFICATION_MODE,
            filter_by_positive_class: List[str] = None,
    ):
        self._dataset_path = dataset_path
        self._transforms = transforms
        self.mode = mode
        df = input_df.copy(deep=True)

        if type(df['Bboxes'].iloc[0]) == str:
            df['Bboxes'] = df['Bboxes'].map(lambda x: json.loads(x))
        if type(df['Label'].iloc[0]) == str:
            df['Label'] = df['Label'].map(lambda x: json.loads(x))

        # if mode == self.BBOX_ONLY_MODE:
        #     df = df[df.apply(lambda x: x['Bboxes'] != [], axis=1)]

        if filter_by_positive_class is not None:
            df = df[df.apply(lambda row: len(set(row['Label_Str']) & set(filter_by_positive_class)) > 0, axis=1)]

        if mode == self.BBOX_ART_DETECTION_MODE:
            df = self._extract_bbox_art_df(df)

        self._df = df

    def __len__(self):
        return len(self._df)

    def get_img_path(self, idx: int) -> Path:
        return Path(self._df['Image_Path'].iloc[idx])

    def __getitem__(self, idx):
        image_path = self._dataset_path / self.get_img_path(idx)
        image = cv2.imread(image_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform_params = dict(
            image=image
        )

        target = None
        if self.mode == self.CLASSIFICATION_MODE:
            target = self._df['Label'].iloc[idx]
        elif self.mode == self.BBOX_CLASSIFICATION_MODE:
            transform_params['bboxes'] = self._df['Bboxes'].iloc[idx]
        elif self.mode == self.BBOX_ART_DETECTION_MODE:
            transform_params['bboxes'] = self._df['Bboxes_Art'].iloc[idx]
        else:
            raise ValueError()

        transformed = self._transforms(**transform_params)

        if 'bboxes' in transform_params:
            target = transformed['bboxes']

        return transformed['image'], target

    def _extract_bbox_art_df(self, all_df: DataFrame) -> DataFrame:
        df = all_df[all_df.apply(lambda x: x['Bboxes_Art'] != [], axis=1)]

        art_labels_flat = np.array([single_label[-1] for img_labels in df['Bboxes_Art'].to_list()
                                    for single_label in img_labels], dtype=int)
        _, counts = np.unique(art_labels_flat, return_counts=True)
        mean_img_per_art_class = int(counts.mean())

        no_label_img_df = all_df[all_df.apply(lambda x: x['Bboxes_Art'] == [], axis=1)]
        no_label_img_df = no_label_img_df.sample(mean_img_per_art_class, random_state=0)

        return df.append(no_label_img_df)

    @staticmethod
    def parse_dataset_meta(
            dataset_path: Path,
            split_type: str = SPLIT_OFFICIAL_WITH_VAL,
            classes: List[str] = None,
            drop_no_findings_class=True
    ) -> Dict:
        data_entry_path = dataset_path / 'Data_Entry_2017.csv'
        bbox_list_path = dataset_path / 'BBox_List_2017.csv'
        artifacts_path = dataset_path / 'Artifacts.csv'

        data_entry_df = pd.read_csv(data_entry_path)
        bbox_list_df = pd.read_csv(bbox_list_path)
        artifact_list_df = pd.read_csv(artifacts_path)

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

        classes_art = list(sorted(np.unique(artifact_list_df['label'].to_list())))

        encoder = MultiLabelBinarizer(classes=classes)

        labels_str = [c.split(',') for c in list(data_entry_df['Finding Labels'])]
        labels_all = encoder.fit_transform(labels_str)

        all_df = pd.DataFrame()

        all_df['Image_Index'] = data_entry_df['Image Index']
        all_df['Image_Path'] = all_df['Image_Index'].map(all_image_paths.get)
        all_df['Label'] = labels_all.tolist()
        all_df['Label_Str'] = labels_str
        all_df['Patient_ID'] = data_entry_df['Patient ID']

        grouped_rows = bbox_list_df.groupby(by='Image Index').indices
        all_df['Bboxes'] = all_df['Image_Index'].map(
            NIHDataset.extract_bboxes_for_img(bbox_list_df, grouped_rows, classes)
        )

        grouped_rows = artifact_list_df.groupby(by='image_id').indices
        all_df['Bboxes_Art'] = all_df['Image_Index'].map(
            NIHDataset.extract_artifact_bboxes_for_img(artifact_list_df, grouped_rows, classes_art)
        )

        train_val_list = pd.read_csv(dataset_path / 'train_val_list.txt', header=None)
        train_val_list.columns = ['Image_Index']
        test_list = pd.read_csv(dataset_path / 'test_list.txt', header=None)
        test_list.columns = ['Image_Index']

        train_val_df = all_df.merge(train_val_list, on='Image_Index')
        test_df = all_df.merge(test_list, on='Image_Index')

        out_dct = {
            'classes': classes,
            'classes_art': classes_art,
            'train_df': train_val_df,
            'test_df': test_df
        }

        if split_type == NIHDataset.SPLIT_OFFICIAL_WITH_VAL:
            # TODO: don't split dataset randomly
            train_df, val_df = NIHDataset.split_df(train_val_df, 1.0 - 0.05)
            out_dct.update({
                'train_df': train_df,
                'val_df': val_df
            })
        elif split_type == NIHDataset.SPLIT_OFFICIAL_VAL_FROM_TEST:
            # TODO: don't split dataset randomly
            test_df, val_df = NIHDataset.split_df(test_df, 1.0 - 0.25)
            out_dct.update({
                'test_df': test_df,
                'val_df': val_df
            })
        elif split_type == NIHDataset.SPLIT_STRATIFIED:
            raise NotImplementedError()
        else:
            raise ValueError(f'Invalid split type : {split_type}.')

        logger.info('Dataframes generated!')

        return out_dct

    @staticmethod
    def extract_bboxes_for_img(
            bbox_list_df: DataFrame,
            grouped_rows: Dict[str, List[int]],
            classes: List[str]
    ) -> Callable:

        def process(img_index: str) -> List[List[int]]:
            if img_index not in grouped_rows:
                return []
            indices = grouped_rows[img_index]
            rows = bbox_list_df.iloc[indices]
            rows = rows[rows['Finding Label'].isin(classes)]
            labels = rows['Finding Label'].map(lambda x: classes.index(x)).to_numpy(dtype=int)

            bboxes = rows[['Bbox [x', 'y', 'w', 'h]']].to_numpy()
            bboxes[..., 2:] = bboxes[..., :2] + bboxes[..., 2:]

            bboxes = np.concatenate([bboxes, np.expand_dims(labels, axis=-1)], axis=-1)
            return [bbox.tolist() for bbox in bboxes]

        return process

    @staticmethod
    def extract_artifact_bboxes_for_img(
            artifact_list_df: DataFrame,
            grouped_rows: Dict[str, List[int]],
            classes_art: List[str]
    ) -> Callable:

        def process(img_index: str) -> List:
            if img_index not in grouped_rows:
                return []
            indices = grouped_rows[img_index]
            rows = artifact_list_df.iloc[indices]

            labels = rows['label'].map(lambda x: classes_art.index(x)).to_numpy(dtype=int)
            bboxes = np.array(rows['bbox'].map(lambda x: json.loads(x)).to_list(), dtype=int)
            bboxes = np.concatenate([bboxes, np.expand_dims(labels, axis=-1)], axis=1)
            return [bbox.tolist() for bbox in bboxes]

        return process

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
            split_type: str,
            drop_no_findings_class=True,
            classes: List[str] = None
    ) -> Dict:
        data = NIHDataset.parse_dataset_meta(dataset_path=dataset_path,
                                             split_type=split_type,
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
