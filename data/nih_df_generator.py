import glob
import os
from itertools import chain
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
import yaml


class NIHDfGenerator:

    @staticmethod
    def generate_and_save_dfs(dataset_path: Path):
        data_entry_path = join(dataset_path, 'Data_Entry_2017.csv')
        print('Reading DataEntry ...')
        all_xray_df = pd.read_csv(data_entry_path)

        print('Scanning ...')
        all_image_paths = glob.glob(f'{dataset_path}/images_*/images/*.png', recursive=True)
        all_image_paths.sort()
        all_image_paths = {os.path.basename(x): os.path.relpath(x, dataset_path) for x in all_image_paths}

        print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])

        all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
        all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

        all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('|', ','))

        classes = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split(',')).tolist())))
        classes = [x for x in classes if len(x) > 0]
        classes.sort()

        encoder = MultiLabelBinarizer(classes=classes)
        labels = encoder.fit_transform([c.split(',') for c in list(all_xray_df['Finding Labels'])])

        df = pd.DataFrame()

        df['Image Index'] = all_xray_df['Image Index']
        df['Label'] = labels.tolist()
        df['Path'] = all_xray_df['path']

        train_val_list = pd.read_fwf(str(dataset_path / 'train_val_list.txt'), header=None).squeeze()
        train_val_list.head()
        train_val_df = df.loc[all_xray_df['Image Index'].isin(train_val_list)]

        train_df, val_df = NIHDfGenerator.split_df(train_val_df)

        test_list = pd.read_fwf(str(dataset_path / 'test_list.txt'), header=None).squeeze()
        test_df = df.loc[all_xray_df['Image Index'].isin(test_list)]
        test_df.head()

        print(
            '\n'
            f'trainset : {len(train_df)}\n'
            f'valset   : {len(val_df)}\n'
            f'testset  : {len(test_df)}\n'
            f'total    : {len(train_df) + len(val_df) + len(test_df)}'
        )

        train_df.to_csv(str(dataset_path / 'orig_train_df.csv'), index=False)
        val_df.to_csv(str(dataset_path / 'orig_val_df.csv'), index=False)
        test_df.to_csv(str(dataset_path / 'orig_test_df.csv'), index=False)

        with open(dataset_path / 'classes.yaml', 'w') as f:
            yaml.dump(list(map(lambda x: str(x), classes)), f)

    @staticmethod
    def split_df(df: DataFrame, chunk1_frac=0.95, seed=0):
        chunk1 = df.sample(frac=chunk1_frac, random_state=seed)
        chunk2 = df.drop(chunk1.index)

        return chunk1, chunk2
