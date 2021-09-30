from pathlib import Path
from typing import Tuple, List

import numpy as np
import zarr
from numcodecs import Blosc


class PredZarrReader:

    def __init__(self, dest_path: Path):
        self._dest_path = dest_path

        self._store = zarr.ZipStore(dest_path.as_posix(), mode='r')

        self.root: zarr.Group = None

    def __enter__(self):
        self.root = zarr.open(self._store)
        return self

    def __exit__(self, *args):
        self._store.close()
        self.root = None

    def read_pred_output(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        return self.root['preds'][:], self.root['targets'][:], self.root.attrs['classes']


class PredZarrWriter:

    def __init__(self, src_path: Path):
        self._src_path = src_path

        self._store = zarr.ZipStore(src_path.as_posix(), mode='w')
        self._compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

        self.root: zarr.Group = None

    def __enter__(self):
        self.root = zarr.open(self._store)
        return self

    def __exit__(self, *args):
        self._store.close()
        self.root = None

    def write_pred_output(
            self,
            preds: np.ndarray,
            targets: np.ndarray,
            classes: List[str]
    ):
        self.root['preds'] = preds
        self.root['targets'] = targets
        self.root.attrs['classes'] = classes
