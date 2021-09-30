import numpy as np
import torch
from albumentations import ImageOnlyTransform
from typing import Tuple, List, Union


class NormalizeBase:

    def __init__(
            self,
            min_max_values: Tuple[int, int],
            mean: List[float],
            std: List[float]
    ):
        self._min_max_value = min_max_values
        self._range_value = abs(min_max_values[0]) + abs(min_max_values[1])

        self._mean = mean
        self._std = std

    def scale_values(
            self,
            x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        min_val, max_val = self._min_max_value

        if min_val < 0:
            x = x - min_val

        x = x / self._range_value

        return x

    def apply_mean_std(
            self,
            x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return (x - np.array(self._mean)) / np.array(self._std)


class NormalizeAlb(NormalizeBase, ImageOnlyTransform):

    def __init__(
            self,
            min_max_values: Tuple[int, int],
            mean: List[float],
            std: List[float]
    ):
        ImageOnlyTransform.__init__(self, always_apply=True)
        NormalizeBase.__init__(self, min_max_values, mean, std)

    def apply(self, img: np.ndarray, **params):
        img = self.scale_values(img)

        return self.apply_mean_std(img)

    def get_transform_init_args_names(self):
        return 'min_max_values', 'mean', 'std'


class NormalizeTorch(NormalizeBase, torch.nn.Module):

    def __init__(
            self,
            min_max_values: Tuple[int, int],
            mean: List[float],
            std: List[float]
    ):
        NormalizeBase.__init__(self, min_max_values, mean, std)
        torch.nn.Module.__init__(self)

    def forward(self, img: torch.Tensor):
        img = self.scale_values(img)
        img = img.permute(1, 2, 0)

        img = self.apply_mean_std(img)

        return img.permute(2, 0, 1)
