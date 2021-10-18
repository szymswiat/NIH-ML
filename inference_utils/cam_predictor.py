from typing import Any

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module

from inference_utils.multi_label_cams import GradCAMPlusPlusMultiLabel


class CamPredictorMultiLabel(LightningModule):

    def __init__(
            self,
            model: LightningModule,
            thresholds: Tensor,
            target_layer: Module
    ):
        super().__init__()

        self.model = model
        self.thresholds = thresholds

        self.cam = GradCAMPlusPlusMultiLabel(model, target_layer=target_layer)

    def forward(self, image: Tensor) -> Any:
        y_pred = self.cam.predict(image)

        y_pred_positive = (y_pred >= self.thresholds).nonzero().squeeze(dim=1)

        pred_positive_count = len(y_pred_positive)

        if pred_positive_count > 0:
            grayscale_cams = self.cam.generate_cams(target_categories=y_pred_positive)

            return y_pred, list(zip(y_pred_positive, grayscale_cams))

        return y_pred, []
