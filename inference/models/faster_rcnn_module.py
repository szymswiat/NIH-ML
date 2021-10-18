from typing import Any

from omegaconf import DictConfig, OmegaConf
from timm.models.resnet import resnet34
from torch.nn import Identity
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, TwoMLPHead, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from data.nih_dataset import NIHDataset
from inference.models.base.loadable_module import LoadableModule


class FasterRCNNModule(LoadableModule):

    def __init__(self, hparams: DictConfig):
        super().__init__(hparams)

        self.classes = OmegaConf.to_object(hparams.dynamic.classes)

        image_size = hparams.phases[0].image_size

        backbone = resnet34(pretrained=True, num_classes=0)
        backbone.out_channels = backbone.num_features
        backbone.global_pool = Identity()

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        box_head = TwoMLPHead(
            in_channels=backbone.out_channels * box_roi_pool.output_size[0] ** 2,
            representation_size=128
        )
        box_predictor = FastRCNNPredictor(128, len(self.classes))

        self.model = FasterRCNN(
            backbone,
            box_roi_pool=box_roi_pool,
            rpn_anchor_generator=anchor_generator,
            box_head=box_head,
            box_predictor=box_predictor,
            min_size=image_size,
            max_size=image_size,
            image_mean=[NIHDataset.MEAN] * 3,
            image_std=[NIHDataset.STD] * 3
        )

    def forward(self, *args) -> Any:
        return self.model(*args)
