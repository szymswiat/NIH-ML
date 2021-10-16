from typing import Dict, List

from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from detmet import DetMetric
from detutils import CoordinatesType, BBType, BBFormat, MethodAveragePrecision
from torch import Tensor
from torchmetrics import Metric


class DetectionMetric(Metric):

    def __init__(
            self,
            classes: List[str],
            iou_match_threshold=0.5
    ):
        super().__init__(compute_on_step=False)

        self._classes = classes

        self._iou_match_threshold = iou_match_threshold

        self.all_pred_data: List[Dict[str, Tensor]] = []
        self.all_target_data: List[Dict[str, Tensor]] = []

        # self.add_state('all_pred_data', default=[])
        # self.add_state('all_target_data', default=[])

    def update(self, target_list: List[Dict[str, Tensor]], pred_list: List[Dict[str, Tensor]]) -> None:
        for p, t in zip(pred_list, target_list):
            self.all_pred_data.append(p)
            self.all_target_data.append(t)

    def compute(self):
        all_bboxes = BoundingBoxes()

        for i, img_data in enumerate(self.all_target_data):
            for label_idx, box in zip(img_data['labels'], img_data['boxes']):
                bbox = BoundingBox(
                    imageName=f'{i}',
                    classId=self._classes[label_idx],
                    x=float(box[0]), y=float(box[1]),
                    w=float(box[2]), h=float(box[3]),
                    typeCoordinates=CoordinatesType.Absolute,
                    bbType=BBType.GroundTruth,
                    format=BBFormat.XYX2Y2
                )
                all_bboxes.addBoundingBox(bbox)

        for i, img_data in enumerate(self.all_pred_data):
            for label_idx, box, score in zip(img_data['labels'], img_data['boxes'], img_data['scores']):
                bbox = BoundingBox(
                    imageName=f'{i}',
                    classId=self._classes[label_idx],
                    x=float(box[0]), y=float(box[1]),
                    w=float(box[2]), h=float(box[3]),
                    typeCoordinates=CoordinatesType.Absolute,
                    bbType=BBType.Detected,
                    classConfidence=score,
                    format=BBFormat.XYX2Y2
                )
                all_bboxes.addBoundingBox(bbox)

        evaluator = DetMetric()
        metrics_per_class, metrics_all = evaluator.GetDetMetrics(
            all_bboxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=self._iou_match_threshold,  # IOU threshold
            beta=1,  # F1-score
            method=MethodAveragePrecision.ElevenPointInterpolation
        )
        return metrics_all, metrics_per_class
