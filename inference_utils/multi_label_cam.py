from collections import Callable
from typing import Any, Tuple, List

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.base_cam import BaseCAM
from torch import Tensor
from torch.nn import Module
import cv2
import numpy as np


class MultiLabelCAM(BaseCAM):

    def __init__(self, model: Module, target_layer: Module, use_cuda: bool = False):
        super().__init__(model, target_layer, use_cuda=use_cuda)

        self.ag_extractor = ActivationsAndGradientsExtractor(model, target_layer, None)

        self.input_tensor: Tensor = None
        self.output: Tensor = None

    def predict(self, input_tensor: Tensor) -> Tensor:
        if len(input_tensor.size()) > 3:
            raise ValueError('Data batching is not supported. Please provide single image.')
        if self.cuda:
            input_tensor = input_tensor.cuda()

        self.input_tensor = input_tensor
        self.output = self.ag_extractor.run_forward(input_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        return self.output

    def generate_cams(self, target_categories: List[int], eigen_smooth=False) -> List[np.ndarray]:
        results = []

        for target_category in target_categories:

            activations, grads = self.ag_extractor(self.output[target_category])

            activations = activations[-1].cpu().data.numpy()
            grads = grads[-1].cpu().data.numpy()

            cam = self.get_cam_image(self.input_tensor, target_category,
                                     activations, grads, eigen_smooth)
            cam = np.maximum(cam, 0)

            result = []
            for img in cam:
                img = cv2.resize(img, self.input_tensor.shape[-2:][::-1])
                img = img - np.min(img)
                img = img / np.max(img)
                result.append(img)

            results.append(np.squeeze(np.float32(result), axis=0))

        return results


class ActivationsAndGradientsExtractor(ActivationsAndGradients):

    def __init__(
            self,
            model: Module,
            target_layer: Module,
            reshape_transform: Callable
    ):
        super().__init__(model, target_layer, reshape_transform)

        self.target_layer = target_layer

        for module in model.modules():
            module.register_forward_hook(self.update_req_grad)

        self.req_grad_updated = False
        self.req_grad_update_state = False

    def __call__(self, backward_source: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        self.gradients = []

        self.model.zero_grad()

        backward_source.backward(retain_graph=True)

        return self.activations, self.gradients

    def run_forward(self, inputs: Tensor) -> Any:
        self.gradients = []
        self.activations = []

        outputs = self.model(inputs)
        self.req_grad_updated = True

        return outputs

    def update_req_grad(self, module: Module, input: Tensor, output: Any):
        if self.req_grad_updated is False:
            if module == self.target_layer:
                self.req_grad_update_state = True
            module.requires_grad_(self.req_grad_update_state)
