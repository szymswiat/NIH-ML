from typing import Any, Tuple, List

import torch
from torch import Tensor
from torch.nn import Module
from torchvision.transforms.functional import resize


class CAMBaseMultiLabel:

    def __init__(self, model: Module, target_layer: Module):
        self.ag_extractor = ActivationsAndGradientsExtractor(model, target_layer)

        self.input_tensor: Tensor = None
        self.output: Tensor = None

    def predict(self, input_tensor: Tensor) -> Tensor:
        if len(input_tensor.size()) > 3:
            raise ValueError('Data batching is not supported. Please provide single image.')

        self.input_tensor = input_tensor
        self.output = self.ag_extractor.run_forward(input_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        return self.output

    def get_cam_weights(
            self,
            input_tensor: Tensor,
            target_category: int,
            activations: Tensor,
            grads: Tensor
    ) -> Tensor:
        raise Exception("Not Implemented")

    def generate_cam(self, target_category: int) -> Tensor:
        activations, grads = self.ag_extractor(self.output[target_category])

        activations = activations[-1][0]
        grads = grads[-1][0]

        cam = self._get_cam_image(self.input_tensor, target_category, activations, grads)
        cam = torch.maximum(cam, torch.zeros_like(cam))

        cam = resize(cam.unsqueeze(dim=0), self.input_tensor.shape[-2:][::-1]).squeeze(dim=0)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)

        return cam.type(torch.float32)

    def generate_cams(self, target_categories: List[int]) -> List[Tensor]:
        return [self.generate_cam(tc) for tc in target_categories]

    def _get_cam_image(
            self,
            input_tensor: Tensor,
            target_category: int,
            activations: Tensor,
            grads: Tensor
    ) -> Tensor:
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        weighted_activations = weights[:, None, None] * activations

        return weighted_activations.sum(dim=0)


class ActivationsAndGradientsExtractor:

    def __init__(
            self,
            model: Module,
            target_layer: Module
    ):
        self.model = model
        self.gradients = []
        self.activations = []

        self.target_layer = target_layer

        for module in model.modules():
            module.register_forward_hook(self.update_req_grad)

        self.req_grad_updated = False
        self.req_grad_update_state = False

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

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

    def update_req_grad(self, module: Module, input: Tensor, output: Tensor):
        if self.req_grad_updated is False:
            if module == self.target_layer:
                self.req_grad_update_state = True
            module.requires_grad_(self.req_grad_update_state)

    def save_activation(self, module: Module, input: Tensor, output: Tensor):
        self.activations.append(output.cpu().detach())

    def save_gradient(self, module: Module, grad_input: Tensor, grad_output: Tensor):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0].cpu().detach()] + self.gradients
