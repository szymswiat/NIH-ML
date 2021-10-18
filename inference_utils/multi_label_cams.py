import torch
from torch import Tensor
from torch.nn import Module

from inference_utils.multi_label_cam import CAMBaseMultiLabel


class GradCAMPlusPlusMultiLabel(CAMBaseMultiLabel):

    def __init__(self, model: Module, target_layer: Module):
        super().__init__(model, target_layer)

    def get_cam_weights(
            self,
            input_tensor: Tensor,
            target_category: int,
            activations: Tensor,
            grads: Tensor
    ) -> Tensor:
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = torch.sum(activations, dim=(1, 2))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))

        weights = torch.maximum(grads, torch.zeros_like(grads)) * aij
        weights = torch.sum(weights, dim=(1, 2))
        return weights
