import torch
import torch.nn as nn
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):#
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, img_tensor, device):
        """
        input_image: tensor of shape [1, 3, 224, 224] (with batch dimension)
        device: torch device
        """
        self.model.zero_grad()
        output = self.model(img_tensor.to(device)).squeeze()

        output.backward()

        grads_val = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(grads_val, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()