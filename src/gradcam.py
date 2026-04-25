"""
gradcam.py
----------
Gradient-weighted Class Activation Mapping (Grad-CAM) implementation for
visualising the image regions most influential to a model's prediction.

Reference
---------
Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep
networks via gradient-based localization. ICCV.

Classes
-------
GradCAM
    Attaches forward and backward hooks to a target layer, captures
    activations and gradients, and produces a normalised CAM heatmap.

Usage
-----
    gradcam = GradCAM(model, target_layer=model.layer4)
    cam = gradcam.generate_heatmap(img_tensor, device)
    gradcam.remove_hooks()
"""

import numpy as np


# ---------------------------------------------------------------------------
# GradCAM Class
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM visualisation for a single target layer.

    Hooks are registered on instantiation and must be explicitly removed
    after use by calling remove_hooks() to avoid unintended side effects
    on subsequent forward passes.

    Parameters
    ----------
    model : torch.nn.Module
        The model to explain. Must be in eval mode during heatmap generation
        to obtain correct batch normalisation behaviour, but gradients must
        not be disabled (do not use torch.no_grad()).
    target_layer : torch.nn.Module
        The convolutional layer from which activations and gradients are
        extracted. Recommended layers per architecture:
            ResNet-50       : model.layer4
            DenseNet-121    : model.features.denseblock4
            EfficientNet-B0 : model.features[8]

    Attributes
    ----------
    activations : torch.Tensor or None
        Feature maps captured during the forward pass.
    gradients : torch.Tensor or None
        Gradients captured during the backward pass.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None

        self.forward_hook  = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_backward_hook(self._save_gradients)

    # ------------------------------------------------------------------
    # Hook Callbacks
    # ------------------------------------------------------------------

    def _save_activations(self, module, input, output):
        """Capture and store the forward pass feature maps."""
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Capture and store the backward pass gradients."""
        self.gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Heatmap Generation
    # ------------------------------------------------------------------

    def generate_heatmap(self, img_tensor, device):
        """
        Perform a forward and backward pass to generate a Grad-CAM heatmap.

        The output is a 2D array of the same spatial size as the target
        layer's feature maps (e.g. 7x7 for ResNet-50 layer4). Resize to
        the input image resolution before overlaying.

        Parameters
        ----------
        img_tensor : torch.Tensor
            Input image tensor of shape (1, 3, H, W) with batch dimension.
        device : torch.device
            Device on which to run the forward pass.

        Returns
        -------
        cam : np.ndarray
            Normalised heatmap of shape (H', W'), with values in [0, 1].
            H' and W' correspond to the spatial resolution of the target layer.
        """
        self.model.zero_grad()
        output = self.model(img_tensor.to(device)).squeeze()
        output.backward()

        grads_val   = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Global average pooling of gradients to obtain channel weights
        weights = np.mean(grads_val, axis=(1, 2))

        # Weighted sum of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # Apply ReLU and normalise to [0, 1]
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        return cam

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self):
        """
        Remove forward and backward hooks from the target layer.

        Always call this method after Grad-CAM generation is complete
        to prevent hooks from interfering with subsequent model usage.
        """
        self.forward_hook.remove()
        self.backward_hook.remove()