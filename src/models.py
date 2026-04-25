"""
models.py
---------
Transfer learning model definitions for chest X-ray pneumonia classification.

Each model is loaded with ImageNet pretrained weights. Early layers are frozen
to preserve general feature representations, while deeper layers and the
classifier head are unfrozen for task-specific fine-tuning.

Differential learning rates are applied at the optimiser level (see training
notebook): 1e-4 for convolutional blocks, 1e-3 for the classifier head.

Functions
---------
get_resnet50(num_classes)
    ResNet-50 with layers 3, 4, and FC unfrozen.

get_densenet121(num_classes)
    DenseNet-121 with denseblock3 onwards and classifier unfrozen.

get_efficientnet_b0(num_classes)
    EfficientNet-B0 with feature blocks 5-8 and classifier unfrozen.

get_model(model_name, num_classes)
    Factory function to retrieve a model by name.
"""

import torchvision.models as models
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_param_summary(model):
    """Print trainable vs total parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} / {total:,}")


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

def get_resnet50(num_classes=1):
    """
    ResNet-50 pretrained on ImageNet with partial fine-tuning.

    Frozen layers  : conv1, bn1, relu, maxpool, layer1, layer2
    Unfrozen layers: layer3, layer4, fc (classifier)

    Parameters
    ----------
    num_classes : int
        Number of output units. Default is 1 for binary classification
        with BCEWithLogitsLoss.

    Returns
    -------
    model : torch.nn.Module
    """
    model = models.resnet50(weights="IMAGENET1K_V2")

    frozen_layers = {'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2'}
    for name, layer in model.named_children():
        if name in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    _print_param_summary(model)
    return model


def get_densenet121(num_classes=1):
    """
    DenseNet-121 pretrained on ImageNet with partial fine-tuning.

    Frozen layers  : conv0, norm0, relu0, pool0, denseblock1,
                     transition1, denseblock2, transition2
    Unfrozen layers: denseblock3, transition3, denseblock4, norm5,
                     classifier

    Parameters
    ----------
    num_classes : int
        Number of output units. Default is 1 for binary classification
        with BCEWithLogitsLoss.

    Returns
    -------
    model : torch.nn.Module
    """
    model = models.densenet121(weights="IMAGENET1K_V1")

    frozen_layers = {
        'conv0', 'norm0', 'relu0', 'pool0',
        'denseblock1', 'transition1',
        'denseblock2', 'transition2'
    }
    for name, layer in model.features.named_children():
        if name in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    _print_param_summary(model)
    return model


def get_efficientnet_b0(num_classes=1):
    """
    EfficientNet-B0 pretrained on ImageNet with partial fine-tuning.

    Frozen layers  : features[0] through features[4]
    Unfrozen layers: features[5] through features[8], classifier

    Parameters
    ----------
    num_classes : int
        Number of output units. Default is 1 for binary classification
        with BCEWithLogitsLoss.

    Returns
    -------
    model : torch.nn.Module
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    frozen_blocks = {'0', '1', '2', '3', '4'}
    for name, layer in model.features.named_children():
        if name in frozen_blocks:
            for param in layer.parameters():
                param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    _print_param_summary(model)
    return model


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    'resnet50':        get_resnet50,
    'densenet121':     get_densenet121,
    'efficientnet_b0': get_efficientnet_b0,
}


def get_model(model_name, num_classes=1):
    """
    Retrieve a model by name from the model registry.

    Parameters
    ----------
    model_name : str
        One of 'resnet50', 'densenet121', 'efficientnet_b0'.
    num_classes : int
        Number of output units. Default is 1.

    Returns
    -------
    model : torch.nn.Module

    Raises
    ------
    ValueError
        If model_name is not in the registry.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available options: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[model_name](num_classes=num_classes)