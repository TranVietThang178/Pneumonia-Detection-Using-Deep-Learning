import torchvision.models as models
import torch.nn as nn


def get_resnet50(num_classes=1):
    model = models.resnet50(weights="IMAGENET1K_V2")
    
    for name, layer in model.named_children():
        if name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2']:
            for param in layer.parameters():
                param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (f"Trainable: {trainable:,} / {total:,}")
    return model

def get_densenet121(num_classes=1):
    model = models.densenet121(weights="IMAGENET1K_V1")
    
    for name, layer in model.features.named_children():
        if name in ['conv0', 'norm0', 'relu0', 'pool0', 'denseblock1', 'transition1', 'denseblock2', 'transition2']:
            for param in layer.parameters():
                param.requires_grad = False

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (f"Trainable: {trainable:,} / {total:,}")
    return model

def get_efficientnet_b0(num_classes=1):
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    
    for name, layer in model.features.named_children():
        if name in ['0', '1', '2', '3', '4']:
            for param in layer.parameters():
                param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (f"Trainable: {trainable:,} / {total:,}")
    return model

def get_model(model_name, num_classes=1):
    models_dict = {
        'resnet50': get_resnet50,
        'densenet121': get_densenet121,
        'efficientnet_b0': get_efficientnet_b0
    }
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models_dict.keys())}")
    return models_dict[model_name](num_classes=num_classes)