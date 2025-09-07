import torch
import torch.nn as nn
import torchvision.models as models


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_transfer_model(model_name: str, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    name = model_name.lower()
    if name in {"mobilenet_v2", "mobilenetv2"}:
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for param in m.features.parameters():
                param.requires_grad = False
        return m
    if name in {"resnet18"}:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for param in m.parameters():
                param.requires_grad = False
            # Re-enable gradients for the new classifier
            for param in m.fc.parameters():
                param.requires_grad = True
        return m
    if name in {"efficientnet_b0", "efficientnet-b0"}:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for param in m.features.parameters():
                param.requires_grad = False
        return m
    raise ValueError(f"Unsupported model_name: {model_name}")
