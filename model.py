import torch
import torchvision
from torch import nn
from torchvision import models

class FineTunedResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(FineTunedResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
