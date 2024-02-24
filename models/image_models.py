import torch
import torch.nn as nn
from transforms.image_transforms import holz_transform
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50


class RensetNoDownClassifier(nn.Module):
    def __init__(self, dct_mean, dct_var, freq_mask):
        super(RensetNoDownClassifier, self).__init__()

        self.model = resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, 2)

        self.register_buffer('dct_mean', dct_mean, False)
        self.register_buffer('dct_var', dct_var, False)
        self.register_buffer('freq_mask', freq_mask, False)

    def transform(self, x):
        tensor = holz_transform(x)
        tensor = (tensor - self.dct_mean) / torch.sqrt(self.dct_var)
        tensor = tensor * self.freq_mask
        return tensor

    def forward(self, x):
        x = self.transform(x)
        x = self.model(x)
        return x
