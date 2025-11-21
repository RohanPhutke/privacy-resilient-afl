# models.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, mobilenet_v2

def get_model(name='SimpleCNN', num_classes=10, dataset='CIFAR10'):
    """Returns the specified model architecture."""
    if name == 'SimpleCNN':
        in_channels = 1 if dataset == 'MNIST' else 3
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)

    elif name == 'ResNet18':
        model = resnet18(weights=None, num_classes=num_classes)
        if dataset == 'MNIST':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model

    elif name == 'MobileNetV2':
        model = mobilenet_v2(weights=None, num_classes=num_classes)
        if dataset == 'MNIST':
            old_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
        return model

    elif name == 'ViT':
        try:
            import timm
            if dataset == 'MNIST':
                model = timm.create_model('vit_tiny_patch16_224', pretrained=False, 
                                          num_classes=num_classes, in_chans=1, img_size=28)
            else:
                model = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                                          num_classes=num_classes, in_chans=3, img_size=32)
            return model
        except ImportError:
            raise ImportError("Please install timm: pip install timm")

    else:
        raise ValueError(f"Unknown model name: {name}")


class SimpleCNN(nn.Module):
    """A simple CNN for baseline tests."""
    def __init__(self, num_classes=10, in_channels=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        if in_channels == 1: 
            flat_dim = 64 * 4 * 4
        else: 
            flat_dim = 64 * 5 * 5
        self.fc1 = nn.Linear(flat_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.flat_dim = flat_dim

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, self.flat_dim)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x