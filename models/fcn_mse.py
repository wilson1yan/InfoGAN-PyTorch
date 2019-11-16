import torch.nn as nn
import torch

class FCN_mse(nn.Module):
    """
    Predict whether pixels are part of the object or the background.
    """

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        c1 = torch.tanh(self.conv1(x))
        c2 = torch.tanh(self.conv2(c1))
        score = (self.classifier(c2))  # size=(N, n_class, H, W)
        return score  # size=(N, n_class, x.H/1, x.W/1)