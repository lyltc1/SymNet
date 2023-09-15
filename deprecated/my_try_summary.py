import torch
import torch.nn as nn
from torchsummary import summary


class SimpleConv(nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, y):
        x1 = self.features(x)
        x2 = self.features(y)
        return x1, x2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleConv().to(device)

summary(model, [(1, 16, 16), (1, 28, 28)])