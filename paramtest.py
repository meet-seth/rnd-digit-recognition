import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_model = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(32, 64, 3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2))
        self.linear_model = nn.Sequential(
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10))

    def feedforward(self, x):
        x = self.conv_model(x)
        y = torch.flatten(x, 1)
        return self.linear_model(y)




