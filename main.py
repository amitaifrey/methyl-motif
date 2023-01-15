import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 30


class MethylCNN(nn.Module):
    def __init__(self):
        super(MethylCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=6, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (K - 4 + 1) * (K - 4 + 1), 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * (K - 4 + 1) * (K - 4 + 1))
        x = F.relu(self.fc1(x))
        return x


def main():
    net = MethylCNN()
    print(net)


if __name__ == "__main__":
    main()
