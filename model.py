import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
import random


class CNN(nn.Module):
    def __init__(self, dimensions=40, num_classes=3, input_length=1014):
        """
        A character-level CNN similar in spirit to Zhang et al. (2015).

        dimensions: embedding size (number of characters, e.g., 40)
        num_classes: number of sentiment classes (3: pos/neu/neg)
        input_length: sequence length (1014 chars)
        """
        super().__init__()

        # Convolution and pooling stack
        self.conv1 = nn.Conv1d(dimensions, 256, kernel_size=7)
        self.pool1 = nn.MaxPool1d(3)

        self.conv2 = nn.Conv1d(256, 256, kernel_size=7)
        self.pool2 = nn.MaxPool1d(3)

        self.conv3 = nn.Conv1d(256, 256, kernel_size=3)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3)
        self.pool3 = nn.MaxPool1d(3)

        # Figure out the size after convs/pools automatically
        with torch.no_grad():
            dummy = torch.zeros(1, dimensions, input_length)
            out = self._forward_conv(dummy)
            conv_out_dim = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(conv_out_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

    def _forward_conv(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(F.relu(self.conv6(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # logits
        return x