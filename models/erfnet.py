import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DownSampleBlock(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.conv = nn.Conv2d(input, output - input, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(output, eps=1e-3)

    def forward(self, input):
        out = torch.cat([self.conv(input), self.pool(input)], dim=1)
        out = self.bn(out)
        return F.relu(out)


class NonBottleneck1d(nn.Module):
    def __init__(self, channel, drop_prob, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(channel, channel, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(channel, channel, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(channel, eps=1e-3)
        self.conv3x1_2 = nn.Conv2d(channel, channel, (3, 1), stride=1, padding=(1*dilated, 0), bias=True)
        self.conv1x3_2 = nn.Conv2d(channel, channel, (1, 3), stride=1, padding=(0, 1*dilated), bias=True)
        self.bn2 = nn.BatchNorm2d(channel, eps=1e-3)
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, input):
        out = self.conv3x1_1(input)
        out = F.relu(out)
        out = self.conv1x3_1(out)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv3x1_2(out)
        out = F.relu(out)
        out = self.conv1x3_2(out)
        out = self.bn2(out)
        if self.dropout != 0:
            out = self.dropout(out)
        return F.relu(out+input)  # residual connection (identity)


