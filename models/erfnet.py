import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSample(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.conv = nn.Conv2d(inp, out - inp, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out, eps=1e-3)

    def forward(self, inp):
        out = torch.cat([self.conv(inp), self.pool(inp)], dim=1)
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

    def forward(self, inp):
        out = self.conv3x1_1(inp)
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
        return F.relu(out+inp)  # residual connection (identity)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownSample(3, 16)
        self.layers = nn.ModuleList()
        self.layers.append(DownSample(16, 64))
        for l in range(5):
            self.layers.append(NonBottleneck1d(64, 0.03, 1))
        self.layers.append(DownSample(64, 128))
        for l in range(2):
            self.layers.append(NonBottleneck1d(128, 0.3, 2))
            self.layers.append(NonBottleneck1d(128, 0.3, 4))
            self.layers.append(NonBottleneck1d(128, 0.3, 8))
            self.layers.append(NonBottleneck1d(128, 0.3, 16))
        self.out_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, inp, predict=False):
        out = self.initial_block(inp)
        for layer in self.layers:
            out = layer(out)
        if predict:
            out = self.out_conv(out)
        return out


class UpSample(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(inp, out, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out, eps=1e-3)

    def forward(self, inp):
        out = self.conv_t(inp)
        out = self.bn(out)
        return F.relu(out)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UpSample(128, 64))
        self.layers.append(NonBottleneck1d(64, 0, 1))
        self.layers.append(NonBottleneck1d(64, 0, 1))
        self.layers.append(UpSample(64, 16))
        self.layers.append(NonBottleneck1d(16, 0, 1))
        self.layers.append(NonBottleneck1d(16, 0, 1))
        self.out_conv_t = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, inp):
        out = inp
        for layer in self.layers:
            out = layer(out)
        out = self.out_conv_t(out)
        return out


class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super().__init__()
        if encoder is None:
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, inp, encode_only=False):
        if encode_only:
            return self.encoder.forward(inp, predict=True)
        else:
            out = self.encoder(inp)
            return self.decoder.forward(out)
