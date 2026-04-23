# Shamya Haria
# CS5330 - Pattern Recognition and Computer Vision
# Siamese network definition + contrastive loss

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class EmbeddingNet(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        # each block halves spatial size, 105 -> 52 -> 26 -> 13 -> 6
        self.cnn = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, emb_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        # normalize so euclidean distance actually means something
        return F.normalize(x, p=2, dim=1)


class SiameseNet(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.backbone = EmbeddingNet(emb_dim)

    def forward(self, x1, x2):
        e1 = self.backbone(x1)
        e2 = self.backbone(x2)
        dist = F.pairwise_distance(e1, e2)
        return e1, e2, dist

    def get_embedding(self, x):
        return self.backbone(x)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, label):
        dist = F.pairwise_distance(e1, e2)
        # 1 = same writer (pull together), 0 = different (push apart)
        loss = label * dist.pow(2) + (1 - label) * F.relu(self.margin - dist).pow(2)
        return loss.mean()


if __name__ == '__main__':
    net = SiameseNet()
    x1 = torch.randn(4, 1, 105, 105)
    x2 = torch.randn(4, 1, 105, 105)
    e1, e2, d = net(x1, x2)
    print(e1.shape, d)