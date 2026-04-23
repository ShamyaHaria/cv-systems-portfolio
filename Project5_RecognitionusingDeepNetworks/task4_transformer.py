# Shamya Haria
# CS 5330 Project 5 - Task 4
# Vision Transformer approach for MNIST, comparing against the CNN from task 1
# 04/05/2026

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os

from task1_train import load_data


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding from 'Attention Is All You Need'."""
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class NetTransformer(nn.Module):
    """ViT-style network for MNIST. Splits 28x28 image into non-overlapping
    patches, embeds them, adds a CLS token, runs through transformer encoder,
    then classifies from the CLS output."""

    def __init__(self, patch_size=7, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, num_classes=10):
        super(NetTransformer, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # (28 / 7)^2 = 16 patches
        n_patches = (28 // patch_size) ** 2
        patch_dim = patch_size * patch_size  # 49 for 7x7, grayscale so 1 channel

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len=n_patches + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # classification head - same structure as the CNN (fc1 -> fc2)
        self.fc1 = nn.Linear(d_model, 50)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(50, num_classes)

    def extract_patches(self, x):
        B = x.size(0)
        p = self.patch_size
        # unfold extracts non-overlapping patches
        x = x.unfold(2, p, p).unfold(3, p, p)   # (B, 1, n_h, n_w, p, p)
        x = x.contiguous().view(B, -1, p * p)    # (B, n_patches, patch_dim)
        return x

    def forward(self, x):
        B = x.size(0)
        patches = self.extract_patches(x)
        tokens = self.patch_embed(patches)

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_enc(tokens)

        encoded = self.transformer_encoder(tokens)
        cls_out = encoded[:, 0, :]  # take the CLS token output

        x = F.relu(self.fc1(cls_out))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def train_one_epoch(model, train_loader, optimizer, epoch, train_losses, train_counter):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * len(data)) + ((epoch - 1) * len(train_loader.dataset))
            )


def evaluate(model, test_loader, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    acc = 100.0 * correct / len(test_loader.dataset)
    print(f'  test loss: {test_loss:.4f}, accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)')
    return acc


def plot_training(train_losses, train_counter, test_losses, test_counter):
    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_counter, train_losses, color='green', linewidth=1.5, label='Train loss')
    plt.scatter(test_counter, test_losses, color='orange', s=60, zorder=5, label='Test loss')
    plt.xlabel('Training examples seen')
    plt.ylabel('NLL loss')
    plt.title('Vision Transformer on MNIST')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/transformer_training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved transformer training curve')


def main():
    print('--- Task 4: Transformer Network ---')
    torch.manual_seed(42)

    model = NetTransformer(patch_size=7, d_model=64, nhead=4, num_layers=2,
                           dim_feedforward=128, dropout=0.1)
    print('Architecture:')
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params:,}')

    train_loader, test_loader = load_data(batch_size=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []
    n_epochs = 5

    # baseline before training
    evaluate(model, test_loader, test_losses)
    test_counter.append(0)

    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}:')
        train_one_epoch(model, train_loader, optimizer, epoch, train_losses, train_counter)
        evaluate(model, test_loader, test_losses)
        test_counter.append(epoch * len(train_loader.dataset))

    plot_training(train_losses, train_counter, test_losses, test_counter)

    torch.save(model.state_dict(), 'outputs/transformer_model.pth')
    print('Saved transformer model')


if __name__ == '__main__':
    main()
