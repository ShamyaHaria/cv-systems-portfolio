# Shamya Haria - CS 5330 Project 5
# Extension 1: replace learned conv1 filters with a fixed Gabor filter bank
# see if hand-crafted orientation filters work as well as learned ones
# 4/5/26

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
import numpy as np
import os

from task1_train import load_data


def create_gabor_filters(n_filters=10, ksize=5):
    """Generate Gabor filters at evenly spaced orientations.
    Each filter detects edges at a different angle."""
    filters = []
    thetas = np.linspace(0, np.pi, n_filters, endpoint=False)
    sigma = 1.5
    lambd = 5.0
    gamma = 0.5
    psi = 0

    for theta in thetas:
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        half = ksize // 2
        for y in range(-half, half + 1):
            for x in range(-half, half + 1):
                x_rot = x * np.cos(theta) + y * np.sin(theta)
                y_rot = -x * np.sin(theta) + y * np.cos(theta)
                gauss = np.exp(-(x_rot**2 + gamma**2 * y_rot**2) / (2 * sigma**2))
                sinusoid = np.cos(2 * np.pi * x_rot / lambd + psi)
                kernel[y + half][x + half] = gauss * sinusoid
        # zero-mean and unit variance
        kernel -= kernel.mean()
        if kernel.std() > 0:
            kernel /= kernel.std()
        filters.append(kernel)

    return filters


class GaborNet(nn.Module):
    """Same architecture as MyNetwork but conv1 uses fixed Gabor filters instead
    of learned ones. Only conv2, fc1, fc2 are trained."""

    def __init__(self, n_gabor=10):
        super(GaborNet, self).__init__()
        self.conv1 = nn.Conv2d(1, n_gabor, kernel_size=5, bias=False)

        # initialize with Gabor filters
        gabor_filters = create_gabor_filters(n_gabor, 5)
        with torch.no_grad():
            for i, filt in enumerate(gabor_filters):
                self.conv1.weight[i, 0] = torch.tensor(filt)
        # freeze - these don't get updated
        for param in self.conv1.parameters():
            param.requires_grad = False

        self.conv2 = nn.Conv2d(n_gabor, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def train_gabor(model, train_loader, test_loader, n_epochs=5):
    # only update parameters that require grad (everything except conv1)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01, momentum=0.5
    )
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []

    # baseline
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
    test_counter.append(0)
    print(f'Before training: loss={test_loss:.4f} acc={100*correct/len(test_loader.dataset):.2f}%')

    for epoch in range(1, n_epochs + 1):
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

        # eval after each epoch
        model.eval()
        tl = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                tl += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        tl /= len(test_loader.dataset)
        test_losses.append(tl)
        test_counter.append(epoch * len(train_loader.dataset))
        print(f'Epoch {epoch}: loss={tl:.4f} acc={100*correct/len(test_loader.dataset):.2f}%')

    return train_losses, train_counter, test_losses, test_counter


def plot_gabor_filters(model):
    os.makedirs('outputs', exist_ok=True)
    weights = model.conv1.weight
    n = weights.shape[0]
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes_flat = list(axes.flat)
    for i in range(n):
        ax = axes_flat[i]
        filt = weights[i, 0].detach().numpy()
        ax.imshow(filt, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'Gabor {i} ({i * 18}°)', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, rows * cols):
        axes_flat[j].axis('off')

    plt.suptitle('Gabor Filter Bank (Fixed First Layer)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/extension_gabor_filters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved gabor filter visualization')


def plot_gabor_curve(train_losses, train_counter, test_losses, test_counter):
    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_counter, train_losses, color='darkred', linewidth=1.5, label='Train loss')
    plt.scatter(test_counter, test_losses, color='gold', s=60, zorder=5, label='Test loss')
    plt.xlabel('Training examples seen')
    plt.ylabel('NLL loss')
    plt.title('MNIST with Fixed Gabor Filters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/extension_gabor_training.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print('--- Extension 1: Gabor Filter Bank ---')
    torch.manual_seed(42)

    train_loader, test_loader = load_data(batch_size=64)
    model = GaborNet(n_gabor=10)
    print(model)

    plot_gabor_filters(model)

    print('\nTraining (conv1 frozen)...')
    tl, tc, vl, vc = train_gabor(model, train_loader, test_loader, n_epochs=5)
    plot_gabor_curve(tl, tc, vl, vc)

    torch.save(model.state_dict(), 'outputs/gabor_model.pth')
    print('Done, saved gabor model')


if __name__ == '__main__':
    main()
