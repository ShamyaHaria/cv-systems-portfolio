# Shamya Haria, CS 5330 Project 5
# Task 5: hyperparameter experiment
# Testing how conv filter count, FC layer size, and activation function affect accuracy
# Date: April 5, 2026

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
import itertools
import json
import os
import time


class FlexNet(nn.Module):
    """Modified version of MyNetwork where I can change filter counts,
    FC size, and activation function for experimentation."""

    def __init__(self, n_filters1=10, n_filters2=20, fc_nodes=50, dropout_rate=0.5,
                 pool_size=2, activation='relu'):
        super(FlexNet, self).__init__()
        self.activation_name = activation
        self.conv1 = nn.Conv2d(1, n_filters1, kernel_size=5)
        self.conv2 = nn.Conv2d(n_filters1, n_filters2, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=dropout_rate)
        self.pool_size = pool_size

        fc_input = n_filters2 * 4 * 4  # after two 5x5 convs + 2x2 pools on 28x28
        self.fc1 = nn.Linear(fc_input, fc_nodes)
        self.fc2 = nn.Linear(fc_nodes, 10)

    def activate(self, x):
        if self.activation_name == 'relu':
            return F.relu(x)
        elif self.activation_name == 'elu':
            return F.elu(x)
        elif self.activation_name == 'leaky_relu':
            return F.leaky_relu(x)
        return F.relu(x)  # fallback

    def forward(self, x):
        x = self.activate(F.max_pool2d(self.conv1(x), self.pool_size))
        x = self.activate(F.max_pool2d(self.conv2_drop(self.conv2(x)), self.pool_size))
        x = x.view(x.size(0), -1)
        x = self.activate(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def load_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def quick_train_eval(model, train_loader, test_loader, n_epochs=3, lr=0.01):
    """Train for a few epochs and return final accuracy. Used for rapid grid search."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    # evaluate
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    elapsed = time.time() - t0
    accuracy = 100.0 * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    return accuracy, test_loss, elapsed


def run_experiment(train_loader, test_loader):
    os.makedirs('outputs', exist_ok=True)

    # My hypotheses before running:
    # 1) More conv1 filters should help up to a point, then overfitting kicks in
    # 2) Bigger FC layer = better accuracy but diminishing returns
    # 3) ELU might slightly beat ReLU because of smoother gradients
    print('\nHypotheses:')
    print('  - More filters in conv1 should improve accuracy (up to a point)')
    print('  - More FC nodes helps but with diminishing returns')
    print('  - ELU/LeakyReLU might slightly outperform ReLU')

    # --- Dimension 1: conv1 filter count ---
    print('\n[Dim 1] Conv1 filter count:')
    filter_options = [5, 10, 20, 32]
    dim1_results = []
    for nf in filter_options:
        torch.manual_seed(42)
        model = FlexNet(n_filters1=nf, n_filters2=20, fc_nodes=50,
                        dropout_rate=0.5, activation='relu')
        acc, loss, t = quick_train_eval(model, train_loader, test_loader, n_epochs=3)
        dim1_results.append({'n_filters1': nf, 'accuracy': acc, 'loss': loss, 'time': t})
        print(f'  filters={nf:2d}  acc={acc:.2f}%  loss={loss:.4f}  time={t:.1f}s')

    # --- Dimension 2: FC hidden layer size ---
    print('\n[Dim 2] FC hidden nodes:')
    fc_options = [20, 50, 100, 200]
    dim2_results = []
    for fc in fc_options:
        torch.manual_seed(42)
        model = FlexNet(n_filters1=10, n_filters2=20, fc_nodes=fc,
                        dropout_rate=0.5, activation='relu')
        acc, loss, t = quick_train_eval(model, train_loader, test_loader, n_epochs=3)
        dim2_results.append({'fc_nodes': fc, 'accuracy': acc, 'loss': loss, 'time': t})
        print(f'  fc_nodes={fc:3d}  acc={acc:.2f}%  loss={loss:.4f}  time={t:.1f}s')

    # --- Dimension 3: activation function ---
    print('\n[Dim 3] Activation function:')
    act_options = ['relu', 'elu', 'leaky_relu']
    dim3_results = []
    for act in act_options:
        torch.manual_seed(42)
        model = FlexNet(n_filters1=10, n_filters2=20, fc_nodes=50,
                        dropout_rate=0.5, activation=act)
        acc, loss, t = quick_train_eval(model, train_loader, test_loader, n_epochs=3)
        dim3_results.append({'activation': act, 'accuracy': acc, 'loss': loss, 'time': t})
        print(f'  activation={act:12s}  acc={acc:.2f}%  loss={loss:.4f}  time={t:.1f}s')

    return dim1_results, dim2_results, dim3_results


def plot_results(dim1, dim2, dim3):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # conv1 filters
    x1 = [r['n_filters1'] for r in dim1]
    y1 = [r['accuracy'] for r in dim1]
    axes[0].bar([str(x) for x in x1], y1, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Conv1 Filters')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Effect of Conv1 Filter Count', fontweight='bold')
    axes[0].set_ylim([min(y1) - 2, 100])
    for i, v in enumerate(y1):
        axes[0].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=10)

    # fc nodes
    x2 = [r['fc_nodes'] for r in dim2]
    y2 = [r['accuracy'] for r in dim2]
    axes[1].bar([str(x) for x in x2], y2, color='darkorange', edgecolor='black')
    axes[1].set_xlabel('FC Hidden Nodes')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Effect of FC Layer Size', fontweight='bold')
    axes[1].set_ylim([min(y2) - 2, 100])
    for i, v in enumerate(y2):
        axes[1].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=10)

    # activation
    x3 = [r['activation'] for r in dim3]
    y3 = [r['accuracy'] for r in dim3]
    axes[2].bar(x3, y3, color='mediumseagreen', edgecolor='black')
    axes[2].set_xlabel('Activation Function')
    axes[2].set_ylabel('Test Accuracy (%)')
    axes[2].set_title('Effect of Activation Function', fontweight='bold')
    axes[2].set_ylim([min(y3) - 2, 100])
    for i, v in enumerate(y3):
        axes[2].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=10)

    plt.suptitle('Hyperparameter Experiment (3 epochs each)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/experiment_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved experiment results plot')


def main():
    print('--- Task 5: Hyperparameter Experiment ---')
    torch.manual_seed(42)

    train_loader, test_loader = load_mnist(batch_size=64)
    dim1, dim2, dim3 = run_experiment(train_loader, test_loader)
    plot_results(dim1, dim2, dim3)

    # summarize best from each dimension
    best1 = max(dim1, key=lambda r: r['accuracy'])
    best2 = max(dim2, key=lambda r: r['accuracy'])
    best3 = max(dim3, key=lambda r: r['accuracy'])
    print(f'\nBest configs:')
    print(f'  filters:    {best1["n_filters1"]} -> {best1["accuracy"]:.2f}%')
    print(f'  fc_nodes:   {best2["fc_nodes"]} -> {best2["accuracy"]:.2f}%')
    print(f'  activation: {best3["activation"]} -> {best3["accuracy"]:.2f}%')


if __name__ == '__main__':
    main()
