# Shamya Haria - CS 5330
# Task 2: examine the trained network filters and visualize their effects
# 04/05/2026

import sys
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from task1_train import MyNetwork


def load_model(path='outputs/mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def load_first_training_image():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    img_tensor, label = train_dataset[0]
    return img_tensor, label


def analyze_first_layer(model):
    """Visualize the 10 learned conv1 filters in a grid."""
    os.makedirs('outputs', exist_ok=True)
    weights = model.conv1.weight
    print(f'conv1 weight shape: {weights.shape}')

    with torch.no_grad():
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for i in range(10):
            ax = axes[i // 4][i % 4]
            filt = weights[i, 0].numpy()
            print(f'\nFilter {i}:\n{np.round(filt, 4)}')
            im = ax.imshow(filt, cmap='viridis')
            ax.set_title(f'Filter {i}', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # blank out the last 2 spots (only 10 filters, grid has 12)
        for j in [10, 11]:
            axes[j // 4][j % 4].axis('off')

        plt.suptitle('Conv1 Learned Filters (5x5)', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/conv1_filters.png', dpi=150, bbox_inches='tight')
        plt.close()
    print('\nSaved conv1 filter visualization')


def show_filter_effects(model, img_tensor, label):
    """Apply each conv1 filter to the first training image using cv2.filter2D
    and show filter + result side by side."""
    os.makedirs('outputs', exist_ok=True)

    img_np = img_tensor[0].numpy()
    # denormalize to get original pixel values back
    img_raw = img_np * 0.3081 + 0.1307
    img_raw = np.clip(img_raw, 0, 1).astype(np.float32)

    with torch.no_grad():
        weights = model.conv1.weight
        fig, axes = plt.subplots(5, 4, figsize=(14, 18))

        for i in range(10):
            filt = weights[i, 0].numpy()
            filtered = cv2.filter2D(img_raw, -1, filt)
            filtered = np.clip(filtered, 0, 1)

            row = i // 2
            col_filt = (i % 2) * 2
            col_result = col_filt + 1

            ax_f = axes[row][col_filt]
            ax_f.imshow(filt, cmap='viridis')
            ax_f.set_title(f'Filter {i}', fontsize=10, fontweight='bold')
            ax_f.set_xticks([])
            ax_f.set_yticks([])

            ax_r = axes[row][col_result]
            ax_r.imshow(filtered, cmap='gray')
            ax_r.set_title(f'Filter {i} applied', fontsize=10)
            ax_r.set_xticks([])
            ax_r.set_yticks([])

        plt.suptitle(f'Filter Effects on First Training Image (Label: {label})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/conv1_filter_effects.png', dpi=150, bbox_inches='tight')
        plt.close()
    print('Saved filter effects visualization')


def main():
    print('--- Task 2: Examine Network ---')

    model = load_model()

    # print model architecture
    print('\nModel structure:')
    print(model)
    print('\nLayers:')
    for name, module in model.named_modules():
        if name:
            print(f'  {name}: {module}')

    print('\nAnalyzing conv1 filters...')
    analyze_first_layer(model)

    print('\nApplying filters to first training image...')
    img_tensor, label = load_first_training_image()
    show_filter_effects(model, img_tensor, label)


if __name__ == '__main__':
    main()
