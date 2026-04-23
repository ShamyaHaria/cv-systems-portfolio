# Shamya Haria
# CS 5330, Project 5 - Extension 2
# Looking at pre-trained ResNet18's first conv layer and comparing
# to our simple MNIST network
# 04/05/2026

import sys
import torch
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def load_resnet():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    print('Loaded pre-trained ResNet18')
    print(f'  conv1 weights: {model.conv1.weight.shape}')  # should be (64, 3, 7, 7)
    return model


def visualize_resnet_filters(model):
    """Show the first 16 (of 64) conv1 filters as RGB images."""
    os.makedirs('outputs', exist_ok=True)
    weights = model.conv1.weight.detach()  # (64, 3, 7, 7)

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for i, ax in enumerate(axes.flat):
        # normalize each filter to [0,1] so we can display as RGB
        filt = weights[i].permute(1, 2, 0).numpy()
        filt_min = filt.min()
        filt_max = filt.max()
        filt_norm = (filt - filt_min) / (filt_max - filt_min + 1e-8)
        ax.imshow(filt_norm)
        ax.set_title(f'Filter {i}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('ResNet18 conv1 Filters (first 16 of 64)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/extension_resnet_filters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved resnet filter vis')


def load_and_upscale_mnist():
    """Get first MNIST test image and resize to 224x224 for ResNet."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    img_tensor, label = test_dataset[0]
    img_np = img_tensor[0].numpy()
    # undo normalization
    img_raw = np.clip(img_np * 0.3081 + 0.1307, 0, 1)
    img_resized = cv2.resize(img_raw, (224, 224))
    return img_resized, label


def show_resnet_filter_effects(model, img_np, label):
    """Apply first 16 ResNet conv1 filters to the MNIST digit."""
    os.makedirs('outputs', exist_ok=True)
    weights = model.conv1.weight.detach()

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for i, ax in enumerate(axes.flat):
        # average the 3 RGB channels since our image is grayscale
        filt = weights[i].mean(dim=0).numpy()
        filtered = cv2.filter2D(img_np.astype(np.float32), -1, filt)
        filtered = np.clip(filtered, 0, 1)
        ax.imshow(filtered, cmap='gray')
        ax.set_title(f'Filter {i}', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'ResNet conv1 effects on MNIST digit {label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/extension_resnet_filter_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved resnet filter effects')


def visualize_resnet_layer2(model):
    """Also look at the first residual block's conv1 weights for comparison."""
    os.makedirs('outputs', exist_ok=True)
    weights = model.layer1[0].conv1.weight.detach()
    print(f'  layer1[0].conv1 weights: {weights.shape}')

    n_show = 8
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.axis('off')
            continue
        # average across input channels
        filt = weights[i].mean(dim=0).numpy()
        im = ax.imshow(filt, cmap='RdBu_r')
        ax.set_title(f'Filter {i}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('ResNet18 layer1[0].conv1 Filters (8 of 64)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/extension_resnet_layer2_filters.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved layer1 filter vis')


def main():
    print('--- Extension 2: ResNet18 Filter Analysis ---')
    model = load_resnet()
    visualize_resnet_filters(model)
    visualize_resnet_layer2(model)

    img_np, label = load_and_upscale_mnist()
    show_resnet_filter_effects(model, img_np, label)


if __name__ == '__main__':
    main()
