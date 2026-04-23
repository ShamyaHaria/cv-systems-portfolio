# Shamya Haria
# CS 5330 Project 5 - Task 1c
# Test network on my own handwritten digit photos
# Date: 4/5/2026

import sys
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image

from task1_train import MyNetwork

# MNIST normalization constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def preprocess_image(pil_img):
    """Convert a photo of a handwritten digit to match MNIST input format.
    
    Need to: resize to 28x28, invert colors (my photos are dark on white,
    but MNIST is white on black), then normalize with MNIST stats.
    """
    img = pil_img.convert('L').resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # invert: my photos have black digits on white paper, MNIST is opposite
    arr = 255.0 - arr
    arr = arr / 255.0
    arr = (arr - MNIST_MEAN) / MNIST_STD
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def load_real_digits(custom_dir='outputs/custom_digits'):
    digit_images = []
    for d in range(10):
        matches = glob.glob(f'{custom_dir}/digit_{d}.*')
        if matches:
            pil_img = Image.open(matches[0]).convert('L')
            digit_images.append((d, pil_img))
            print(f'  loaded digit_{d} from {matches[0]}')
        else:
            print(f'  WARNING: digit_{d} not found in {custom_dir}/')
    return digit_images


def evaluate_custom_digits(model, digit_images):
    os.makedirs('outputs', exist_ok=True)
    results = []

    print('\nCustom digit results:')
    for true_label, pil_img in digit_images:
        tensor = preprocess_image(pil_img)
        with torch.no_grad():
            output = model(tensor)
        pred = output.argmax(dim=1).item()
        status = 'correct' if pred == true_label else 'WRONG'
        print(f'  digit {true_label} -> predicted {pred} ({status})')
        results.append((true_label, pred, pil_img))

    # plot results in 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for i, ax in enumerate(axes.flat):
        true_label, pred, pil_img = results[i]
        resized = pil_img.resize((28, 28), Image.LANCZOS)
        arr = np.array(resized)
        color = 'green' if pred == true_label else 'red'
        ax.imshow(arr, cmap='gray')
        ax.set_title(f'True: {true_label}\nPred: {pred}', fontsize=11,
                     color=color, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Network on Custom Handwritten Digits (0-9)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = 'outputs/custom_digits_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved plot to {save_path}')

    correct_count = sum(1 for tl, pred, _ in results if tl == pred)
    print(f'Accuracy on custom digits: {correct_count}/10 ({correct_count * 10}%)')


def main():
    print('--- Task 1c: Custom Handwritten Digits ---')

    # load saved model
    model = MyNetwork()
    model.load_state_dict(torch.load('outputs/mnist_model.pth', map_location='cpu'))
    model.eval()

    digit_images = load_real_digits()
    if not digit_images:
        print('No digit images found! Place digit_0.png ... digit_9.png in outputs/custom_digits/')
        return

    evaluate_custom_digits(model, digit_images)


if __name__ == '__main__':
    main()