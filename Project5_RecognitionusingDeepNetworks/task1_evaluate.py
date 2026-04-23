# Shamya Haria, CS 5330
# Project 5 - Task 1b: evaluate the trained model on first 10 test examples
# 04/05/2026

import sys
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from task1_train import MyNetwork


def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    # batch of 10 so we get exactly the first 10
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=False
    )
    return test_loader


def run_first_ten(model, test_loader):
    """Run model on first 10 examples, print predictions vs labels."""
    examples = enumerate(test_loader)
    _, (data, targets) = next(examples)
    data_10 = data[:10]
    targets_10 = targets[:10]

    with torch.no_grad():
        outputs = model(data_10)

    print('\nFirst 10 test predictions:')
    for i in range(10):
        out_vals = outputs[i].numpy()
        pred = out_vals.argmax()
        label = targets_10[i].item()
        vals_str = '  '.join([f'{v:.2f}' for v in out_vals])
        status = 'correct' if pred == label else 'WRONG'
        print(f'  [{i}] label={label} pred={pred} ({status})  outputs: [{vals_str}]')

    return data_10, targets_10, outputs


def plot_first_nine(data, targets, outputs):
    os.makedirs('outputs', exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        pred = outputs[i].argmax().item()
        label = targets[i].item()
        color = 'green' if pred == label else 'red'
        ax.imshow(data[i][0], cmap='gray')
        ax.set_title(f'Pred: {pred}  (Label: {label})', fontsize=11,
                     color=color, fontweight='bold')
        ax.axis('off')

    plt.suptitle('First 9 MNIST Test Digits with Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/first_nine_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved prediction plot to outputs/first_nine_predictions.png')


def main():
    print('--- Task 1b: Evaluate on first 10 test examples ---')

    model = MyNetwork()
    model.load_state_dict(torch.load('outputs/mnist_model.pth', map_location='cpu'))
    model.eval()
    print('Loaded model from outputs/mnist_model.pth')

    test_loader = load_test_data()
    data_10, targets_10, outputs = run_first_ten(model, test_loader)
    plot_first_nine(data_10, targets_10, outputs)


if __name__ == '__main__':
    main()
