# Shamya Haria
# CS 5330, Project 5
# Task 3 - transfer learning, fine-tune MNIST network on Greek letters
# April 5 2026

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
import glob
from PIL import Image, ImageDraw

from task1_train import MyNetwork


class GreekTransform:
    """Convert the 133x133 RGB greek letter images to 28x28 grayscale
    that matches MNIST format. Based on the procedure from the assignment."""
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def load_greek_data(training_set_path, batch_size=5):
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                GreekTransform(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size,
        shuffle=True
    )
    return greek_train


def build_greek_model(mnist_path='outputs/mnist_model.pth'):
    """Load pretrained MNIST model, freeze everything, replace last layer for 3 classes."""
    model = MyNetwork()
    model.load_state_dict(torch.load(mnist_path, map_location='cpu'))

    # freeze conv layers and fc1 - only train the new last layer
    for param in model.parameters():
        param.requires_grad = False

    # replace fc2: was 50->10 (digits), now 50->3 (alpha, beta, gamma)
    model.fc2 = nn.Linear(50, 3)
    print('Modified network (last layer replaced):')
    print(model)
    return model


def train_greek(model, greek_loader, epochs=50):
    # only optimize the new fc2 layer
    optimizer = optim.Adam(model.fc2.parameters(), lr=0.001)
    train_losses = []
    train_counter = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for data, target in greek_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        avg_loss = epoch_loss / len(greek_loader)
        acc = 100.0 * correct / total
        train_losses.append(avg_loss)
        train_counter.append(epoch)

        if epoch % 5 == 0 or epoch == 1:
            print(f'  epoch {epoch:3d}  loss: {avg_loss:.4f}  accuracy: {correct}/{total} ({acc:.1f}%)')

        # stop early if we hit 100%
        if acc >= 100.0:
            print(f'  reached 100% at epoch {epoch}, stopping')
            break

    return train_losses, train_counter, epoch


def plot_greek_loss(train_losses, train_counter):
    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_counter, train_losses, color='purple', linewidth=2, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('NLL Loss')
    plt.title('Greek Letter Transfer Learning - Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/greek_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved training loss plot')


def test_custom_greek(model, samples, class_names):
    """Test the model on my handwritten Greek letter photos."""
    os.makedirs('outputs', exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    model.eval()
    results = []
    print('\nCustom Greek letter results:')

    for true_label, name, fpath, pil_img in samples:
        tensor = transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
        pred = output.argmax(dim=1).item()
        status = 'correct' if pred == true_label else 'WRONG'
        print(f'  {name} -> predicted {class_names[pred]} ({status})')
        results.append((name, true_label, pred, pil_img))

    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axes_flat = list(axes.flat)

    for i, (name, true_label, pred, pil_img) in enumerate(results):
        ax = axes_flat[i]
        resized = pil_img.resize((128, 128))
        ax.imshow(resized)
        color = 'green' if pred == true_label else 'red'
        ax.set_title(f'True: {name}\nPred: {class_names[pred]}',
                     fontsize=11, color=color, fontweight='bold')
        ax.axis('off')

    # hide any empty subplot slots
    for j in range(i + 1, rows * cols):
        axes_flat[j].axis('off')

    plt.suptitle('Custom Greek Letter Recognition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/custom_greek_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved custom greek results plot')


def main():
    print('--- Task 3: Transfer Learning on Greek Letters ---')

    greek_data_path = 'greek_train'
    if not os.path.exists(greek_data_path):
        print(f'ERROR: Greek data not found at {greek_data_path}/')
        print('Need the greek_train folder with alpha/, beta/, gamma/ subdirectories.')
        return

    model = build_greek_model()
    greek_loader = load_greek_data(greek_data_path)
    class_names = greek_loader.dataset.classes
    print(f'\nClasses: {class_names}')
    print(f'Training samples: {len(greek_loader.dataset)}')

    print('\nTraining...')
    train_losses, train_counter, final_epoch = train_greek(model, greek_loader, epochs=50)
    plot_greek_loss(train_losses, train_counter)

    torch.save(model.state_dict(), 'outputs/greek_model.pth')
    print('Saved greek model')

    # test on my handwritten Greek letter photos
    my_greek_path = 'my_greek'
    label_map = {'alpha': 0, 'beta': 1, 'gamma': 2}
    samples = []
    for name, label_idx in label_map.items():
        for ext in ['jpg', 'jpeg', 'png']:
            for fpath in sorted(glob.glob(f'{my_greek_path}/my_{name}*.{ext}')):
                pil_img = Image.open(fpath).convert('RGB')
                samples.append((label_idx, name, fpath, pil_img))

    if samples:
        print(f'Found {len(samples)} custom Greek samples')
        test_custom_greek(model, samples, class_names)
    else:
        print('No custom Greek samples found in my_greek/')


if __name__ == '__main__':
    main()
