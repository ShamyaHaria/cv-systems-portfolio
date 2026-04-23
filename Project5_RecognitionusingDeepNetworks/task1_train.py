# Shamya Haria
# CS 5330 - Project 5
# Build and train a CNN on MNIST
# April 5, 2026

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
import os


class MyNetwork(nn.Module):
    """CNN for MNIST digit recognition - architecture from the assignment spec."""
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)  # 320 = 20 * 4 * 4 after two conv+pool layers
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # conv1 -> maxpool -> relu
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # conv2 -> dropout -> maxpool -> relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)   # flatten
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    # don't shuffle test set so we always get same first examples
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def plot_first_six(test_loader):
    """Show the first 6 digits from test set to verify data loading works."""
    os.makedirs('outputs', exist_ok=True)
    examples = enumerate(test_loader)
    _, (data, targets) = next(examples)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i][0], cmap='gray')
        ax.set_title(f'Label: {targets[i].item()}', fontsize=13)
        ax.axis('off')
    plt.suptitle('First 6 MNIST Test Set Digits', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/first_six_test_digits.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved first six test digits plot')


def train_epoch(model, train_loader, optimizer, epoch, train_losses, train_counter):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # log every 100 batches
        if batch_idx % 100 == 0:
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * len(data)) + ((epoch - 1) * len(train_loader.dataset))
            )


def test_epoch(model, test_loader, test_losses):
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
    print(f'  Test loss: {test_loss:.4f}, accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)')
    return acc


def train_network(model, train_loader, test_loader, num_epochs=5, lr=0.01, momentum=0.5):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []

    # test before any training to get baseline
    print('Before training:')
    test_epoch(model, test_loader, test_losses)
    test_counter.append(0)

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}:')
        train_epoch(model, train_loader, optimizer, epoch, train_losses, train_counter)
        test_epoch(model, test_loader, test_losses)
        test_counter.append(epoch * len(train_loader.dataset))

    return train_losses, train_counter, test_losses, test_counter


def plot_training_curve(train_losses, train_counter, test_losses, test_counter):
    plt.figure(figsize=(10, 6))
    plt.plot(train_counter, train_losses, color='blue', linewidth=1.5, label='Train loss')
    plt.scatter(test_counter, test_losses, color='red', s=60, zorder=5, label='Test loss')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.title('MNIST CNN Training and Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved training curve')


def main():
    torch.manual_seed(42)

    print('--- Task 1: Build and Train MNIST CNN ---')

    train_loader, test_loader = load_data(batch_size=64)
    print(f'Training set: {len(train_loader.dataset)} images')
    print(f'Test set: {len(test_loader.dataset)} images')

    plot_first_six(test_loader)

    model = MyNetwork()
    print('\nNetwork:')
    print(model)

    # train for 5 epochs (seems to converge pretty well by then)
    print('\nTraining...')
    train_losses, train_counter, test_losses, test_counter = train_network(
        model, train_loader, test_loader, num_epochs=5
    )

    plot_training_curve(train_losses, train_counter, test_losses, test_counter)

    # save model weights
    os.makedirs('outputs', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/mnist_model.pth')
    print('Model saved to outputs/mnist_model.pth')


if __name__ == '__main__':
    main()
