# Shamya Haria
# CS5330 - Pattern Recognition and Computer Vision
# Training loop for the Siamese network

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import SiameseNet, ContrastiveLoss
from .dataset import HandwritingPairDataset
from tqdm import tqdm


def train(config):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"using device: {device}")

    train_ds = HandwritingPairDataset(
        config['iam_root'], config['words_txt'],
        num_pairs=config['num_pairs'], split='train'
    )
    val_ds = HandwritingPairDataset(
        config['iam_root'], config['words_txt'],
        num_pairs=config['num_pairs'] // 5, split='val'
    )

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False, num_workers=2)

    model     = SiameseNet(emb_dim=256).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # halve lr every 5 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float('inf')

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        train_loss = 0.0

        for img1, img2, label in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            e1, e2, _ = model(img1, img2)
            loss = criterion(e1, e2, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                e1, e2, _ = model(img1, img2)
                val_loss += criterion(e1, e2, label).item()
        val_loss /= len(val_loader)

        scheduler.step()
        print(f"epoch {epoch}/{config['epochs']}  train loss: {train_loss:.4f}  val loss: {val_loss:.4f}")

        # keep the best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], 'best_model.pth'))

    print("training done. best val loss:", round(best_val_loss, 4))


if __name__ == '__main__':
    config = {
        'iam_root':       'data/iam',
        'words_txt':      'data/iam/words.txt',
        'num_pairs':      50000,
        'batch_size':     32,
        'epochs':         20,
        'lr':             1e-4,
        'checkpoint_dir': 'outputs/checkpoints'
    }
    train(config)