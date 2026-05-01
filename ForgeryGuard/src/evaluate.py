# Shamya Haria
# CS5330 - Pattern Recognition and Computer Vision
# Evaluation: EER and verification accuracy on held-out writers

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from tqdm import tqdm

from .model import SiameseNet
from .dataset import HandwritingPairDataset


def compute_eer(labels, distances):
    # flip distances to scores since ROC expects higher = more similar
    scores = -np.array(distances)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    # find where FPR and FNR cross
    idx    = np.argmin(np.abs(fpr - fnr))
    eer    = (fpr[idx] + fnr[idx]) / 2
    thresh = -thresholds[idx]
    return eer, thresh


def evaluate(config):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    model = SiameseNet(emb_dim=256).to(device)
    model.load_state_dict(torch.load(config['checkpoint'], map_location=device))
    model.eval()

    ds = HandwritingPairDataset(
        config['iam_root'], config['words_txt'],
        num_pairs=config['num_pairs'], split='val'
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    all_labels    = []
    all_distances = []

    with torch.no_grad():
        for img1, img2, label in tqdm(loader, desc='evaluating'):
            img1, img2 = img1.to(device), img2.to(device)
            _, _, dist = model(img1, img2)
            all_distances.extend(dist.cpu().numpy())
            all_labels.extend(label.numpy())

    eer, best_thresh = compute_eer(all_labels, all_distances)

    # threshold from EER point
    preds   = [1 if d <= best_thresh else 0 for d in all_distances]
    correct = sum(p == l for p, l in zip(preds, all_labels))
    acc     = correct / len(all_labels)

    print(f"EER             : {eer * 100:.2f}%")
    print(f"best threshold  : {best_thresh:.4f}")
    print(f"accuracy        : {acc * 100:.2f}%")
    return eer, acc


if __name__ == '__main__':
    config = {
        'iam_root':   'data/iam',
        'words_txt':  'data/iam/words.txt',
        'num_pairs':  2000,
        'checkpoint': 'outputs/checkpoints/best_model.pth',
    }
    evaluate(config)