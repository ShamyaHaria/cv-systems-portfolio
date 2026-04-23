# Shamya Haria
# CS5330 - Pattern Recognition and Computer Vision
# Evaluate trained model on self-collected handwriting samples

import os
import torch
import itertools
from PIL import Image
from torchvision import transforms

from .model import SiameseNet


def load_patch(path, img_size=105):
    tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return tf(Image.open(path))


def evaluate_self_collected(processed_dir, model_path, threshold=0.4402):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    model = SiameseNet(emb_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # load one patch per person (use passage)
    persons = sorted(p for p in os.listdir(processed_dir) if not p.startswith('.'))
    patches = {}
    for p in persons:
        path = os.path.join(processed_dir, p, 'passage.png')
        if os.path.exists(path):
            patches[p] = load_patch(path).to(device)

    correct = 0
    total   = 0
    results = []

    with torch.no_grad():
        # same-writer pairs: passage vs words from same person
        for p in persons:
            p_path = os.path.join(processed_dir, p, 'passage.png')
            w_path = os.path.join(processed_dir, p, 'words.png')
            if not (os.path.exists(p_path) and os.path.exists(w_path)):
                continue
            img1 = load_patch(p_path).unsqueeze(0).to(device)
            img2 = load_patch(w_path).unsqueeze(0).to(device)
            _, _, dist = model(img1, img2)
            pred = 1 if dist.item() <= threshold else 0
            correct += (pred == 1)
            total   += 1
            results.append((p, p, dist.item(), pred, 1))

        # different-writer pairs: passage from person A vs passage from person B
        for p1, p2 in itertools.combinations(persons, 2):
            img1 = patches[p1].unsqueeze(0)
            img2 = patches[p2].unsqueeze(0)
            _, _, dist = model(img1, img2)
            pred = 1 if dist.item() <= threshold else 0
            correct += (pred == 0)
            total   += 1
            results.append((p1, p2, dist.item(), pred, 0))

    acc = correct / total
    print(f"\nself-collected evaluation ({total} pairs)")
    print(f"accuracy: {acc * 100:.2f}%\n")
    print(f"{'pair':<30} {'distance':>10} {'predicted':>12} {'actual':>8} {'correct':>8}")
    print("-" * 72)
    for p1, p2, dist, pred, actual in results:
        label     = "same" if pred == 1 else "diff"
        gt        = "same" if actual == 1 else "diff"
        is_correct = "✓" if pred == actual else "✗"
        print(f"{p1} vs {p2:<15} {dist:>10.4f} {label:>12} {gt:>8} {is_correct:>8}")

    return acc


if __name__ == '__main__':
    evaluate_self_collected(
        'data/processed/self_collected',
        'outputs/checkpoints/best_model.pth'
    )