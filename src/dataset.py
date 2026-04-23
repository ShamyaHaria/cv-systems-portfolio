# Shamya Haria
# CS5330 - Pattern Recognition and Computer Vision
# Dataset loading and pair generation for Siamese training

import os
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def parse_words_txt(words_txt_path):
    writer_words = defaultdict(list)

    with open(words_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if parts[1] != 'ok':
                continue

            word_id   = parts[0]  # e.g. a01-000u-00-00
            writer_id = word_id.split('-')[0]  # a01
            form_id   = '-'.join(word_id.split('-')[:2])  # a01-000u

            # path structure: a01/a01-000u/a01-000u-00-00.png
            img_path = os.path.join(writer_id, form_id, word_id + '.png')
            writer_words[writer_id].append(img_path)

    return writer_words


class HandwritingPairDataset(Dataset):
    def __init__(self, iam_root, words_txt, num_pairs=50000, img_size=105, split='train', split_ratio=0.8):
        self.iam_root = iam_root
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        writer_words = parse_words_txt(words_txt)
        # drop writers with too few samples, not enough to make good pairs
        writer_words = {w: imgs for w, imgs in writer_words.items() if len(imgs) >= 10}

        writers = sorted(writer_words.keys())
        cut     = int(len(writers) * split_ratio)
        writers = writers[:cut] if split == 'train' else writers[cut:]

        self.writer_words = {w: writer_words[w] for w in writers}
        self.writers      = writers
        self.pairs        = self._build_pairs(num_pairs)

    def _build_pairs(self, n):
        pairs = []
        for _ in range(n // 2):
            # same writer
            w      = random.choice(self.writers)
            i1, i2 = random.sample(self.writer_words[w], 2)
            pairs.append((i1, i2, 1))

            # different writers
            w1, w2 = random.sample(self.writers, 2)
            i1     = random.choice(self.writer_words[w1])
            i2     = random.choice(self.writer_words[w2])
            pairs.append((i1, i2, 0))

        random.shuffle(pairs)
        return pairs

    def _load(self, rel_path):
        full = os.path.join(self.iam_root, rel_path)
        try:
            img = Image.open(full).convert('RGB')
            return self.transform(img)
        except Exception:
            # some IAM files are corrupt, just return blank
            return torch.zeros(1, self.img_size, self.img_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        return self._load(p1), self._load(p2), torch.tensor(label, dtype=torch.float32)


if __name__ == '__main__':
    IAM_ROOT  = 'data/iam'
    WORDS_TXT = 'data/iam/words.txt'

    ds = HandwritingPairDataset(IAM_ROOT, WORDS_TXT, num_pairs=100, split='train')
    print(f"writers: {len(ds.writers)}, pairs: {len(ds)}")
    img1, img2, label = ds[0]
    print(f"img shape: {img1.shape}, label: {label}")