# ForgeryGuard — Handwriting Forgery Detection using Siamese Networks

## CS5330 Pattern Recognition and Computer Vision — Final Project
**Shamya Haria**

---

## Project Description
A deep learning system that compares two handwriting samples and determines whether they were written by the same person. Built using a Siamese Network trained with contrastive loss on the IAM Handwriting Database, supplemented with self-collected samples from 6 individuals.

---

## Results
| Evaluation                      | Accuracy |  EER   |
|---------------------------------|----------|--------|
| IAM val set                     | 55.16%   | 44.84% |
| Self-collected (unseen writers) | 71.43%   | 20.00% |

---

## Project Structure
ForgeryGuard/
├── data/
│   ├── iam/                  ← IAM Handwriting Database
│   ├── self_collected/       ← handwriting samples from 6 people
│   └── processed/            ← preprocessed patches
├── src/
│   ├── model.py              ← Siamese network + contrastive loss
│   ├── dataset.py            ← pair generation from IAM
│   ├── train.py              ← training loop
│   ├── evaluate.py           ← EER + accuracy on IAM val set
│   ├── evaluate_self_collected.py  ← evaluation on self-collected samples
│   ├── gradcam.py            ← Grad-CAM visualisation
│   └── selfcollected.py      ← preprocessing for self-collected images
├── outputs/
│   ├── checkpoints/          ← saved model weights
│   ├── gradcam/              ← Grad-CAM output images
│   └── results/
├── main.py
└── requirements.txt

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data
Download the IAM Handwriting Database from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database. Download `words.tgz` and `ascii.tgz` and extract both into `data/iam/`.

```bash
tar -xzf words.tgz -C data/iam/
tar -xzf ascii.tgz -C data/iam/
```

---

## Usage

```bash
# train
python3 main.py train

# evaluate on IAM val set
python3 main.py evaluate

# grad-cam visualisation on a pair of images
python3 main.py gradcam --img1 path/to/img1.png --img2 path/to/img2.png

# evaluate on self-collected samples
python3 -m src.evaluate_self_collected
```

---

## Demo Video
https://youtu.be/phX7P3hl0Hw

This Project was done Solo.