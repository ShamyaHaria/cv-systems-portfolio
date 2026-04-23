# Shamya Haria
# CS5330 - Pattern Recognition and Computer Vision
# Grad-CAM visualisation on the embedding network

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from .model import SiameseNet


class GradCAM:
    def __init__(self, model):
        self.model      = model
        self.gradient   = None
        self.activation = None

        # attach hooks to last conv layer
        target = model.backbone.cnn[-1].conv
        target.register_forward_hook(self._save_activation)
        target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activation = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradient = grad_out[0].detach()

    def generate(self, img_tensor):
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0)
        emb = self.model.get_embedding(img_tensor)

        self.model.zero_grad()
        # use embedding norm as scalar to backprop through
        emb.norm().backward()

        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activation).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()

        # scale to 0-1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def visualise_pair(img1_path, img2_path, model_path, out_path, img_size=105):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    model = SiameseNet(emb_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img1 = tf(Image.open(img1_path)).to(device)
    img2 = tf(Image.open(img2_path)).to(device)

    gcam = GradCAM(model)
    cam1 = gcam.generate(img1)
    cam2 = gcam.generate(img2)

    _, _, dist = model(img1.unsqueeze(0), img2.unsqueeze(0))
    dist = dist.item()

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0][0].imshow(img1.cpu().squeeze(), cmap='gray')
    axes[0][0].set_title('sample 1')
    axes[0][1].imshow(cam1, cmap='jet')
    axes[0][1].set_title('grad-cam 1')
    axes[1][0].imshow(img2.cpu().squeeze(), cmap='gray')
    axes[1][0].set_title('sample 2')
    axes[1][1].imshow(cam2, cmap='jet')
    axes[1][1].set_title('grad-cam 2')

    fig.suptitle(f'distance: {dist:.4f}')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"saved to {out_path}")


if __name__ == '__main__':
    visualise_pair(
        'data/iam/a01/a01-000u/a01-000u-00-00.png',
        'data/iam/a01/a01-000u/a01-000u-00-01.png',
        'outputs/checkpoints/best_model.pth',
        'outputs/gradcam/sample_pair.png'
    )