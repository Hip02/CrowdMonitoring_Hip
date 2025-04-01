import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

class DataAugmentor:
    def __init__(self, config):
        self.cfg = config['augmentation']

    def apply(self, x):
        # x: (B, C=10, H, W)
        B, C, H, W = x.shape
        x = x.clone()

        # Fusionne B et C pour appliquer les opé en batch
        x = x.view(B * C, H, W).unsqueeze(1)  # (B*C, 1, H, W)

        # --- Flip / Rotation ---
        if self.cfg['flip_rotation']['enable'] and random.random() < self.cfg['flip_rotation']['prob_apply']:
            op = random.choice(['hflip', 'vflip', 'rot90'])
            if op == 'rot90':
                k = random.choice([1, 2, 3])
                x = torch.rot90(x, k=k, dims=[-2, -1])
            elif op == 'hflip':
                x = TF.hflip(x)
            elif op == 'vflip':
                x = TF.vflip(x)

        # --- Brightness / Contrast ---
        if self.cfg['color']['enable'] and random.random() < self.cfg['color']['prob_apply']:
            b = random.uniform(*self.cfg['color']['brightness'])
            c = random.uniform(*self.cfg['color']['contrast'])

            # Brightness: multiplication directe
            x = TF.adjust_brightness(x, b)
            x = TF.adjust_contrast(x, c)

        # --- Noise / Blur ---
        if self.cfg['noise_blur']['enable'] and random.random() < self.cfg['noise_blur']['prob_apply']:
            if random.random() < self.cfg['noise_blur']['prob_noise_vs_blur']:
                std = random.uniform(*self.cfg['noise_blur']['gaussian_noise_std'])
                noise = torch.randn_like(x) * std
                x = x + noise
            else:
                sigma = random.uniform(*self.cfg['noise_blur']['blur_std'])
                k = max(3, int(2 * round(sigma * 2) + 1))
                blur = T.GaussianBlur(kernel_size=k, sigma=sigma)
                x = blur(x)

        # Retour à (B, C, H, W)
        x = x.squeeze(1).view(B, C, H, W)
        return x