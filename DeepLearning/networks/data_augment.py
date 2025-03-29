import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

class DataAugmentor:
    def __init__(self, config):
        self.cfg = config['augmentation']

    def apply(self, x):
        # x: (B, 10, H, W)
        B, C, H, W = x.shape
        x = x.clone()

        # --- Préparer les opérations communes ---
        ops = []

        # --- Flip / Rotation ---
        if self.cfg['flip_rotation']['enable'] and random.random() < self.cfg['flip_rotation']['prob_apply']:
            op = random.choice(['hflip', 'vflip', 'rot90'])
            if op == 'rot90':
                k = random.choice([1, 2, 3])
                ops.append(lambda img: torch.rot90(img, k=k, dims=[-2, -1]))
            elif op == 'hflip':
                ops.append(TF.hflip)
            elif op == 'vflip':
                ops.append(TF.vflip)

        # --- Brightness / Contrast ---
        # On crée les facteurs à l’avance pour les réutiliser
        if self.cfg['color']['enable'] and random.random() < self.cfg['color']['prob_apply']:
            b = random.uniform(*self.cfg['color']['brightness'])
            c = random.uniform(*self.cfg['color']['contrast'])
            ops.append(lambda img: TF.adjust_brightness(img, b))
            ops.append(lambda img: TF.adjust_contrast(img, c))

        # --- Noise / Blur ---
        if self.cfg['noise_blur']['enable'] and random.random() < self.cfg['noise_blur']['prob_apply']:
            if random.random() < self.cfg['noise_blur']['prob_noise_vs_blur']:
                std = random.uniform(*self.cfg['noise_blur']['gaussian_noise_std'])
                ops.append(lambda img: img + torch.randn_like(img) * std)
            else:
                sigma = random.uniform(*self.cfg['noise_blur']['blur_std'])
                k = max(3, int(2 * round(sigma * 2) + 1))
                blur = T.GaussianBlur(kernel_size=k, sigma=sigma)
                ops.append(blur)

        # --- Appliquer à chaque frame (canal) séparément, batch par batch ---
        for b in range(B):
            for c in range(C):
                img = x[b, c].unsqueeze(0)  # (1, H, W)
                for op in ops:
                    img = op(img)
                x[b, c] = img.squeeze(0)

        return x