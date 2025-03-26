import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

class DataAugmentor:
    def __init__(self, config):
        self.cfg = config['augmentation']

    def apply(self, x):
        img = x.clone()

        # --- Flip / Rotation ---
        if self.cfg['flip_rotation']['enable'] and random.random() < self.cfg['flip_rotation']['prob_apply']:
            flip_ops = ['hflip', 'vflip', 'rot90']
            op = random.choice(flip_ops)

            if op == 'hflip':
                img = TF.hflip(img)
            elif op == 'vflip':
                img = TF.vflip(img)
            elif op == 'rot90':
                # Choix d'un nombre de rotations aléatoire (1, 2, ou 3 fois 90°)
                k = random.choice([1, 2, 3])
                img = torch.rot90(img, k=k, dims=[-2, -1])  # dernière dim = H, W

        # --- Brightness / Contrast ---
        if self.cfg['color']['enable'] and random.random() < self.cfg['color']['prob_apply']:
            bmin, bmax = self.cfg['color']['brightness']
            cmin, cmax = self.cfg['color']['contrast']
            img = TF.adjust_brightness(img, brightness_factor=random.uniform(bmin, bmax))
            img = TF.adjust_contrast(img, contrast_factor=random.uniform(cmin, cmax))

        # --- Noise / Blur ---
        if self.cfg['noise_blur']['enable'] and random.random() < self.cfg['noise_blur']['prob_apply']:
            if random.random() < self.cfg['noise_blur']['prob_noise_vs_blur']:
                gmin, gmax = self.cfg['noise_blur']['gaussian_noise_std']
                std = random.uniform(gmin, gmax)
                noise = torch.randn_like(img) * std
                img = img + noise
            else:
                bmin, bmax = self.cfg['noise_blur']['blur_std']
                sigma = random.uniform(bmin, bmax)
                kernel_size = max(3, int(2 * round(sigma * 2) + 1))
                blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
                img = blur(img)

        return img
