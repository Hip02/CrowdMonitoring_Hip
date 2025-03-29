import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

class DataAugmentor:
    def __init__(self, config):
        self.cfg = config['augmentation']

    def apply(self, x):
        img = x.clone()  # (C, H, W), où C = nb de frames concaténées
        frames = torch.unbind(img, dim=0)  # liste de (H, W), une par frame

        # On stocke les opérations à appliquer à toutes les frames
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

        # Appliquer toutes les opérations à chaque frame
        frames = [self._apply_ops(f, ops) for f in frames]
        return torch.stack(frames, dim=0)

    def _apply_ops(self, img, ops):
        for op in ops:
            img = op(img)
        return img