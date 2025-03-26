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
            mode = self.cfg['flip_rotation'].get('mode', 'flip')
            if mode == 'flip':
                img = TF.hflip(img) if random.random() < 0.5 else TF.vflip(img)
            elif mode == 'flip90':
                img = torch.rot90(img, k=random.choice([1,2,3]), dims=[1,2])
            elif mode == 'rotate15':
                angle = random.uniform(-15, 15)
                # Appliquer la rotation image par image si batch
                if img.ndim == 4:
                    img = torch.stack([
                        TF.rotate(im, angle=angle,
                                  interpolation=TF.InterpolationMode.BILINEAR,
                                  expand=False, fill=0.0)
                        for im in img
                    ])
                else:
                    img = TF.rotate(img, angle=angle,
                                    interpolation=TF.InterpolationMode.BILINEAR,
                                    expand=False, fill=0.0)

        # --- Brightness / Contrast ---
        if self.cfg['color']['enable'] and random.random() < self.cfg['color']['prob_apply']:
            bmin, bmax = self.cfg['color']['brightness']
            cmin, cmax = self.cfg['color']['contrast']
            img = TF.adjust_brightness(img, brightness_factor=random.uniform(bmin, bmax))
            img = TF.adjust_contrast(img, contrast_factor=random.uniform(cmin, cmax))

        # --- Noise / Blur ---
        if self.cfg['noise_blur']['enable'] and random.random() < self.cfg['noise_blur']['prob_apply']:
            if random.random() < self.cfg['noise_blur']['prob_noise_vs_blur']:
                std = self.cfg['noise_blur']['gaussian_noise_std']
                noise = torch.randn_like(img) * std
                img = img + noise
            else:
                sigma = self.cfg['noise_blur']['blur_radius']
                kernel_size = 3 if sigma <= 1 else 5
                blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
                img = blur(img)

        return img
