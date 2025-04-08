import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

class DataAugmentor:
    def __init__(self, config, data_loader):
        self.cfg = config['augmentation']
        self.data_loader = data_loader

    def apply(self, image_magnitude, max_doppler, labels):
        # image_magnitude: (B=16, C=10, H, W)
        x = image_magnitude.clone()

        # Unsqueeze to (B, C, 1, H, W) for augmentation
        x = x.unsqueeze(2)  # (B, C, 1, H, W)

        # --- Flip / Rotation ---
        if self.cfg['flip_rotation']['enable'] and random.random() < self.cfg['flip_rotation']['prob_apply']:
            op = random.choice(['hflip', 'vflip', 'rot90'])
            if op == 'rot90':
                k = random.choice([1, 2, 3])
                x = torch.rot90(x, k=k, dims=[-1, -2])
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

        # --- Mixup ---
        if self.cfg['mixup']['enable'] and random.random() < self.cfg['mixup']['prob_apply']:
            batch_size = x.size(0)
            
            nb_maps = random.randint(self.cfg['mixup'].get('min_maps', 1), self.cfg['mixup'].get('max_maps', 3))

            # Initialisation accumulations
            total_images = torch.zeros_like(x)
            total_labels = torch.zeros_like(labels)
            total_max = torch.zeros_like(max_doppler)

            mean_label = self.data_loader.dataset.mean_label
            std_label = self.data_loader.dataset.std_label
            mean_max = self.data_loader.dataset.mean_max_values
            std_max = self.data_loader.dataset.std_max_values

            for _ in range(nb_maps):
                # Tirer un indice aléatoire pour chaque image du batch
                indices = torch.randint(0, len(self.data_loader), (batch_size,))
                y_images, y_labels, y_max = [], [], []

                for idx in indices:
                    y_image, max_val, label = self.data_loader.dataset[idx]
                    y_images.append(y_image.to(x.device))
                    y_labels.append(label.to(x.device))
                    y_max.append(max_val.to(x.device))

                y_images = torch.stack(y_images).unsqueeze(2)  # (B, C, 1, H, W)
                y_labels = torch.stack(y_labels)
                y_max = torch.stack(y_max)

                total_images += y_images

                # Déstandardiser puis accumuler les labels
                total_labels += y_labels * std_label + mean_label

                # Déstandardiser puis accumuler les max
                total_max += y_max * std_max + mean_max

            # Appliquer le mixup : additionner les cartes
            x = x + total_images

            # Labels : ajouter puis re-standardiser
            labels = (labels * std_label + mean_label) + total_labels
            labels = (labels - mean_label) / std_label

            # Max doppler : moyenne puis re-standardiser
            max_val = max_doppler * std_max + mean_max
            max_val = (max_val + total_max) / (nb_maps + 1)
            max_doppler = (max_val - mean_max) / std_max


        # Retour à (B, C, H, W)
        x = x.squeeze(2)

        return x, max_doppler, labels