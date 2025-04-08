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
            # Récupérer un batch de données complet
            batch_size = x.size(0)  # B (taille du batch)

            # Sélectionner des indices aléatoires différents pour chaque image du batch
            indices = torch.randint(0, len(self.data_loader), (batch_size,))  # indices pour chaque sample du batch
            
            # Accéder à un batch entier du DataLoader, et récupérer les images, max et labels
            y_images, y_max, y_labels = [], [], []
            for idx in indices:
                y_image, max_val, label = self.data_loader[idx]
                y_images.append(y_image)
                y_max.append(max_val)
                y_labels.append(label)

            # Convertir les listes en tenseurs
            y_images = torch.stack(y_images)
            y_max = torch.stack(y_max)
            y_labels = torch.stack(y_labels)

            # Appliquer MixUp avec lambda = 0.5
            x = 0.5 * x + 0.5 * y_images

            # Mélanger les labels (additionner)
            # De-standardiser les labels
            mean_label = self.data_loader.dataset.mean_label
            std_label = self.data_loader.dataset.std_label
            y_labels = y_labels * std_label + mean_label
            labels = labels * std_label + mean_label
            new_labels = labels + y_labels
            # Restandardiser
            labels = (new_labels - mean_label) / std_label


            # Calculer la moyenne des max
            # De-standardiser les max
            mean_max = self.data_loader.dataset.mean_max_values
            std_max = self.data_loader.dataset.std_max_values
            y_max = y_max * std_max + mean_max
            max_val = max_doppler * std_max + mean_max
            # Calculer les nouvelles valeurs max
            new_max = (max_val + y_max) / 2
            # Restandardiser
            max_doppler = (new_max - mean_max) / std_max

        # Retour à (B, C, H, W)
        x = x.squeeze(2)
        
        return x, max_doppler, labels