import os
import random
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# ====== PARAMÈTRES À CONFIGURER ======
exp_numbers = [2, 10, 17, 26, 41]
fold_number = 2
num_samples = 10
frame_indices = sorted(random.sample(range(1000), num_samples))
output_dir = "./output_gradcam_pairs"
target_size = (256, 256)  # Resize commun à toutes les cartes
# =====================================

# Crée le dossier de sortie
os.makedirs(output_dir, exist_ok=True)

for exp_number in exp_numbers:

    # Répertoires
    base_dataset = f"/linux/hhilgers/Dataset/NewExp{exp_number}"
    base_heatmap = f"/linux/hhilgers/Code/CrowdMonitoring_Hip/DeepLearning/results/Regression/AFinalExp16/fold{fold_number}/_GradCAM_Heatmaps/NewExp{exp_number}"

    def find_image_path(directory, frame_index):
        """Trouve le chemin de la map_{frame}_{n}.png sans connaître n."""
        pattern = os.path.join(directory, f"map_{frame_index}_*.png")
        matches = glob.glob(pattern)
        return matches[0] if matches else None

    def superpose_heatmap(base_img, heatmap_gray):
        """Superpose la heatmap (grayscale) sur l’image de base avec alpha progressive."""
        base_color = cv2.applyColorMap(base_img, cv2.COLORMAP_VIRIDIS).astype(np.float32)
        heatmap_norm = cv2.normalize(heatmap_gray.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
        alpha = 0.4 + 0.4 * heatmap_norm[..., None]
        overlay = (1 - alpha) * base_color + alpha * heatmap_colored
        return np.clip(overlay, 0, 255).astype(np.uint8)

    for i, frame_idx in enumerate(frame_indices):
        # Fichiers
        heatmap_path = os.path.join(base_heatmap, f"frame_{frame_idx}.png")
        mag_path = find_image_path(os.path.join(base_dataset, "RadarMagnitudes"), frame_idx)
        diff_path = find_image_path(os.path.join(base_dataset, "DiffPhases"), frame_idx)

        if not (os.path.exists(heatmap_path) and mag_path and diff_path):
            print(f"⚠️  Frame {frame_idx}: fichiers manquants, on saute.")
            continue

        # Chargements
        heatmap_gray = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        mag = cv2.imread(mag_path, cv2.IMREAD_GRAYSCALE)
        diff = cv2.imread(diff_path, cv2.IMREAD_GRAYSCALE)

        # Resize
        heatmap = cv2.resize(heatmap_gray, target_size, interpolation=cv2.INTER_LINEAR)
        mag = cv2.resize(mag, target_size, interpolation=cv2.INTER_LINEAR)
        diff = cv2.resize(diff, target_size, interpolation=cv2.INTER_LINEAR)

        # Superpositions
        mag_overlay = superpose_heatmap(mag, heatmap)
        diff_overlay = superpose_heatmap(diff, heatmap)

        # Concaténer horizontalement
        combined = np.hstack((mag_overlay, diff_overlay))

        # Enregistrer
        save_path = os.path.join(output_dir, f"exp_{exp_number}_frame_{frame_idx}.png")
        cv2.imwrite(save_path, combined)
        print(f"✅ Sauvegardé : {save_path}")
