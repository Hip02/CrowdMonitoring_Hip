import torch
import torchvision.transforms.functional as TF
# Rechargement de tout l'environnement proprement

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import os

base = "/Volumes/HIP_BACKUP/MEMOIRE/Code/MyDB2"

base_img_path = base + "/NewExp26/RadarMagnitudes/map_105_15.png"
other_img_paths=[
        base + "/NewExp29/RadarMagnitudes/map_35_19.png",
        base + "/NewExp30/RadarMagnitudes/map_625_19.png"
    ]

output_dir = "/Users/hippolytehilgers/Downloads/DataAugmentationExamples"
os.makedirs(output_dir, exist_ok=True)

# Chargement image de base (grayscale)
base_img = Image.open(base_img_path).convert("L")
base_tensor = TF.to_tensor(base_img)  # (1, H, W)

# Appliquer différentes augmentations (2 intensités quand pertinent)
results = [
    ("original", base_tensor[0]),
    ("hflip", TF.hflip(base_tensor)[0]),
    ("vflip", TF.vflip(base_tensor)[0]),
    ("rot90", torch.rot90(base_tensor, k=1, dims=[-1, -2])[0]),
]

# Brightness
results.append(("brightness_low", TF.adjust_brightness(base_tensor, 0.7)[0]))
results.append(("brightness_high", TF.adjust_brightness(base_tensor, 1.4)[0]))

# Contrast
results.append(("contrast_low", TF.adjust_contrast(base_tensor, 0.7)[0]))
results.append(("contrast_high", TF.adjust_contrast(base_tensor, 1.5)[0]))

# Gaussian Noise
noise_low = torch.randn_like(base_tensor) * 0.05
noise_high = torch.randn_like(base_tensor) * 0.2
results.append(("gaussian_noise_low", (base_tensor + noise_low).clamp(0, 1)[0]))
results.append(("gaussian_noise_high", (base_tensor + noise_high).clamp(0, 1)[0]))

# Gaussian Blur
blur = T.GaussianBlur(kernel_size=5, sigma=1.0)
results.append(("gaussian_blur_low", blur(base_tensor)[0]))
blur_strong = T.GaussianBlur(kernel_size=11, sigma=3.0)
results.append(("gaussian_blur_high", blur_strong(base_tensor)[0]))

# Mixup
mix = base_tensor.clone()
for i, path in enumerate(other_img_paths):
    mix_img = Image.open(path).convert("L")
    mix_tensor = TF.to_tensor(mix_img)
    results.append((f"mixup_source_{i+1}", mix_tensor[0]))
    mix += mix_tensor
mix /= (1 + len(other_img_paths))
results.append((f"mixup_combined", mix[0]))

# Enregistrement des figures
for name, tensor_img in results:
    fig, ax = plt.subplots()
    ax.imshow(tensor_img, cmap="inferno")
    #ax.set_title(name)
    ax.axis("off")
    fig.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)