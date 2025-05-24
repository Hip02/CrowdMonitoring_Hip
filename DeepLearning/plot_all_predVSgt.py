import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
import os
import glob

# Base path to search
base_path = "/Users/hippolytehilgers/Desktop/UCL_Hip/Mémoire/Code/DeepLearning/results/Regression/AFinalExp16"

all_gts = []
all_preds = []

# Loop through folds
for f in range(1, 6):  # Assuming folds are 1 through 5
    fold_path = os.path.join(base_path, f"fold{f}", f"fold{f}", "_PerExperimentPlots")
    if not os.path.exists(fold_path):
        continue
    
    # Find all *_gts.npy files and infer corresponding *_preds.npy
    gt_files = glob.glob(os.path.join(fold_path, "NewExp*_gts.npy"))
    for gt_file in gt_files:
        n = os.path.basename(gt_file).split("_")[0].replace("NewExp", "")
        pred_file = os.path.join(fold_path, f"NewExp{n}_preds.npy")
        if not os.path.exists(pred_file):
            continue

        gts = np.load(gt_file)
        preds = np.load(pred_file)

        all_gts.append(gts)
        all_preds.append(preds)

# Flatten arrays
all_gts = np.concatenate(all_gts)
all_preds = np.concatenate(all_preds)

# Compute density
xy = np.vstack([all_gts, all_preds])
z = gaussian_kde(xy)(xy)

# Sort by density
idx = z.argsort()
x, y, z = all_gts[idx], all_preds[idx], z[idx]

# Plotting
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

plt.figure(figsize=(6, 6))
sc = plt.scatter(x, y, c=z, cmap='plasma', norm=LogNorm(), s=10, edgecolors='none')
plt.plot([0, max(x.max(), y.max())], [0, max(x.max(), y.max())], color='black', linestyle='--', alpha=0.6, linewidth=1.5, label='Ideal')

plt.xlabel('Number of People (Pseudo Label)', fontsize=12)
plt.ylabel('Number of People (Predicted)', fontsize=12)
plt.title('Predictions vs Pseudo Label', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.axis('equal')
plt.tight_layout()
#plt.show()
plt.savefig("Predictions_vs_Ground_Truth.pdf")
