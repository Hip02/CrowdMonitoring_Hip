from utils.utils import DataLoader
from utils.utils import plot_label_distribution
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

base_path = "/home/hhilgers/Dataset"
exp_list = [f"NewExp{i}" for i in range(51, 81)]
to_load = ["labels"]
data_loader = DataLoader(base_path, exp_list=exp_list, to_load=to_load)

labels = data_loader.get_labels()

## FIRST GRAPH

plot_label_distribution(labels, filename="test_labels_distribution.pdf")



## SECOND GRAPH

ft_size = 15
lbl_size = 14

# Paramètres
groups_size = 1000
num_experiments = 30  # 10x5 grid
n_rows, n_cols = 3, 10
duration_sec = 60

# Temps exprimé en secondes
time = np.linspace(0, duration_sec, groups_size)

# Calcul des moyennes par expérience et tri
group_means = []
for i in range(num_experiments):
    group_start = i * groups_size
    group_end = group_start + groups_size
    mean_value = np.mean(labels[group_start:group_end])
    group_means.append((i, mean_value))  # (index_original, moyenne)

# Tri par moyenne de personnes (ordre croissant)
sorted_groups = sorted(group_means, key=lambda x: x[1])

# Normalisation pour la colormap (fixée à max 20.194)
all_means_sorted = [mean for (_, mean) in sorted_groups]
norm = colors.Normalize(vmin=min(all_means_sorted), vmax=20.194)
cmap = cm.inferno
scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

# Définition des 5 catégories (très faible à très forte densité)
thresholds = [2, 4, 8, 15]

def get_density_category(mean_val):
    if mean_val <= thresholds[0]:
        return "Very Low"
    elif mean_val <= thresholds[1]:
        return "Low"
    elif mean_val <= thresholds[2]:
        return "Medium"
    elif mean_val <= thresholds[3]:
        return "High"
    else:
        return "Very High"

# Couleurs de bordure pour les 5 catégories
border_colors = {
    "Very Low": "deepskyblue",
    "Low": "limegreen",
    "Medium": "gold",
    "High": "darkorange",
    "Very High": "crimson"
}

# Création de la figure
fig, axs = plt.subplots(n_rows, n_cols, figsize=(24, 8), sharex=True, sharey=True)
axs = axs.flatten()

# Échelle Y uniforme (fixée à 0-26)
ymin, ymax = 0, 26

# Dictionnaire pour imprimer les groupes à la fin
category_indices = {"Very Low": [], "Low": [], "Medium": [], "High": [], "Very High": []}

for plot_idx, (original_idx, mean_val) in enumerate(sorted_groups):
    group_start = original_idx * groups_size
    group_end = group_start + groups_size

    if group_end > len(labels):
        continue

    group_labels = labels[group_start:group_end]
    color = scalar_map.to_rgba(mean_val)

    axs[plot_idx].plot(time, group_labels, color=color, linewidth=1.8)
    axs[plot_idx].set_title(f'exp {original_idx+1} | µ = {mean_val:.2f}', fontsize=ft_size)
    axs[plot_idx].grid(True, linestyle='--', alpha=0.6)
    axs[plot_idx].set_ylim(ymin, ymax)
    axs[plot_idx].set_xlim(0, duration_sec)
    axs[plot_idx].tick_params(axis='both', labelsize=lbl_size)

    # Catégorie et couleur de bordure
    category = get_density_category(mean_val)
    category_indices[category].append(original_idx + 1)
    for spine in axs[plot_idx].spines.values():
        spine.set_color(border_colors[category])
        spine.set_linewidth(2.5)

# Étiquettes des axes sur les bords
for i in range(n_rows):
    axs[i * n_cols].set_ylabel("People", fontsize=ft_size)
for j in range(n_cols):
    axs[(n_rows - 1) * n_cols + j].set_xlabel("Time (s)", fontsize=ft_size)

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
cbar = fig.colorbar(scalar_map, cax=cbar_ax)
cbar.set_label('Mean number of people', fontsize=ft_size+2)

# Légende des catégories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='white', edgecolor=border_colors["Very Low"], label=f'Very Low (µ ≤ {thresholds[0]})', linewidth=2.5),
    Patch(facecolor='white', edgecolor=border_colors["Low"], label=f'Low ({thresholds[0]} < µ ≤ {thresholds[1]})', linewidth=2.5),
    Patch(facecolor='white', edgecolor=border_colors["Medium"], label=f'Medium ({thresholds[1]} < µ ≤ {thresholds[2]})', linewidth=2.5),
    Patch(facecolor='white', edgecolor=border_colors["High"], label=f'High ({thresholds[2]} < µ ≤ {thresholds[3]})', linewidth=2.5),
    Patch(facecolor='white', edgecolor=border_colors["Very High"], label=f'Very High (µ > {thresholds[3]})', linewidth=2.5)
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=5, fontsize=ft_size+2, frameon=False)

# Mise en page finale
plt.tight_layout(rect=[0, 0, 0.8, 0.88]) 
plt.subplots_adjust(top=0.75)
fig.suptitle("Experiments sorted by average number of people (5 Density Categories Highlighted)", fontsize=ft_size+8, y=0.95)

plt.savefig("test_experiments_sorted_by_density.pdf")

# Print saved path
print("✅ Saved figure to test_experiments_sorted_by_density.pdf")

# Affichage des indices par catégorie
print("\n=== Category indices by density level ===")
for cat, indices in category_indices.items():
    print(f"{cat} ({len(indices)} exps): {sorted(indices)}")
