import matplotlib.pyplot as plt

# Données
exp_ids = list(range(1, 17))
mse_values = [5.1229, 5.4736, 5.4782, 7.2121, 5.8854, 5.8813,
              5.3137, 5.2961, 7.7982, 6.0973, 5.7008, 6.1151, 7.1309, 6.5295, 8.5339, 9.3310]
std_values = [1.63, 1.74, 1.90, 2.43, 1.83, 2.23,
              2.01, 1.59, 2.13, 2.34, 1.89, 2.56, 2.48, 1.98, 2.34, 2.94]

# Création du graphique
fig, ax = plt.subplots(figsize=(10, 5))

# Couleur spéciale pour la baseline
colors = ['tab:red'] + ['black'] * (len(exp_ids) - 1)

# Tracé des barres d'erreur (style "caps" pour barres horizontales)
ax.errorbar(exp_ids, mse_values, yerr=std_values, fmt='o',
            ecolor='gray', capsize=10, elinewidth=2, capthick=2,
            color='black', markerfacecolor='white', markeredgewidth=1, alpha=0.4)

# Marqueur de la baseline (Exp 1) en rouge
ax.plot(exp_ids[0], mse_values[0], marker='o', color='tab:red', markersize=8, label='Baseline (Exp 1)')

# Ajout de la ligne horizontale
ax.axhline(y=mse_values[0], color='darkred', linestyle='--')

# Annotations des points
for x, y in zip(exp_ids, mse_values):
    ax.text(x, y + 0.15, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

# Mise en forme
ax.set_xticks(exp_ids)
ax.set_xlabel("Experiment #")
ax.set_ylabel("MSE")
ax.set_title("Data Augmentation Experiments Results")
ax.set_ylim(2, 13)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

# Sauvegarde en PDF
output_pdf = "/Users/hippolytehilgers/Downloads/DataAugmentationResults.pdf"
plt.tight_layout()
plt.savefig(output_pdf, bbox_inches="tight")