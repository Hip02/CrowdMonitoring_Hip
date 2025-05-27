import matplotlib.pyplot as plt
import seaborn as sns

# Données
labels_colored = [
    [("1 frame", 'black'), ("Magnitude", 'black')],
    [("1 frame", 'black'), ("Magnitude", 'black'), ("Cropped", "#276523")],
    [("1 frame", 'black'), ("Magnitude", 'black'), ("Phase", "#2980b9")],
    [("1 frame", 'black'), ("Magnitude", 'black'), ("Phase", "#2980b9"), ("Cropped", "#276523")],
    [("10 frames", 'black'), ("Magnitude", 'black')],
    [("10 frames", 'black'), ("Magnitude", 'black'), ("Cropped", "#276523")],
]

mean_mse = [5.12, 5.22, 4.65, 4.94, 4.31, 5.23]
std_mse = [1.63, 0.88, 0.95, 1.26, 1.97, 1.49]
colors = ["#95a5a6", "#276523", "#95a5a6", "#276523", "#95a5a6", "#276523"]

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = range(len(mean_mse))

# Tracer les barres
for i, (x, mean, std, color) in enumerate(zip(x_pos, mean_mse, std_mse, colors)):
    ax.bar(x, 0.1, bottom=mean-0.05, width=0.6, color=color, zorder=3)
    ax.vlines(x, mean-std, mean+std, color=color, alpha=0.3, linewidth=12)
    ax.text(x, mean+0.1, f"{mean:.2f}", ha='center', va='bottom', fontsize=11, color=color)

# Tracer des lignes verticales pointillées pour grouper les experiences par 2
for i in range(2, len(mean_mse), 2):
    ax.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=1.8, alpha=0.5)

# Supprimer les ticks x
ax.set_xticks([])
ax.set_xlim(-0.5, len(mean_mse) - 0.5)

# Ajouter les étiquettes en plusieurs lignes sous les barres
for i, label_parts in enumerate(labels_colored):
    ypos = 1.8  # position de base sous l'axe y
    for j, (text, color) in enumerate(label_parts):
        ax.text(i, ypos - j * 0.25, text, color=color, fontsize=10, ha='center', va='top')

ax.set_ylabel("Mean Squared Error (MSE)")
ax.set_title("Ablation Study: Impact of Cropping RDMs")
ax.set_ylim(2, 7)
ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

plt.tight_layout()
plt.savefig("effect_of_crop.pdf", dpi=300)
#plt.show()