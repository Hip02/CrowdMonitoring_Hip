import matplotlib.pyplot as plt
import numpy as np

# Create an empty 32x32 grid
grid_size = 32
grid = np.zeros((grid_size, grid_size))


# Create the figure again for export without title
fig, ax = plt.subplots(figsize=(6, 6))

# Draw grid
ax.set_xticks(np.arange(-grid_size // 2, grid_size // 2 + 1, 1))
ax.set_yticks(np.arange(0, grid_size + 1, 1))
ax.grid(True, which='both', color='lightgrey', linewidth=0.5)

# Set axis limits
ax.set_xlim(-grid_size // 2, grid_size // 2)
ax.set_ylim(grid_size, 0)

# Hide all spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Manual axes at top center
ax.axvline(x=0, color='black', linewidth=1.5)
ax.axhline(y=0, color='black', linewidth=1.5)

# Axis arrows and labels
ax.annotate('', xy=(grid_size // 2, 0), xytext=(-grid_size // 2, 0),
            arrowprops=dict(arrowstyle='<->', color='black'))
ax.annotate('', xy=(0, grid_size), xytext=(0, 0),
            arrowprops=dict(arrowstyle='<->', color='black'))

# Axis text
ax.text(6, -1, 'Doppler →', ha='right', va='top', fontsize=10)
ax.text(1, 1.5, '↓ Range', ha='left', va='bottom', fontsize=10)

# Remove tick labels
ax.set_xticklabels([])
ax.set_yticklabels([])

# Save as PDF
plt.savefig("empty_grid.pdf", bbox_inches='tight')

