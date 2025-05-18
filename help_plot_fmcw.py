import numpy as np
import matplotlib.pyplot as plt

# Time vector
t = np.linspace(0, 5, 1000)
# Continuous signal always > 0
e_rt = 0.5 + 0.4 * np.sin(2 * np.pi * 0.6 * t) * np.exp(-0.3 * t)

# Sampling points
Tc = 1.0
Ts = 0.2
n_c = np.arange(0, 5, 1)
n_s = np.arange(0, Tc, Ts)
sample_times = np.array([nc * Tc + ns for nc in n_c for ns in n_s if nc * Tc + ns <= t[-1]])
sample_values = 0.5 + 0.4 * np.sin(2 * np.pi * 0.6 * sample_times) * np.exp(-0.3 * sample_times)

# Plot
plt.figure(figsize=(8, 3))
plt.plot(t, e_rt, color='black', linewidth=1.2, label=r"$e_r(t)$")

# Draw vertical bars and open circles for each sample
for x, y in zip(sample_times, sample_values):
    plt.plot([x, x], [0, y], color='black', linewidth=0.8)
    plt.plot(x, y, 'o', markerfacecolor='white', markeredgecolor='black', markersize=5)

plt.title(r"Sampling of $e_r(t)$ at $t = n_c T_c + n_s T_s$", fontsize=12)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("er_sampling_positive_with_deltas.pdf")

"/mnt/data/er_sampling_positive_with_deltas.pdf"
