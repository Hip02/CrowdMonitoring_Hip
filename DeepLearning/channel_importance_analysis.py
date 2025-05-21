import os
import numpy as np
import matplotlib.pyplot as plt

def average_channel_importance(input_folder, output_file="average_channel_importance.npy"):
    all_importances = []

    for fname in os.listdir(input_folder):
        if fname.endswith(".npy") and "channel_importance" in fname:
            path = os.path.join(input_folder, fname)
            arr = np.load(path)
            if arr.shape[0] == 20:
                all_importances.append(arr)

    if not all_importances:
        print("❌ No valid .npy files found.")
        return

    all_importances = np.stack(all_importances)
    mean_importance = all_importances.mean(axis=0)
    std_importance = all_importances.std(axis=0)

    np.save(os.path.join(input_folder, output_file), mean_importance)

    # Plot
    channel_labels = [f"Magnitude {i+1}" for i in range(10)] + [f"Phase Δϕ {i+1}" for i in range(10)]
    colors = ["#4477AA"] * 10 + ["#CC6677"] * 10

    plt.figure(figsize=(12, 6))
    plt.bar(range(20), mean_importance, yerr=std_importance, capsize=5, color=colors)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(range(20), channel_labels, rotation=45, ha='right')
    plt.ylabel("Avg. Prediction Change (Δy)")
    plt.title("Average Input Channel Importance over Multiple Samples")
    plt.tight_layout()

    plot_path = os.path.join(input_folder, "average_channel_importance.pdf")
    plt.savefig(plot_path, format='pdf')
    plt.close()

    print(f"✅ Saved average importance vector to {output_file}")
    print(f"📊 Plot saved to {plot_path}")

# Exemple d'utilisation :
average_channel_importance("/Users/hippolytehilgers/Downloads/Regression_AFinalExp16_fold6")
