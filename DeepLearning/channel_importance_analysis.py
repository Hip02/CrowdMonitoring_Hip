import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def average_channel_importance(exp, fold):
    base_folder = "/linux/hhilgers/Code/CrowdMonitoring_Hip/DeepLearning/gradcam_output"
    folder_name = f"Regression_AFinalExp{exp}_fold{fold}"
    input_folder = os.path.join(base_folder, folder_name)

    all_importances = []

    for fname in os.listdir(input_folder):
        if fname.endswith(".npy") and "channel_importance" in fname:
            path = os.path.join(input_folder, fname)
            arr = np.load(path)
            if arr.shape[0] == 20:
                all_importances.append(np.abs(arr))

    if not all_importances:
        print("❌ No valid .npy files found in", input_folder)
        return

    all_importances = np.stack(all_importances)
    mean_importance = all_importances.mean(axis=0)
    std_importance = all_importances.std(axis=0)

    np.save(os.path.join(input_folder, "average_channel_importance.npy"), mean_importance)

    # Plot
    channel_labels = [f"$A_{{t-{i}}}$" if i > 0 else "$A_t$" for i in range(10)] + [f"$\\Delta\\phi_{{{i+1}}}$" for i in range(10)]
    colors = ["#4477AA"] * 10 + ["#CC6677"] * 10

    plt.figure(figsize=(12, 6))
    plt.bar(range(20), mean_importance, yerr=std_importance, capsize=5, color=colors)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(range(20), channel_labels)
    plt.ylabel("Avg. Prediction Change (Δy)")
    plt.title(f"Average Channel Importance (Exp {exp}, Fold {fold})")
    plt.tight_layout()

    plot_path = os.path.join(input_folder, "average_channel_importance.pdf")
    plt.savefig(plot_path, format='pdf')
    plt.close()

    print(f"✅ Saved average importance vector to {input_folder}/average_channel_importance.npy")
    print(f"📊 Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", "--exp_indice", type=int, required=True,
                        help="Indices of experiment to run")
    parser.add_argument("-fold", "--fold", type=int, default=1,
                        help="Fold number to run (default=1)")
    args = parser.parse_args()

    average_channel_importance(args.exp_indice, args.fold)
