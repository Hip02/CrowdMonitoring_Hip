import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter

# Convert MLP data (activation=ReLU only for now) into a list of tuples
mlp_data_relu = [
    ("ReLU", 0.1, "32", 9.47), ("ReLU", 0.1, "64", 9.71), ("ReLU", 0.1, "128", 9.37), ("ReLU", 0.1, "256", 9.18),
    ("ReLU", 0.1, "32-16", 9.69), ("ReLU", 0.1, "64-32", 9.37), ("ReLU", 0.1, "128-64", 11.74), ("ReLU", 0.1, "256-128", 18.11),
    ("ReLU", 0.1, "32-16-8", 9.74), ("ReLU", 0.1, "64-32-16", 9.51), ("ReLU", 0.1, "128-64-32", 10.55), ("ReLU", 0.1, "256-128-64", 9.93),

    ("ReLU", 0.01, "32", 9.06), ("ReLU", 0.01, "64", 9.20), ("ReLU", 0.01, "128", 9.26), ("ReLU", 0.01, "256", 9.26),
    ("ReLU", 0.01, "32-16", 9.19), ("ReLU", 0.01, "64-32", 9.18), ("ReLU", 0.01, "128-64", 9.12), ("ReLU", 0.01, "256-128", 9.34),
    ("ReLU", 0.01, "32-16-8", 9.19), ("ReLU", 0.01, "64-32-16", 9.29), ("ReLU", 0.01, "128-64-32", 9.23), ("ReLU", 0.01, "256-128-64", 9.15),

    ("ReLU", 0.001, "32", 9.29), ("ReLU", 0.001, "64", 9.29), ("ReLU", 0.001, "128", 9.18), ("ReLU", 0.001, "256", 9.20),
    ("ReLU", 0.001, "32-16", 9.18), ("ReLU", 0.001, "64-32", 9.09), ("ReLU", 0.001, "128-64", 9.08), ("ReLU", 0.001, "256-128", 9.14),
    ("ReLU", 0.001, "32-16-8", 9.21), ("ReLU", 0.001, "64-32-16", 9.28), ("ReLU", 0.001, "128-64-32", 9.20), ("ReLU", 0.001, "256-128-64", 9.08)
]

mlp_data_tanh = [
    ("Tanh", 0.1, "32", 9.59), ("Tanh", 0.1, "64", 9.59), ("Tanh", 0.1, "128", 9.41), ("Tanh", 0.1, "256", 9.37),
    ("Tanh", 0.1, "32-16", 10.31), ("Tanh", 0.1, "64-32", 10.46), ("Tanh", 0.1, "128-64", 11.00), ("Tanh", 0.1, "256-128", 12.21),
    ("Tanh", 0.1, "32-16-8", 12.51), ("Tanh", 0.1, "64-32-16", 12.79), ("Tanh", 0.1, "128-64-32", 13.77), ("Tanh", 0.1, "256-128-64", 14.38),

    ("Tanh", 0.01, "32", 9.39), ("Tanh", 0.01, "64", 9.08), ("Tanh", 0.01, "128", 9.36), ("Tanh", 0.01, "256", 9.27),
    ("Tanh", 0.01, "32-16", 9.04), ("Tanh", 0.01, "64-32", 9.31), ("Tanh", 0.01, "128-64", 9.32), ("Tanh", 0.01, "256-128", 9.15),
    ("Tanh", 0.01, "32-16-8", 9.36), ("Tanh", 0.01, "64-32-16", 9.41), ("Tanh", 0.01, "128-64-32", 9.32), ("Tanh", 0.01, "256-128-64", 9.57),

    ("Tanh", 0.001, "32", 9.24), ("Tanh", 0.001, "64", 9.34), ("Tanh", 0.001, "128", 9.36), ("Tanh", 0.001, "256", 9.47),
    ("Tanh", 0.001, "32-16", 9.00), ("Tanh", 0.001, "64-32", 9.15), ("Tanh", 0.001, "128-64", 9.02), ("Tanh", 0.001, "256-128", 9.01),
    ("Tanh", 0.001, "32-16-8", 9.05), ("Tanh", 0.001, "64-32-16", 9.28), ("Tanh", 0.001, "128-64-32", 9.09), ("Tanh", 0.001, "256-128-64", 9.05)
]

mlp_data_sigmoid = [
    ("Sigmoid", 0.1, "32", 9.48), ("Sigmoid", 0.1, "64", 9.41), ("Sigmoid", 0.1, "128", 9.43), ("Sigmoid", 0.1, "256", 9.31),
    ("Sigmoid", 0.1, "32-16", 9.58), ("Sigmoid", 0.1, "64-32", 9.65), ("Sigmoid", 0.1, "128-64", 10.97), ("Sigmoid", 0.1, "256-128", 11.34),
    ("Sigmoid", 0.1, "32-16-8", 12.26), ("Sigmoid", 0.1, "64-32-16", 19.77), ("Sigmoid", 0.1, "128-64-32", 34.32), ("Sigmoid", 0.1, "256-128-64", 36.12),

    ("Sigmoid", 0.01, "32", 9.24), ("Sigmoid", 0.01, "64", 9.18), ("Sigmoid", 0.01, "128", 9.10), ("Sigmoid", 0.01, "256", 9.39),
    ("Sigmoid", 0.01, "32-16", 9.08), ("Sigmoid", 0.01, "64-32", 9.20), ("Sigmoid", 0.01, "128-64", 9.08), ("Sigmoid", 0.01, "256-128", 9.09),
    ("Sigmoid", 0.01, "32-16-8", 9.32), ("Sigmoid", 0.01, "64-32-16", 9.06), ("Sigmoid", 0.01, "128-64-32", 9.15), ("Sigmoid", 0.01, "256-128-64", 9.29),

    ("Sigmoid", 0.001, "32", 9.36), ("Sigmoid", 0.001, "64", 9.38), ("Sigmoid", 0.001, "128", 9.31), ("Sigmoid", 0.001, "256", 9.28),
    ("Sigmoid", 0.001, "32-16", 9.16), ("Sigmoid", 0.001, "64-32", 9.08), ("Sigmoid", 0.001, "128-64", 9.09), ("Sigmoid", 0.001, "256-128", 9.12),
    ("Sigmoid", 0.001, "32-16-8", 9.12), ("Sigmoid", 0.001, "64-32-16", 9.21), ("Sigmoid", 0.001, "128-64-32", 9.05), ("Sigmoid", 0.001, "256-128-64", 9.12)
]

# Create a dataframe
df_mlp = pd.DataFrame(mlp_data_sigmoid, columns=["activation", "lr", "layers", "Mean MSE"])

activation_name = "Sigmoid"

def plot_heatmap_with_tuple_labels(data, color):
    data = data.copy()
    # Convert layer string to tuple-style label and sort key
    data["tuple_label"] = data["layers"].apply(lambda x: f"{tuple(map(int, x.split('-')))}")
    data["sort_key"] = data["layers"].apply(lambda x: (len(x.split("-")), list(map(int, x.split("-")))))
    data = data.sort_values("sort_key")
    # Set tuple_label as categorical for correct order on y-axis
    data["tuple_label"] = pd.Categorical(data["tuple_label"], categories=data["tuple_label"].unique(), ordered=True)
    # Plot using tuple-style labels
    pivot = data.pivot(index="tuple_label", columns="activation", values="Mean MSE")
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="magma", cbar=True, linewidths=0.5)

# Replot with formatted architecture labels
g = sns.FacetGrid(df_mlp, col="lr", col_wrap=3, height=4, sharex=False, sharey=True)
g.map_dataframe(plot_heatmap_with_tuple_labels)
g.set_titles(col_template="Learning rate = {col_name}")
g.set_axis_labels("activation", "architecture")


plt.subplots_adjust(top=0.85)
g.fig.suptitle(f"Mean MSE across MLP hyperparameters (activation={activation_name})", fontsize=14)

# Save
plt.savefig("/Users/hippolytehilgers/Downloads/mlp_sigmoid_gridsearch_heatmap.pdf", format="pdf")
