import numpy as np
import matplotlib.pyplot as plt
from utils.utils import train_on_custom_folds, DataLoader
from sklearn.svm import SVR
import itertools

# Load Data
base_path = "/linux/hhilgers/Dataset"
all = [f"NewExp{i}" for i in range(1, 51)]
data_loader = DataLoader(base_path, exp_list=all)

# Load Labels and Features
labels = data_loader.get_labels()
features = np.column_stack([
    data_loader.get_max_values(),
    data_loader.get_feature("MeanMagnitudes"),
    data_loader.get_feature("StdMagnitudes"),
    data_loader.get_feature("MedianMagnitudes"),
    data_loader.get_feature("SkewnessMagnitudes"),
    data_loader.get_feature("KurtosisMagnitudes"),
    data_loader.get_feature("EntropyMagnitudes"),
    data_loader.get_feature("SpectralEntropyMagnitudes"),
    data_loader.get_feature("PeakCountMagnitudes"),
    data_loader.get_feature("ImpulseFactorMagnitudes"),
    data_loader.get_feature("CrestFactorMagnitudes"),
    data_loader.get_feature("ClearanceFactorMagnitudes")
])

# Hyperparameter grid
C_values = [1, 10, 100]
epsilon_values = [0.01, 0.1, 0.5]
gamma_values = [1e-3, 1e-2, 1e-1, 1]
degrees = [1, 2, 3]

# To store results
results = []

# Grid search
for degree, C, epsilon, gamma in itertools.product(degrees, C_values, epsilon_values, gamma_values):
    model = SVR(kernel='poly', degree=degree, C=C, epsilon=epsilon, gamma=gamma, coef0=1)
    mean_mse = train_on_custom_folds(model, data_loader, features, "folds_config.yaml")
    results.append(((degree, C, epsilon, gamma), mean_mse))
    print(f"Tested degree={degree}, C={C}, epsilon={epsilon}, gamma={gamma} → Mean MSE: {mean_mse:.4f}")

# Display best
if results:
    best_params, best_mse = min(results, key=lambda x: x[1])
    print(f"\n🏆 Best config: degree={best_params[0]}, C={best_params[1]}, epsilon={best_params[2]}, gamma={best_params[3]}")
    print(f"→ Best mean MSE: {best_mse:.4f}")
