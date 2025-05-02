import numpy as np
import matplotlib.pyplot as plt
from utils.utils import train_on_custom_folds
from utils.utils import DataLoader

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

from sklearn.svm import SVR
import itertools

features = ["MeanMagnitudes", "StdMagnitudes", "MedianMagnitudes", "SkewnessMagnitudes", 
            "KurtosisMagnitudes", "EntropyMagnitudes", "SpectralEntropyMagnitudes", 
            "PeakCountMagnitudes", "ImpulseFactorMagnitudes", "CrestFactorMagnitudes", 
            "ClearanceFactorMagnitudes"]

# Grille d'hyperparamètres
C_values = [1, 10, 100, 1000]
epsilon_values = [0.01, 0.1, 0.5]
gamma_values = [1e-3, 1e-2, 1e-1, 1]

# Combinaisons déjà testées (tuple de (C, epsilon, gamma))
completed = {
    (1, 0.01, 0.001), (1, 0.01, 0.01), (1, 0.01, 0.1), (1, 0.01, 1),
    (1, 0.1, 0.001), (1, 0.1, 0.01), (1, 0.1, 0.1), (1, 0.1, 1),
    (1, 0.5, 0.001), (1, 0.5, 0.01), (1, 0.5, 0.1), (1, 0.5, 1),
    (10, 0.01, 0.001), (10, 0.01, 0.01), (10, 0.01, 0.1), (10, 0.01, 1),
    (10, 0.1, 0.001), (10, 0.1, 0.01), (10, 0.1, 0.1), (10, 0.1, 1),
    (10, 0.5, 0.001), (10, 0.5, 0.01), (10, 0.5, 0.1), (10, 0.5, 1),
    (100, 0.01, 0.001), (100, 0.01, 0.01), (100, 0.01, 0.1), (100, 0.01, 1),
    (100, 0.1, 0.001), (100, 0.1, 0.01), (100, 0.1, 0.1), (100, 0.1, 1)  # inclut la combinaison finale
}

# Pour stocker les résultats
results = []

# Grid search partiel : uniquement les combinaisons restantes
for C, epsilon, gamma in itertools.product(C_values, epsilon_values, gamma_values):
    if (C, epsilon, gamma) in completed:
        continue
    model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
    mean_mse = train_on_custom_folds(model, data_loader, features, "folds_config.yaml")
    results.append(((C, epsilon, gamma), mean_mse))
    print(f"Tested C={C}, epsilon={epsilon}, gamma={gamma} → Mean MSE: {mean_mse:.4f}")

# Optionnel : afficher le meilleur modèle parmi les nouveaux résultats
if results:
    best_params, best_mse = min(results, key=lambda x: x[1])
    print(f"\nBest new parameters: C={best_params[0]}, epsilon={best_params[1]}, gamma={best_params[2]}")
    print(f"Best mean MSE from new runs: {best_mse:.4f}")
else:
    print("All parameter combinations have already been tested.")