import numpy as np
import matplotlib.pyplot as plt
from utils.utils import train_on_custom_folds, DataLoader
from sklearn.neural_network import MLPRegressor
import itertools

# Load Data
base_path = "/linux/hhilgers/Dataset"
all = [f"NewExp{i}" for i in range(1, 51)]
data_loader = DataLoader(base_path, exp_list=all)

features = ["MeanMagnitudes", "StdMagnitudes", "MedianMagnitudes", "SkewnessMagnitudes", 
            "KurtosisMagnitudes", "EntropyMagnitudes", "SpectralEntropyMagnitudes", 
            "PeakCountMagnitudes", "ImpulseFactorMagnitudes", "CrestFactorMagnitudes", 
            "ClearanceFactorMagnitudes"]

# Hyperparameter grid (MLP)
hidden_layer_sizes = [(32,), (64,), (128,), (256,), (32,16), (64,32), (128,64), (256,128), 
                      (32,16,8), (64,32,16), (128,64,32), (256,128,64)]
activation_functions = ['relu', 'tanh', 'logistic']
learning_rates_init = [1e-2, 1e-3, 1e-4]

# To store results
results = []

# Grid search
for hid_l_s, act, lr in itertools.product(activation_functions, learning_rates_init, hidden_layer_sizes):
    model = MLPRegressor(hidden_layer_sizes=hid_l_s, activation=act, learning_rate_init=lr, max_iter=1000)
    mean_mse = train_on_custom_folds(model, data_loader, features, "folds_config.yaml")
    results.append(((hid_l_s, act, lr), mean_mse))
    print(f"Tested hidden_layer_sizes={hid_l_s}, activation={act}, learning_rate_init={lr} → Mean MSE: {mean_mse:.4f}")

# Display best
if results:
    best_params, best_mse = min(results, key=lambda x: x[1])
    print(f"\n🏆 Best config: hidden_layer_sizes={best_params[0]}, activation={best_params[1]}, learning_rate_init={best_params[2]}")
    print(f"→ Best mean MSE: {best_mse:.4f}")
