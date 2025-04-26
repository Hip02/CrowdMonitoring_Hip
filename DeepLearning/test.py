import argparse
import os
import sys
import yaml
import numpy as np
from utils.utils import DataLoader, plot_learning_curves
from networks.model import Network_Class

# Argument parser
parser = argparse.ArgumentParser(description="Run one or more experiments from the l_exp list")
parser.add_argument("-exp", "--exp_indices", type=int, nargs='+', required=True,
                    help="Indices of experiments to run (in l_exp list)")
parser.add_argument("-fold", "--fold", type=int, default=1, help="Fold number to run (default=1)")
parser.add_argument("-model", "--model", type=str, default=None,
                    help="Optional model name or path to load weights into (default=None)")
args = parser.parse_args()

# Load Data
base_path = "/home/hhilgers/Dataset"
if args.fold == 9 :
    all = [f"NewExp{i}" for i in range(1, 81)]
else:
    all = [f"NewExp{i}" for i in range(1, 51)]

l_exp = [f"Regression/AFinalExp{i}" for i in range(0, 40)]

fold = args.fold

for exp_index in args.exp_indices:
    try:
        exp = l_exp[exp_index]
    except IndexError:
        print(f"❌ Invalid index {exp_index}. Please choose a value between 0 and {len(l_exp) - 1}.")
        continue

    print(f"🔔 Starting {exp} fold {fold}")

    # Load experiment parameters from YAML
    yaml_path = os.path.join("exp_list", f"{exp}.yaml")
    if not os.path.isfile(yaml_path):
        print(f"❌ YAML file not found at {yaml_path}")
        continue

    with open(yaml_path, 'r') as stream:
        param = yaml.safe_load(stream)

    param["DATASET"]["FOLD_NUMBER"] = fold

    resultsPath = os.path.join("results", exp, f"fold{fold}")
    os.makedirs(resultsPath, exist_ok=True)

    data_loader = DataLoader(base_path, param, exp_list=all)
    myNetwork = Network_Class(data_loader, param, resultsPath, sub_sample_factor=1)

    myNetwork.loadWeight(modelPath=args.model)
    myNetwork.test()

    train_losses = np.load(os.path.join(resultsPath, "train_losses.npy"))
    val_losses = np.load(os.path.join(resultsPath, "val_losses.npy"))

    plot_learning_curves(train_losses, val_losses, resultsPath)

print("✅ All experiments completed.")
