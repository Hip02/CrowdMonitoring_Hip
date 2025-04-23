import argparse
import os
import sys
import yaml
import numpy as np
from utils.utils import DataLoader
from networks.model import Network_Class

# Argument parser
parser = argparse.ArgumentParser(description="Run one or more experiments from the l_exp list")
parser.add_argument("-exp", "--exp_indices", type=int, nargs='+', required=True,
                    help="Indices of experiments to run (in l_exp list)")
parser.add_argument("-fold", "--fold", type=int, default=1, help="Fold number to run (default=1)")
args = parser.parse_args()

# Load Data
base_path = "/home/hhilgers/Dataset"
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

    # Prepare results directory
    resultsPath = os.path.join("results", exp, f"fold{fold}")
    os.makedirs(resultsPath, exist_ok=True)

    # Initialize and run the network
    data_loader = DataLoader(base_path, param, exp_list=all)
    myNetwork = Network_Class(data_loader, param, resultsPath, sub_sample_factor=1)

    print(f"{len(myNetwork.dataSetTrain)} samples in training set")
    print(f"{len(myNetwork.dataSetTest)} samples in test set")
    print(f"{len(myNetwork.dataSetVal)} samples in validation set")

    train_losses, val_losses = myNetwork.train()
    # for idx in range(4):
    #     myNetwork.visualize_data_augmentation(idx=idx)

print("✅ All experiments completed.")
