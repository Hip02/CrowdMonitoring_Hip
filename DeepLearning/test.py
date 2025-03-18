import argparse
import os
import sys
import yaml
import numpy as np
from utils.utils import DataLoader
from networks.model import Network_Class

from utils.utils import DataLoader, DopplerDataset, plot_learning_curves
from networks.model import Network_Class
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

# Argument parser
parser = argparse.ArgumentParser(description="Run experiment by index from l_exp list")
parser.add_argument("-exp", "--exp_index", type=int, default=0, help="Index of the experiment to run (in l_exp list, default=0)")
args = parser.parse_args()

# Load Data
base_path = "/home/hhilgers/Dataset"
all = [f"NewExp{i}" for i in range(1, 51)]

l_exp = ["Regression/BET_ResNet1", "Regression/BET_ResNet2","Regression/BET_ResNet3", "Regression/BET_ResNet4", "Regression/BET_ResNet5", "Regression/BET_ResNet6"]

# Get experiment path from index
try:
    exp = l_exp[args.exp_index]
except IndexError:
    print(f"❌ Invalid index {args.exp_index}. Please choose a value between 0 and {len(l_exp) - 1}.")
    sys.exit(1)

print(f"🔔 Starting {exp}")

# Load experiment parameters from YAML
yaml_path = os.path.join("exp_list", f"{exp}.yaml")
if not os.path.isfile(yaml_path):
    print(f"❌ YAML file not found at {yaml_path}")
    sys.exit(1)

with open(yaml_path, 'r') as stream:
    param = yaml.safe_load(stream)

resultsPath = os.path.join("results", exp)
os.makedirs(resultsPath, exist_ok=True)

data_loader = DataLoader(base_path, exp_list=all)
myNetwork = Network_Class(data_loader, param, resultsPath, sub_sample_factor=1)

myNetwork.loadWeight()
myNetwork.test()

train_losses = np.load(os.path.join(resultsPath, "train_losses.npy"))
val_losses = np.load(os.path.join(resultsPath, "val_losses.npy"))

plot_learning_curves(train_losses, val_losses, resultsPath)