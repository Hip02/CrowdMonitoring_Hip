import argparse
import os
import sys
import yaml
import numpy as np
from utils.utils import DataLoader
from networks.model import Network_Class

# Argument parser
parser = argparse.ArgumentParser(description="Run experiment by index from l_exp list")
parser.add_argument("-exp", "--exp_index", type=int, default=0, help="Index of the experiment to run (in l_exp list, default=0)")
parser.add_argument("-fold", "--fold", type=int, default=1, help="Fold number to run (default=1)")
args = parser.parse_args()

# Load Data
base_path = "/home/hhilgers/Dataset"
all = [f"NewExp{i}" for i in range(1, 51)]

#l_exp = ["Regression/ResNet_Folds_ExpLR", "Regression/ResNet_Folds_OneCycleLR", "Regression/ResNet_Folds"]
l_exp = [f"Regression/ResNet_Folds_debug{i}" for i in range(1, 40)]

# Get experiment path from index
try:
    exp = l_exp[args.exp_index]
except IndexError:
    print(f"❌ Invalid index {args.exp_index}. Please choose a value between 0 and {len(l_exp) - 1}.")
    sys.exit(1)

fold = args.fold

print(f"🔔 Starting {exp} fold {fold}")

# Load experiment parameters from YAML
yaml_path = os.path.join("exp_list", f"{exp}.yaml")
if not os.path.isfile(yaml_path):
    print(f"❌ YAML file not found at {yaml_path}")
    sys.exit(1)

with open(yaml_path, 'r') as stream:
    param = yaml.safe_load(stream)

param["DATASET"]["FOLD_NUMBER"] = fold

# Prepare results directory
resultsPath = os.path.join("results", exp, f"fold{fold}")
os.makedirs(resultsPath, exist_ok=True)

# Setup logging
log_file_path = os.path.join(resultsPath, "log.txt")
class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
sys.stdout = TeeLogger(sys.stdout, open(log_file_path, "w"))
sys.stderr = sys.stdout

# Initialize and run the network
data_loader = DataLoader(base_path, exp_list=all)
myNetwork = Network_Class(data_loader, param, resultsPath, sub_sample_factor=1)

print(f"{len(myNetwork.dataSetTrain)} samples in training set")
print(f"{len(myNetwork.dataSetTest)} samples in test set")
print(f"{len(myNetwork.dataSetVal)} samples in validation set")

train_losses, val_losses = myNetwork.train()

# Save results
np.save(os.path.join(resultsPath, 'train_losses.npy'), train_losses)
np.save(os.path.join(resultsPath, 'val_losses.npy'), val_losses)
