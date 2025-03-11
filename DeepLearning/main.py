from utils.utils import DataLoader
from networks.model import Network_Class
import yaml
import numpy as np
import os
import sys

# Load Data
base_path = "/home/hhilgers/Dataset"
all = [f"NewExp{i}" for i in range(1, 51)]

l_exp = ["Regression/BET_Reg_4"]#, "Regression/BET_Reg_48CH"]

data_loader = DataLoader(base_path, exp_list=all)

for exp in l_exp:
    print(f"🔔 Starting {exp}")
    stream = open(f"exp_list/{exp}.yaml", 'r')
    param  = yaml.safe_load(stream)

    resultsPath = os.path.join("results", exp)

    # Create the directory
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    # Setup logging to file AND console
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

    myNetwork = Network_Class(data_loader, param, resultsPath, sub_sample_factor=1)

    # Train the network
    train_losses, val_losses = myNetwork.train()

    # Save the losses as results
    np.save(os.path.join(resultsPath, 'train_losses.npy'), train_losses)
    np.save(os.path.join(resultsPath, 'val_losses.npy'), val_losses)
