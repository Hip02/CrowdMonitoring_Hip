from utils.utils import DataLoader
from networks.model import Network_Class
import yaml
import numpy as np
import os

# Load Data
base_path = "/linux/hhilgers/Dataset"
all = [f"NewExp{i}" for i in range(1, 51)]

l_exp = [f"Regression/Reg_1"] # Regression/Reg_1

data_loader = DataLoader(base_path, exp_list=all)

for exp in l_exp :
    print(f"🔔 Starting {exp}")
    stream = open(f"exp_list/{exp}.yaml", 'r')
    param  = yaml.safe_load(stream)

    resultsPath = os.path.join("results", exp)

    # Create the directory
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    myNetwork = Network_Class(data_loader, param, resultsPath, sub_sample_factor=5)

    # Train the network
    train_losses, val_losses =  myNetwork.train()

    # Save the losses as results
    np.save(resultsPath + '/train_losses.npy', train_losses)
    np.save(resultsPath + '/val_losses.npy', val_losses)
  