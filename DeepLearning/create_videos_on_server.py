from utils.utils import DataLoader, DopplerDataset
from utils.utils import generate_prediction_video
from networks.model import Network_Class
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2
import os

base_path = "/home/hhilgers/Dataset"
exp_list = [f"NewExp{i}" for i in range(75, 81)]
exp = "Regression/AFinalExp7"
fold = 9

stream = open(f"exp_list/{exp}.yaml", 'r')
param  = yaml.safe_load(stream)
param["DATASET"]["FOLD_NUMBER"] = fold

to_load = ["max_values", "labels", "magnitudes", "video_frames"]

data_loader = DataLoader(base_path, param, exp_list=exp_list, to_load=to_load)

resultsPath = os.path.join("results", exp, f"fold{fold}")

for exp_n in range(51, 81):

    exp_name = f"NewExp{exp_n}"

    generate_prediction_video(
        exp_name=exp_name,
        data_loader=data_loader,
        results_path=resultsPath,
        save_path=f"visualizations/fold{fold}_exp{exp_n}_visu.mp4",
        output_fps=24
    )