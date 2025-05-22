from utils.utils import DataLoader
from utils.utils import generate_prediction_video, generate_prediction_video_with_gradcam
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2
import os

# Load Data
base_path = "/home/hhilgers/Dataset"
all = [f"NewExp{i}" for i in range(1, 51)]


l_exp = [f"Regression/AFinalExp16"]
folds = list(range(2, 3))

for exp in l_exp :
    for fold in folds :
        print(f"🔔 Starting {exp} fold {fold}")
        stream = open(f"exp_list/{exp}.yaml", 'r')
        param  = yaml.safe_load(stream)
        param["DATASET"]["FOLD_NUMBER"] = fold

        to_load = ["max_values", "labels", "magnitudes", "video_frames"]

        data_loader = DataLoader(base_path, param, exp_list=all, to_load=to_load)
        
        resultsPath = os.path.join("results", exp, f"fold{fold}")

        # Create the directory
        if not os.path.exists(resultsPath):
            os.makedirs(resultsPath)

        exp_ns = [26]
        
        for exp_n in exp_ns:
            exp_name = f"NewExp{exp_n}"

            generate_prediction_video_with_gradcam(
                exp_name=exp_name,
                data_loader=data_loader,
                results_path=resultsPath,
                save_path=f"visualizations/fold{fold}_exp{exp_n}_visu_gradcam.mp4",
                output_fps=24
            )