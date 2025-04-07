import os
import sys
import argparse
import numpy as np
import gc
from utils.utils import process_radar_file, process_video_file, save_radar_maps, save_video_frames

base_path = "/linux/hhilgers/Dataset"

source_data_path = "/linux/hhilgers/DATA_COLLECTION"
radar_data_path = os.path.join(source_data_path, "Radar")
video_data_path = os.path.join(source_data_path, "Video")

parser = argparse.ArgumentParser(description="Process radar and video files")
# from and to arguments (pour pouvoir paralléliser la création du dataset)
parser.add_argument("-f", "--fr", type=int, default=0)
parser.add_argument("-t", "--to", type=int, default=51)
args = parser.parse_args()

n_exp = np.arange(args.fr, args.to)

antenna_i = 0

# C'est pour crop les 3 premiers mètres du radar et les vitesses + extremes, je t'ai mis en False par défaut
cropCenter = False

input_videos_filenames = [f"{video_data_path}/exp{i}.MOV" for i in n_exp]
input_radar_raw_filenames  = [f"{radar_data_path}/exp{i}.npz" for i in n_exp]
exp_names = [f"NewExp{i}" for i in n_exp]

for f, (input_video_filename, input_radar_raw_filename) in enumerate(zip(input_videos_filenames, input_radar_raw_filenames)):
    print(f"Processing {input_video_filename} and {input_radar_raw_filename}")

    radar_data = process_radar_file(input_radar_raw_filename, saveMagn=True, savePhase=True, iAntennaShow=antenna_i, cropCenter=cropCenter)
    timestamps = radar_data['timestamps']
    video_data = process_video_file(input_video_filename, timestamps, saveFrames=True)

    # Load Labels (que quand tu as déjà créé les labels et tu veux rerun)
    # labels_saved = np.load(f"{base_path}/{exp_names[f]}/Labels/labels.npy")
    # n_labels = len(labels_saved)

    n_radar_maps = len(radar_data['magnitudes'])
    n_labels = len(video_data['labels'])

    max_values = []
    min_values = []

    for i in range(min(n_radar_maps, n_labels)):
        save_radar_maps(radar_data['magnitudes'][i], i, video_data['labels'][i], f"{base_path}/{exp_names[f]}/RadarPhasesAntenna{antenna_i}")
        save_radar_maps(radar_data['phases'][i], i, video_data['labels'][i], f"{base_path}/{exp_names[f]}/RadarPhases")
        save_video_frames(video_data['frames'][i], i, video_data['labels'][i], f"{base_path}/{exp_names[f]}/VideoFrames")
        max_values.append(radar_data['magnitudes'][i].max())
        min_values.append(radar_data['magnitudes'][i].min())

    # Save max values for antenna 1 as a numpy array into a new folder
    os.makedirs(f"{base_path}/{exp_names[f]}/MaxValues", exist_ok=True)
    np.save(f"{base_path}/{exp_names[f]}/MaxValues/max_values.npy", np.array(max_values))

    # Save min values for antenna 1 as a numpy array into a new folder
    os.makedirs(f"{base_path}/{exp_names[f]}/MinValues", exist_ok=True)
    np.save(f"{base_path}/{exp_names[f]}/MinValues/min_values.npy", np.array(min_values))

    # Save the labels as a numpy array into a new folder
    os.makedirs(f"{base_path}/{exp_names[f]}/Labels", exist_ok=True)
    np.save(f"{base_path}/{exp_names[f]}/Labels/labels.npy", np.array(video_data['labels']))

    # Libération de la mémoire après chaque vidéo (pas nécessaire mais pour être sur de ne pas saturer la mémoire)
    del radar_data
    del video_data
    del timestamps
    del max_values
    del min_values

    gc.collect()  # Forcer la récupération de mémoire