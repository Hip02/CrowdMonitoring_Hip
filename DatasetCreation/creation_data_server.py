import os
import sys
import numpy as np
import gc
from utils.utils import process_radar_file, process_video_file, save_radar_maps, save_video_frames

base_path = "/linux/hhilgers/Dataset"

source_data_path = "/linux/hhilgers/DATA_COLLECTION"
radar_data_path = os.path.join(source_data_path, "Radar")
video_data_path = os.path.join(source_data_path, "Video")

### NOUVEAU DATASET ###
n_exp = np.arange(1, 51)

antenna_i = 1

input_videos_filenames = [f"{video_data_path}/exp{i}.MOV" for i in n_exp]
input_radar_raw_filenames  = [f"{radar_data_path}/exp{i}.npz" for i in n_exp]
exp_names = [f"NewExp{i}" for i in n_exp]

crop_index = None

if crop_index == None: crop_index = len(input_videos_filenames)

for f, (input_video_filename, input_radar_raw_filename) in enumerate(zip(input_videos_filenames[:crop_index], input_radar_raw_filenames[:crop_index])):
    print(f"Processing {input_video_filename} and {input_radar_raw_filename}")

    radar_data = process_radar_file(input_radar_raw_filename, saveMagn=True, savePhase=True, iAntennaShow=antenna_i)
    timestamps = radar_data['timestamps']
    #video_data = process_video_file(input_video_filename, timestamps, saveFrames=True)

    # Load Labels
    labels_saved = np.load(f"{base_path}/{exp_names[f]}/Labels/labels.npy")
    n_labels = len(labels_saved)

    n_radar_maps = len(radar_data['magnitudes'])
    #n_labels = len(video_data['labels'])

    max_values = []
    min_values = []

    for i in range(min(n_radar_maps, n_labels)):
        #save_radar_maps(radar_data['magnitudes'][i], i, video_data['labels'][i], f"{base_path}/{exp_names[f]}/RadarMagnitudes")
        #save_radar_maps(radar_data['phases'][i], i, video_data['labels'][i], f"{base_path}/{exp_names[f]}/RadarPhases")
        save_radar_maps(radar_data['magnitudes'][i], i, labels_saved[i], f"{base_path}/{exp_names[f]}/RadarMagnitudesAntenna{antenna_i}")
        save_radar_maps(radar_data['phases'][i], i, labels_saved[i], f"{base_path}/{exp_names[f]}/RadarPhasesAntenna{antenna_i}")
        #save_video_frames(video_data['frames'][i], i, video_data['labels'][i], f"{base_path}/{exp_names[f]}/VideoFrames")
        #max_values.append(radar_data['magnitudes'][i].max())
        #min_values.append(radar_data['magnitudes'][i].min())

    # Save the labels as a numpy array into a new folder
    #os.makedirs(f"{base_path}/{exp_names[f]}/Labels", exist_ok=True)
    #np.save(f"{base_path}/{exp_names[f]}/Labels/labels.npy", np.array(video_data['labels']))

    # Save max values as a numpy array into a new folder
    #os.makedirs(f"{base_path}/{exp_names[f]}/MaxValues", exist_ok=True)
    #p.save(f"{base_path}/{exp_names[f]}/MaxValues/max_values.npy", np.array(max_values))

    # Save min values as a numpy array into a new folder
    #os.makedirs(f"{base_path}/{exp_names[f]}//MinValues", exist_ok=True)
    #np.save(f"{base_path}/{exp_names[f]}/MinValues/min_values.npy", np.array(min_values))

    # ✅ Libération de la mémoire après chaque vidéo
    del radar_data
    #del video_data
    del timestamps
    #del max_values
    #del min_values

    gc.collect()  # Forcer la récupération de mémoire