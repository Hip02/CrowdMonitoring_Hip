import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.solutions import object_counter
from PIL import Image
import os
import cv2

def fmcwProcess(filename, osFactor, indexFrameProcess=None, clrRF=True, disp=True):
    """
    Input:
        - filename: name of the radar raw file
        - osFactor: oversampling factor
        - indexFrameProcess: list of indexes of frames to process
        - clrRF: remove RF coupling
        - disp: display processing progress
    Output:
        dict(
            - rangeSpeedMap: radar map
            - ranges: ranges
            - speeds: speeds
            - raw: raw data
            - timestamps: timestamps of the radar maps
            - chirp: chirp data
            - info: information on the sample
        )
    """
    
    ##### LOAD SIGNAL DATA
    fileData = np.load(filename)
    
    ## Load Signals
    #   - from bin to millivolts
    signals_raw = fileData['data'] * fileData['chan_ranges_mv'][:, np.newaxis] / 2**16
    nFrame, nChan, M = signals_raw.shape
    nRX = nChan//2
    #   - from channels to Baseband IQ
    chanIQ  = np.arange(nChan).reshape(nChan//2, 2)
    signals = signals_raw.take(chanIQ[:, 0], axis=-2) + 1j * signals_raw.take(chanIQ[:, 1], axis=-2)
    
    ## Chirp data
    (f0, B, Ms, Mc, Ts, Tc) = fileData['chirp']
    cel = 299792458
    Ms = int(Ms)
    Mc = int(Mc)
    #   - Total number of samples per pulse repetition period (including those taken in pauses)
    Ms2 = int(np.round((Tc+1e-10)/Ts))
    #   - Resolution and max ranges
    rResol = cel / (2*B)
    rMax   = Ms * rResol
    #   - Resolution and max speeds
    uResol = cel / (2*f0*Mc*Tc)
    uMax   = Mc * uResol
    
    
    ##### PROCESSING PARAMETERS
    # Oversampling (in the Fourier domain) factors for the range (R) and the speed (U)
    osR, osU = osFactor, osFactor
    # Compute useful quantities
    Nr = int(osR * Ms) # Number of samples in the Fourier domain in the range dimension
    Nu = int(osU * Mc) # Number of samples in the Fourier domain in the speed dimension
    # Generate grids of ranges and speeds
    ranges = np.arange(Nr) / osR * rResol
    speeds = np.arange(-Nu/2, Nu/2) / osU * uResol
    
    
    ##### INITIALIZE
    # Frame to process
    if indexFrameProcess is None:
        indexFrameProcess = np.arange(nFrame)
    nFrameProcess = len(indexFrameProcess)

    # Allocate mem
    rangeSpeedMap = np.empty((nFrameProcess, nRX, Nr, Nu), dtype="complex")
    
    ##### PROCESS
    for i, iFrame in enumerate(indexFrameProcess):
        if disp and len(indexFrameProcess) > 20:
            print("\r Frame iD {:>4d}/{:d}".format(iFrame, len(indexFrameProcess)), end="")
    
        for iRX in range(nRX):
            # Format signal into a matrix and discard unwanted samples
            sig = signals[iFrame, iRX]
            y = np.reshape(sig, (Mc, Ms2))
            y = y[:, :Ms].T
            
            if clrRF:
                # Remove RF coupling (removes zero-speed targets too)
                y = (y.T-np.mean(y, axis=1)).T

                # Remov

            # In range-speed domain
            Y = np.fft.ifft2(y, s=(Nr, Nu))
            Y = np.fft.fftshift(Y, axes=1)
            
            rangeSpeedMap[iFrame, iRX] = Y
            
    if disp and len(indexFrameProcess) > 20:
        print()
        
        
    ##### Information on the sample
    info =  'f0 = %2.2f [GHz]'%(f0*1e-9)+' , B = %4.4f [MHz]'%(B*1e-6)\
            +' ,  %d chirps %3.4f ms long with %d samples acquired per chirp. (sampling freq. = %4.4f kHz)'%(Mc, 1000*Tc, Ms, 1e-3/Ts)\
            +' ,   Total time : %3.3f ms'%(1000*Mc*Tc)\
            +'\nRange Resolution = %3.4f'%(rResol) +'[m] , max Range = %3.4f [m]'%(rMax)\
            +'\nSpeed Resolution = %3.4f'%(uResol) +'[m/s] , max Speed= +- %3.4f'%(uMax/2) +'[m/s] (%3.4f [km / h])'%(3.6 * uMax/2)
    
    
    return {
        "rsmap"         : rangeSpeedMap,
        "ranges"        : ranges,
        "speeds"        : speeds,
        "raw"           : fileData['data'],
        "timestamps"    : fileData["data_times"][indexFrameProcess],
        "chirp"         : fileData['chirp'],
        "info"          : info
    }

def process_radar_file(input_radar_raw_filename, indexFrameProcess=None, saveMagn=True, savePhase=False, osFactor=8, clrRF=True, iAntennaShow=0):
    """
    Input:
        - input_radar_raw_filename: name of the radar raw file
        - indexFrameProcess: list of indexes of frames to process
        - saveMagn: save magnitude of the radar map
        - savePhase: save phase of the radar map
        - osFactor: oversampling factor
        - clrRF: remove RF coupling
        - iAntennaShow: antenna to show
    Output:
        dict(
            - magnitudes: magnitude images created of the radar maps (empty if saveMagn is False)
            - phases: phase images created of the radar maps (empty if savePhase is False)
            - timestamps: timestamps of the radar maps
        )
    """

    rsData = fmcwProcess(input_radar_raw_filename, osFactor, indexFrameProcess=indexFrameProcess, clrRF=clrRF, disp=True)

    timestamps = rsData["timestamps"]
    magnitudes = []
    phases = []

    if indexFrameProcess is None:
        indexFrameProcess = np.arange(0, len(timestamps))
    
    for i in indexFrameProcess:
        frameDataToSave = rsData['rsmap'][i, iAntennaShow]
        
        if saveMagn:
            # Keep only magnitude
            frameDataToSaveAbs = np.abs(frameDataToSave)

            magnitudes.append(frameDataToSaveAbs)
        
        if savePhase:
            # Keep only phase
            frameDataToSavePhase = np.angle(frameDataToSave)

            phases.append(frameDataToSavePhase)

    return {
        "magnitudes": magnitudes,
        "phases": phases,
        "timestamps": timestamps
    }

def save_radar_maps(radar_map, number, label, output_folder):

    """
    Input:
        - radar_maps: radar map 512x512 array to save in a folder
        - output_folder: folder to save the radar maps
    Effect:
        Save the radar map in the output folder as "map_i.png"
    """

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Normalize radar map to 0-255 (if not already in that range)
    radar_map = np.array(radar_map, dtype=np.float32)
    radar_map = (255 * (radar_map - radar_map.min()) / (radar_map.max() - radar_map.min())).astype(np.uint8)

    # Convert to PIL image
    image = Image.fromarray(radar_map)

    # Save image without compression
    filename = os.path.join(output_folder, f"map_{number}_{label}.png")
    image.save(filename, format="PNG", compress_level=0)

def save_radar_max_values(max_values, timestamps, output_folder):
    """
    Input:
        - max_values: list of maximum absolute values of the radar maps
        - timestamps: timestamps of the radar maps
        - output_folder: folder to save the radar maps
    Effect:
        Save the radar map in the output folder as "max_values.txt"
    """

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save max values to a text file
    with open(os.path.join(output_folder, "max_values.txt"), "w") as f:
        for i, (max_value, timestamp) in enumerate(zip(max_values, timestamps)):
            f.write(f"{i}: {max_value} at {timestamp}\n")

def process_video_file(input_video_filename, timestamps, saveFrames=True):
    """
    Input:
        - input_video_filename: name of the video file
        - timestamps: timestamps of the radar maps
        - saveFrames: save the frames of the video
    Output:
        dict(
            - frames: list of annoted frames of the video (empty if saveFrames is False)
            - labels: list number of people in each frame
        )
    """

    """
    YOLOv8n	Très petit	    Moins précis	        Très rapide ⚡
    YOLOv8s	Petit	        Bonne précision	        Rapide ⚡
    YOLOv8m	Moyen	        Très bonne précision	Moyen
    YOLOv8l	Grand	        Excellente précision	Lent!
    YOLOv8x	Très grand	    Ultra précis	        Très lent !!
    """

    model = YOLO("yolov8l.pt")

    frames = []
    labels = []

    video_path = os.path.expanduser(input_video_filename)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Compute indexes of frames corresponding (approximate) to the timestamps
    frames_useful = list(dict.fromkeys([round(fps * t) for t in timestamps]))

    frame_number = 0
    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            break

        if frame_number in frames_useful:

            results = model.track(frame, persist=True, show=False, classes=[0])
 
            if saveFrames:
                annotated_frame = results[0].plot()
                frames.append(annotated_frame)
            
            count_people = 0
            for t in results:
                for b in t.boxes:
                    if b.cls == 0 : # == "person"
                        count_people += 1
            
            labels.append(count_people)

        frame_number += 1

    cap.release()

    return {
        "frames": frames,
        "labels": labels
    }

def process_video_file_wide_angle(input_video_filename, timestamps, saveFrames=True):
    """
    Process a video recorded in wide-angle mode (0.5x on iPhone).
    It corrects the lens distortion before detecting people with YOLO.

    Input:
        - input_video_filename: name of the video file
        - timestamps: timestamps of the radar maps
        - saveFrames: save the frames of the video
    Output:
        dict(
            - frames: list of annotated frames of the video (empty if saveFrames is False)
            - labels: list of the number of people in each frame
        )
    """
    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Initialize lists
    frames = []
    labels = []

    # Check if the video exists
    video_path = os.path.expanduser(input_video_filename)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Compute frame indices corresponding to timestamps
    frames_useful = list(dict.fromkeys([round(fps * t) for t in timestamps]))

    frame_number = 0

    # Camera intrinsic parameters (iPhone 0.5x default values, adjust if necessary)
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])  # Approximate focal length and center
    D = np.array([-0.3, 0.1, 0, 0, 0])  # Approximate distortion coefficients

    # Get frame size
    ret, test_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Could not read video frame.")
    
    h, w = test_frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_number in frames_useful:
            # Apply lens distortion correction
            frame_undistorted = cv2.undistort(frame, K, D, None, new_K)

            # Detect people with YOLO
            results = model.track(frame_undistorted, persist=True, show=False)

            if saveFrames:
                annotated_frame = results[0].plot()
                frames.append(annotated_frame)

            count_people = 0
            for t in results:
                for b in t.boxes:
                    if b.cls == 0:  # Person class
                        count_people += 1

            labels.append(count_people)

        frame_number += 1

    cap.release()

    return {
        "frames": frames,
        "labels": labels
    }

def save_video_frames(video_frame, number, label, output_folder):
    """
    Input:
        - video_frames: video frame to save in a folder
        - output_folder: folder to save the video frames
    Effect:
        Save the video frame in the output folder as "frame_i.png"
    """

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Convertir l'image OpenCV (BGR) en format Pillow (RGB)
    image_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

    Image.fromarray(image_rgb).save(f"{output_folder}/frame_{number}_{label}.jpg", quality=50)

