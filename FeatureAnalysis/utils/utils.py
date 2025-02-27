"""
Remarque: La majorité des fonctions de la classe ont été générés/revues par ChatGPT et GitHub Copilot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks
import cv2
from utils import radar_toolbox as rtb


class DataLoader:
    def __init__(self, base_path, exp_list=None, to_load=None):
        """
        Initializes the DataLoader with the database path and specific experiments to load.

        Args:
            base_path (str): Path to the directory containing experiment data.
            exp_list (list, optional): List of experiments to load (e.g., ["Exp1", "Exp3"]).
                                       If None, it loads all available experiments.
        """
        self.base_path = base_path
        self.exp_list = exp_list if exp_list else self._discover_experiments()
        self.data = {
            "min_values": {}, "max_values": {}, "labels": {},
            "magnitudes": {}, "phases": {}, "fft": {}, "video_frames": {}, "features": {}
        }

        # Load data for selected experiments
        self._load_data(to_load)

    def _discover_experiments(self):
        """Automatically detects available experiments in the database."""
        return [exp for exp in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, exp))]

    def _load_data(self, to_load):

        if to_load is None:
            to_load = ["max_values", "labels", "features"]

        """Loads data for each selected experiment."""
        for exp in self.exp_list:
            print(f"🔄 Loading data for {exp}...")
            if "min_values" in to_load: 
                self.data["min_values"][exp] = self._load_min_values(exp)
            if "max_values" in to_load: 
                self.data["max_values"][exp] = self._load_max_values(exp)
            if "labels" in to_load: 
                self.data["labels"][exp] = self._load_labels(exp)
            if "magnitudes" in to_load: 
                self.data["magnitudes"][exp] = self._load_magnitudes(exp)
            if "phases" in to_load: 
                self.data["phases"][exp] = self._load_phases(exp)
            if "fft" in to_load: 
                self.data["fft"][exp] = self._load_fft(exp)
            if "video_frames" in to_load: 
                self.data["video_frames"][exp] = self._load_video_frames(exp)
            if "features" in to_load: 
                self.data["features"][exp] = self._load_features(exp)  # Dynamically load all features

        print("✅ Loading complete!")

    def _load_min_values(self, exp_name):
        """Loads min values from a single file."""
        min_path = os.path.join(self.base_path, exp_name, "MinValues", "min_values.npy")
        return np.load(min_path) if os.path.exists(min_path) else np.array([])

    def _load_max_values(self, exp_name):
        """Loads max values from a single file."""
        max_path = os.path.join(self.base_path, exp_name, "MaxValues", "max_values.npy")
        return np.load(max_path) if os.path.exists(max_path) else np.array([])

    def _load_labels(self, exp_name):
        """Loads labels from a single file."""
        labels_path = os.path.join(self.base_path, exp_name, "Labels", "labels.npy")
        return np.load(labels_path) if os.path.exists(labels_path) else np.array([])

    def _load_magnitudes(self, exp_name):
        """Loads all magnitude data for the experiment."""
        mag_path = os.path.join(self.base_path, exp_name, "RadarMagnitudes")
        return self._load_radar_maps(mag_path)

    def _load_phases(self, exp_name):
        """Loads all radar phase data for the experiment."""
        phase_path = os.path.join(self.base_path, exp_name, "RadarPhases")
        return self._load_radar_maps(phase_path)
    
    def _load_fft(self, exp_name):
        """Loads all FFT data for the experiment."""
        fft_path = os.path.join(self.base_path, exp_name, "FFT")
        return self._load_radar_maps(fft_path)

    def _load_video_frames(self, exp_name):
        """Loads all video frames for the experiment."""
        video_path = os.path.join(self.base_path, exp_name, "VideoFrames")
        return self._load_image_files(video_path)

    def _load_features(self, exp_name):
        """
        Dynamically loads all features from the Features folder if it exists.
        It automatically detects all .npy feature files inside.

        Args:
            exp_name (str): Name of the experiment.

        Returns:
            dict: A dictionary containing all feature arrays.
        """
        features_path = os.path.join(self.base_path, exp_name, "Features")
        features = {}

        if os.path.exists(features_path):
            for root, _, files in os.walk(features_path):
                for file in files:
                    if file.endswith(".npy"):
                        feature_name = os.path.splitext(file)[0]  # Extract feature name
                        file_path = os.path.join(root, file)
                        features[feature_name] = np.load(file_path)

        return features

    def _load_numpy_files(self, directory):
        """Loads all .npy files from a given directory into a single NumPy array."""
        data = []
        if os.path.exists(directory):
            for file in sorted(os.listdir(directory)):
                file_path = os.path.join(directory, file)
                if file.endswith(".npy"):
                    data.append(np.load(file_path))
        return np.concatenate(data) if data else np.array([])

    def _load_image_files(self, directory):
        """
        Loads all .jpg and .jpeg video frames from a given directory into a NumPy array.
        Frames are loaded in RGB format.

        Args:
            directory (str): Path to the folder containing video frames.

        Returns:
            np.ndarray: Array of video frames with shape (num_frames, height, width, 3) or empty array if none exist.
        """
        data = []
        if os.path.exists(directory):
            for file in sorted(os.listdir(directory)):
                file_path = os.path.join(directory, file)
                if file.endswith((".jpg", ".jpeg")):
                    img = cv2.imread(file_path)  # Load in BGR format
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                        data.append(img)
        return np.array(data) if data else np.array([])

    def _load_radar_maps(self, directory):
        """
        Loads all .png radar maps from a given directory into a NumPy array.
        Radar maps are loaded in grayscale with shape (512, 512, 1).

        Args:
            directory (str): Path to the folder containing radar maps.

        Returns:
            np.ndarray: Array of radar maps with shape (num_maps, 512, 512, 1) or empty array if none exist.
        """
        data = []
        if os.path.exists(directory):
            for file in sorted(os.listdir(directory)):
                file_path = os.path.join(directory, file)
                if file.endswith(".png"):
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
                    if img is not None:
                        img = np.expand_dims(img, axis=-1)  # Add channel dimension (512, 512, 1)
                        data.append(img)
        return np.array(data) if data else np.array([])

    def _get_combined_data(self, data_dict):
        """
        Combines all experiments' data into a single NumPy array, ensuring the order is alphabetical.

        Args:
            data_dict (dict): A dictionary where keys are experiment names and values are NumPy arrays.

        Returns:
            np.ndarray: Concatenated NumPy array of all experiments' data in alphabetical order.
        """
        # Sort experiment names alphabetically
        sorted_experiments = sorted(data_dict.keys())

        # Collect data in sorted order
        all_data = [data_dict[exp] for exp in sorted_experiments if data_dict[exp].size > 0]

        return np.concatenate(all_data) if all_data else np.array([])


    def get_min_values(self, exp_name=None):
        """Retrieves min values, either for a specific experiment or combined."""
        return self.data["min_values"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["min_values"])

    def get_max_values(self, exp_name=None):
        """Retrieves max values, either for a specific experiment or combined."""
        return self.data["max_values"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["max_values"])

    def get_labels(self, exp_name=None):
        """Retrieves labels, either for a specific experiment or combined."""
        return self.data["labels"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["labels"])

    def get_magnitudes(self, exp_name=None):
        """Retrieves magnitudes, either for a specific experiment or combined."""
        return self.data["magnitudes"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["magnitudes"])

    def get_phases(self, exp_name=None):
        """Retrieves radar phases, either for a specific experiment or combined."""
        return self.data["phases"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["phases"])

    def get_fft(self, exp_name=None):
        """Retrieves FFT magnitudes, either for a specific experiment or combined."""
        return self.data["fft"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["fft"])

    def get_video_frames(self, exp_name=None):
        """Retrieves video frames, either for a specific experiment or combined."""
        return self.data["video_frames"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["video_frames"])

    def get_feature(self, feature_name, exp_name=None):
        """Retrieves specific feature (feature_name), either for a specific experiment or combined."""
        if exp_name:
            # Return the feature from a specific experiment
            return self.data["features"].get(exp_name, {}).get(feature_name, np.array([]))

        # Return the feature for all experiments as a dictionary {exp_name: feature_data}
        all_features = {
            exp: features.get(feature_name, np.array([]))
            for exp, features in self.data["features"].items()
        }
        return self._get_combined_data(all_features)

    def _compute_feature_per_experiment(self, feature_name, feature_func, exp_name=None, save_path=None, load_func=None):
        """
        Generic function to compute and save a feature for a specific experiment or all experiments.

        Args:
            feature_name (str): Name of the feature to compute (e.g., "MeanMagnitudes").
            feature_func (function): A function that computes the feature given (magnitudes, max_values).
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed feature arrays.
                                       Defaults to 'Features/{feature_name}' inside each experiment.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of computed values for that experiment.
                - If None, returns a dictionary with computed feature arrays for all experiments.
        """
        if load_func is None:
            load_func = self.get_magnitudes

        if save_path is None:
            save_path = self.base_path

        computed_feature_dict = {}

        # Determine which experiments to process
        experiments_to_process = [exp_name] if exp_name else self.exp_list

        for exp in experiments_to_process:
            print(f"🔄 Computing {feature_name} for {exp}...")

            # Retrieve all radar magnitude maps for this experiment
            magnitudes = load_func(exp)

            # Retrieve the max values for each individual map
            max_values_per_map = self.get_max_values(exp)  # Should be (num_maps,)

            # Check if data exists
            if magnitudes.size == 0 or max_values_per_map.size == 0:
                print(f"⚠️ Skipping {exp} (no magnitudes or max values found).")
                continue

            # Compute the feature using the provided function
            computed_feature = feature_func(magnitudes, max_values_per_map)

            # Define output path and save as NumPy array
            output_dir = os.path.join(save_path, exp, 'Features', feature_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{feature_name}.npy")
            np.save(output_path, computed_feature)

            # Store in dictionary
            computed_feature_dict[exp] = computed_feature

            print(f"✅ {feature_name} for {exp} saved at {output_path}")

        print(f"🎉 {feature_name} computation complete!")

        # Return only one array if a single experiment was requested
        if exp_name:
            return computed_feature_dict.get(exp_name, None)

        return computed_feature_dict

    def compute_mean_magnitudes(self, exp_name=None, save_path=None):
        """
        Computes the real mean intensity of each radar magnitude map for a specific experiment or all experiments.
        The mean value is scaled by dividing each map by its corresponding max value.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed mean magnitude arrays.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of real mean values.
                - If None, returns a dictionary with all experiments' real mean magnitude arrays.
        """
        def mean_magnitude_function(magnitudes, max_values_per_map):
            """Function to compute mean magnitudes, normalized by their max values."""
            return np.array([np.mean(mag / 255 * max_val) for mag, max_val in zip(magnitudes, max_values_per_map)])

        return self._compute_feature_per_experiment(
            feature_name="MeanMagnitudes",
            feature_func=mean_magnitude_function,
            exp_name=exp_name,
            save_path=save_path
        )

    def compute_std_magnitudes(self, exp_name=None, save_path=None):
        """
        Computes the standard deviation of each radar magnitude map for a specific experiment or all experiments.
        The standard deviation is scaled by dividing each map by its corresponding max value.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed standard deviation arrays.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of standard deviation values.
                - If None, returns a dictionary with all experiments' standard deviation arrays.
        """
        def std_magnitude_function(magnitudes, max_values_per_map):
            """Function to compute the standard deviation of magnitudes, normalized by their max values."""
            return np.array([np.std(mag / 255 * max_val) for mag, max_val in zip(magnitudes, max_values_per_map)])

        return self._compute_feature_per_experiment(
            feature_name="StdMagnitudes",
            feature_func=std_magnitude_function,
            exp_name=exp_name,
            save_path=save_path
        )
    
    def compute_median_magnitudes(self, exp_name=None, save_path=None):
        """
        Computes the median intensity of each radar magnitude map for a specific experiment or all experiments.
        The median value is scaled by dividing each map by its corresponding max value.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed median magnitude arrays.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of median values.
                - If None, returns a dictionary with all experiments' median magnitude arrays.
        """
        def median_magnitude_function(magnitudes, max_values_per_map):
            """Function to compute median magnitudes, normalized by their max values."""
            return np.array([np.median(mag / 255 * max_val) for mag, max_val in zip(magnitudes, max_values_per_map)])

        return self._compute_feature_per_experiment(
            feature_name="MedianMagnitudes",
            feature_func=median_magnitude_function,
            exp_name=exp_name,
            save_path=save_path
        )
    
    def compute_skewness_magnitudes(self, exp_name=None, save_path=None):
        """
        Computes the skewness of each radar magnitude map for a specific experiment or all experiments.
        The skewness value is scaled by dividing each map by its corresponding max value.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed skewness magnitude arrays.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of skewness values.
                - If None, returns a dictionary with all experiments' skewness magnitude arrays.
        """
        def skewness_magnitude_function(magnitudes, max_values_per_map):
            """
            Compute the skewness of normalized magnitudes using SciPy with memory optimization.

            Args:
                magnitudes (np.ndarray): 4D array (batch, H, W, 1) of magnitude values from radar data.
                max_values_per_map (np.ndarray): 1D array (batch) of max values per Doppler map.

            Returns:
                np.ndarray: Skewness values for each Doppler map.
            """
            # Convert to float32 for memory efficiency
            magnitudes = magnitudes.astype(np.float32)
            max_values_per_map = max_values_per_map.astype(np.float32)

            # Ensure max_values_per_map has shape (batch, 1, 1, 1) for broadcasting
            max_values_per_map = max_values_per_map.reshape(-1, 1, 1, 1)

            # Normalize efficiently
            normalized_magnitudes = np.where(
                max_values_per_map > 1e-10, magnitudes / max_values_per_map, 1e-10
            )

            # Compute skewness using SciPy (optimized C implementation)
            return np.array([skew(img.flatten(), bias=False) for img in normalized_magnitudes])


        return self._compute_feature_per_experiment(
            feature_name="SkewnessMagnitudes",
            feature_func=skewness_magnitude_function,
            exp_name=exp_name,
            save_path=save_path
        )
    
    def compute_kurtosis_magnitudes(self, exp_name=None, save_path=None):
        """
        Computes the kurtosis of each radar magnitude map for a specific experiment or all experiments.
        The skewness value is scaled by dividing each map by its corresponding max value.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed kurtosis magnitude arrays.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of kurtosis values.
                - If None, returns a dictionary with all experiments' kurtosis magnitude arrays.
        """
        def kurtosis_magnitude_function(magnitudes, max_values_per_map):
            """
            Compute the kurtosis of normalized magnitudes using SciPy for optimization.

            Args:
                magnitudes (np.ndarray): 2D or 3D array of magnitude values from radar data.
                max_values_per_map (np.ndarray): Array containing the max value per map for normalization.

            Returns:
                float: Kurtosis of the normalized magnitudes.
            """

            # Convert to float32 for memory efficiency
            magnitudes = magnitudes.astype(np.float32)
            max_values_per_map = max_values_per_map.astype(np.float32)

            # Ensure max_values_per_map has shape (batch, 1, 1, 1) for broadcasting
            max_values_per_map = max_values_per_map.reshape(-1, 1, 1, 1)

            # Normalize efficiently
            normalized_magnitudes = np.where(
                max_values_per_map > 1e-10, magnitudes / max_values_per_map, 1e-10
            )

            # Compute kurtosis using SciPy (optimized C implementation)
            return np.array([kurtosis(img.flatten(), bias=False, fisher=True) for img in normalized_magnitudes])

        return self._compute_feature_per_experiment(
            feature_name="KurtosisMagnitudes",
            feature_func=kurtosis_magnitude_function,
            exp_name=exp_name,
            save_path=save_path
        )


    def compute_entropy_magnitudes(self, exp_name=None, save_path=None):
        """
        Computes the entropy of each radar magnitude map for a specific experiment or all experiments.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed entropy magnitude arrays.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of entropy values.
                - If None, returns a dictionary with all experiments' entropy magnitude arrays.
        """
        def entropy_magnitude_function(magnitudes, max_values_per_map, num_bins=256):
            """
            Compute the Shannon entropy of normalized magnitudes using a histogram.

            Args:
                magnitudes (np.ndarray): 4D array (batch, H, W, 1) of magnitude values from radar data.
                max_values_per_map (np.ndarray): 1D array (batch) of max values per Doppler map.
                num_bins (int): Number of bins for the histogram (default: 256).

            Returns:
                np.ndarray: Entropy values for each Doppler map.
            """

            # Convert to float32 to reduce memory usage
            magnitudes = magnitudes.astype(np.float32)

            # Compute global min and max per image
            min_values = np.min(magnitudes, axis=(1, 2, 3), keepdims=True)
            max_values = np.max(magnitudes, axis=(1, 2, 3), keepdims=True)

            # Apply Min-Max normalization
            normalized_magnitudes = (magnitudes - min_values) / (max_values - min_values + 1e-10)


            # Compute entropy for each Doppler map
            entropy_values = []
            for img in normalized_magnitudes:
                # Flatten image and compute histogram
                hist, _ = np.histogram(img.flatten(), bins=num_bins, range=(0, 1), density=True)
                
                # Compute entropy (Shannon)
                entropy_value = entropy(hist, base=2)
                entropy_values.append(entropy_value)

            return np.array(entropy_values)

        return self._compute_feature_per_experiment(
            feature_name="EntropyMagnitudes",
            feature_func=entropy_magnitude_function,
            exp_name=exp_name,
            save_path=save_path
        )
    
    def compute_spectral_entropy_magnitudes(self, exp_name=None, save_path=None, num_bins=256, epsilon=1e-10):
            """
            Computes the spectral entropy using the precomputed FFTs for a specific experiment or all experiments.

            Args:
                exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
                save_path (str, optional): Path to save the computed spectral entropy magnitude arrays.
                num_bins (int): Number of bins for the histogram (default: 256).
                epsilon (float): Small constant to avoid division by zero.

            Returns:
                dict or np.ndarray: 
                    - If exp_name is specified, returns a 1D NumPy array of spectral entropy values.
                    - If None, returns a dictionary with all experiments' spectral entropy magnitude arrays.
            """
            def spectral_entropy_magnitude_function(fft_magnitudes, max_values_per_map):
                """
                Compute spectral entropy from precomputed FFT magnitude maps.

                Args:
                    fft_magnitudes (np.ndarray): 4D array (batch, H, W) of precomputed FFT magnitude maps.

                Returns:
                    np.ndarray: Spectral entropy values for each FFT magnitude map.
                """
                # Ensure input is float32
                fft_magnitudes = fft_magnitudes.astype(np.float32)

                # Compute global min and max per image
                min_values = np.min(fft_magnitudes, axis=(1, 2, 3), keepdims=True)
                max_values = np.max(fft_magnitudes, axis=(1, 2, 3), keepdims=True)

                # Apply Min-Max normalization
                normalized_fft_magnitudes = (fft_magnitudes - min_values) / (max_values - min_values + 1e-10)

                # Compute entropy for each Doppler map
                spectral_entropy_values = []

                for img in normalized_fft_magnitudes:
                    # Flatten image and compute histogram
                    hist, _ = np.histogram(img.flatten(), bins=num_bins, range=(0, 1), density=True)
                    
                    # Compute entropy (Shannon)
                    spectral_entropy_value = entropy(hist, base=2)
                    spectral_entropy_values.append(spectral_entropy_value)

                return np.array(spectral_entropy_values)

            return self._compute_feature_per_experiment(
                feature_name="SpectralEntropyMagnitudes",
                feature_func=spectral_entropy_magnitude_function,
                exp_name=exp_name,
                save_path=save_path,
                load_func=self.get_fft  # Use precomputed FFT loader
            )
    
    def compute_peak_count_magnitudes(self, exp_name=None, save_path=None, threshold=1e-4, min_distance=3, relative=True):
        """
        Computes the number of peaks in each radar magnitude map for a specific experiment or all experiments.

        The peak count is determined using `scipy.signal.find_peaks`, considering:
        - `relative_threshold`: Minimum height relative to the max value of each map.
        - `min_distance`: Minimum pixel distance between two peaks.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed peak count arrays.
            relative_threshold (float): Minimum height threshold as a fraction of the map's max value.
            min_distance (int): Minimum required distance between peaks.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of peak counts.
                - If None, returns a dictionary with all experiments' peak count arrays.
        """

        def peak_count_function(magnitudes, max_values_per_map):
            """
            Function to count peaks in normalized magnitude maps.

            Args:
                magnitudes (np.ndarray): 4D array (batch, H, W, 1) of magnitude values from radar data.
                max_values_per_map (np.ndarray): 1D array (batch) of max values per Doppler map.

            Returns:
                np.ndarray: Array of peak counts per map.
            """
            # Convert to float32 for memory efficiency
            magnitudes = magnitudes.astype(np.float32)
            max_values_per_map = max_values_per_map.astype(np.float32)

            # Ensure max_values_per_map has shape (batch, 1, 1, 1) for broadcasting
            max_values_per_map = max_values_per_map.reshape(-1, 1, 1, 1)

            # Normalize efficiently
            normalized_magnitudes = np.where(
                max_values_per_map > 1e-10, (magnitudes / 255) * max_values_per_map, 1e-10
            )

            # Count peaks in each map
            peak_counts = []
            for img in normalized_magnitudes:
                # Flatten image to 1D for peak detection (considering projection on one axis)
                img_1d = np.mean(img, axis=1).flatten()  # Mean projection over one axis

                

                if relative == True : 
                    # Set threshold relative to the max value
                    min_height = threshold * np.max(img_1d)
                else :
                    min_height = threshold
                
                peaks, _ = find_peaks(img_1d, height=min_height, distance=min_distance)

                # Store peak count
                peak_counts.append(len(peaks))

            return np.array(peak_counts)

        return self._compute_feature_per_experiment(
            feature_name="PeakCountMagnitudes",
            feature_func=peak_count_function,
            exp_name=exp_name,
            save_path=save_path
        )
    
    def compute_cfar_peak_count_magnitudes(self, exp_name=None, save_path=None, absolute_threshold=1e-4, min_distance=5):
        """
        Computes the number of peaks in each radar magnitude map for a specific experiment or all experiments.

        The peak count is determined using `scipy.signal.find_peaks`, considering:
        - `relative_threshold`: Minimum height relative to the max value of each map.
        - `min_distance`: Minimum pixel distance between two peaks.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed peak count arrays.
            relative_threshold (float): Minimum height threshold as a fraction of the map's max value.
            min_distance (int): Minimum required distance between peaks.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of peak counts.
                - If None, returns a dictionary with all experiments' peak count arrays.
        """

        def cfar_peak_count_function(magnitudes, max_values_per_map):
            """
            Function to count peaks in normalized magnitude maps.

            Args:
                magnitudes (np.ndarray): 4D array (batch, H, W, 1) of magnitude values from radar data.
                max_values_per_map (np.ndarray): 1D array (batch) of max values per Doppler map.

            Returns:
                np.ndarray: Array of peak counts per map.
            """
            # Convert to float32 for memory efficiency
            magnitudes = magnitudes.astype(np.float32)
            max_values_per_map = max_values_per_map.astype(np.float32)

            # Ensure max_values_per_map has shape (batch, 1, 1, 1) for broadcasting
            max_values_per_map = max_values_per_map.reshape(-1, 1, 1, 1)

            # Normalize efficiently
            normalized_magnitudes = np.where(
                max_values_per_map > 1e-10, (magnitudes / 255) * max_values_per_map, 1e-10
            )

            # Count peaks in each map
            peak_counts = []
            for i,img in enumerate(normalized_magnitudes):

                # Find peaks using SciPy
                P_FA = 1e-3
                Nl = np.array([10, 10])  # Training cells (left & right)
                Nr = np.array([10, 10])
                Gl = np.array([4, 4])    # Guard cells
                Gr = np.array([4, 4])
                img = img.squeeze()

                # Pool the 512x512 image to 128x128
                #img = img.reshape(128, 4, 128, 4).mean(-1).mean(1)

                thresh_map, binary_map = rtb.CFAR_detector(img, P_FA, Nl, Nr, Gl, Gr, kind="OS")
                print(f"{i+1}/{len(normalized_magnitudes)}")

                # Count number of 1 in the binary map
                count = np.sum(binary_map)
                print(f"count: {count}")
                peak_counts.append(count)

            return np.array(peak_counts)

        return self._compute_feature_per_experiment(
            feature_name="CFARPeakCountMagnitudes",
            feature_func=cfar_peak_count_function,
            exp_name=exp_name,
            save_path=save_path
        )
    
    def compute_impulse_factor_magnitudes(self, exp_name=None, save_path=None):
        """
        Computes the impulse factor for each radar magnitude map for a specific experiment or all experiments.
        The impulse factor is defined as the ratio of the max value to the mean absolute value.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed impulse factor arrays.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of impulse factor values.
                - If None, returns a dictionary with all experiments' impulse factor arrays.
        """
        def impulse_factor_function(magnitudes, max_values_per_map):
            """
            Compute the impulse factor of normalized magnitudes.

            Args:
                magnitudes (np.ndarray): 4D array (batch, H, W, 1) of magnitude values.
                max_values_per_map (np.ndarray): 1D array (batch) of max values per Doppler map.

            Returns:
                np.ndarray: Impulse factor values for each Doppler map.
            """
            magnitudes = magnitudes.astype(np.float32)
            max_values_per_map = max_values_per_map.astype(np.float32).reshape(-1, 1, 1, 1)

            normalized_magnitudes = np.where(
                max_values_per_map > 1e-10, magnitudes / max_values_per_map, 1e-10
            )

            return np.array([np.max(img) / (np.mean(np.abs(img)) + 1e-10) for img in normalized_magnitudes])

        return self._compute_feature_per_experiment(
            feature_name="ImpulseFactorMagnitudes",
            feature_func=impulse_factor_function,
            exp_name=exp_name,
            save_path=save_path
        )

    def compute_crest_factor_magnitudes(self, exp_name=None, save_path=None):
        """
        Computes the crest factor for each radar magnitude map for a specific experiment or all experiments.
        The crest factor is defined as the ratio of the max value to the RMS value.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed crest factor arrays.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of crest factor values.
                - If None, returns a dictionary with all experiments' crest factor arrays.
        """
        def crest_factor_function(magnitudes, max_values_per_map):
            """
            Compute the crest factor of normalized magnitudes.

            Args:
                magnitudes (np.ndarray): 4D array (batch, H, W, 1) of magnitude values.
                max_values_per_map (np.ndarray): 1D array (batch) of max values per Doppler map.

            Returns:
                np.ndarray: Crest factor values for each Doppler map.
            """
            magnitudes = magnitudes.astype(np.float32)
            max_values_per_map = max_values_per_map.astype(np.float32).reshape(-1, 1, 1, 1)

            normalized_magnitudes = np.where(
                max_values_per_map > 1e-10, magnitudes / max_values_per_map, 1e-10
            )

            return np.array([np.max(img) / (np.sqrt(np.mean(img**2)) + 1e-10) for img in normalized_magnitudes])

        return self._compute_feature_per_experiment(
            feature_name="CrestFactorMagnitudes",
            feature_func=crest_factor_function,
            exp_name=exp_name,
            save_path=save_path
        )

    def compute_clearance_factor_magnitudes(self, exp_name=None, save_path=None):
        """
        Computes the clearance factor for each radar magnitude map for a specific experiment or all experiments.
        The clearance factor is defined as the ratio of the max value to the mean squared value.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            save_path (str, optional): Path to save the computed clearance factor arrays.

        Returns:
            dict or np.ndarray: 
                - If exp_name is specified, returns a 1D NumPy array of clearance factor values.
                - If None, returns a dictionary with all experiments' clearance factor arrays.
        """
        def clearance_factor_function(magnitudes, max_values_per_map):
            """
            Compute the clearance factor of normalized magnitudes.

            Args:
                magnitudes (np.ndarray): 4D array (batch, H, W, 1) of magnitude values.
                max_values_per_map (np.ndarray): 1D array (batch) of max values per Doppler map.

            Returns:
                np.ndarray: Clearance factor values for each Doppler map.
            """
            magnitudes = magnitudes.astype(np.float32)
            max_values_per_map = max_values_per_map.astype(np.float32).reshape(-1, 1, 1, 1)

            normalized_magnitudes = np.where(
                max_values_per_map > 1e-10, magnitudes / max_values_per_map, 1e-10
            )

            return np.array([np.max(img) / (np.mean(img**2) + 1e-10) for img in normalized_magnitudes])

        return self._compute_feature_per_experiment(
            feature_name="ClearanceFactorMagnitudes",
            feature_func=clearance_factor_function,
            exp_name=exp_name,
            save_path=save_path
        )


    def compute_and_save_fft(self, exp_name=None, crop_size=100, save_resolution=(512, 512)):
        """
        Computes the FFT of Doppler magnitude maps, crops the central part, and saves them as high-resolution PNG images
        with the original names, including the corresponding label.

        Args:
            exp_name (str, optional): Name of the experiment to process. If None, processes all experiments.
            crop_size (int): Size of the region to keep in the center of the FFT (default: 100).
            save_resolution (tuple): Target resolution (width, height) for saved FFT images.

        Returns:
            None
        """
        print(f"📡 Starting FFT computation and saving for experiment(s): {exp_name if exp_name else 'ALL'}")

        # Select which experiments to process
        experiments_to_process = [exp_name] if exp_name else self.exp_list

        for exp in experiments_to_process:
            print(f"🔄 Processing experiment: {exp}")

            # Load magnitudes and labels
            magnitudes = self.get_magnitudes(exp)
            labels = self.get_labels(exp)

            # Check if data exists
            if magnitudes.size == 0 or labels.size == 0:
                print(f"⚠️ No magnitudes or labels found for {exp}. Skipping...")
                continue

            # Ensure labels match magnitudes in shape
            if labels.shape[0] != magnitudes.shape[0]:
                print(f"⚠️ Mismatch: Labels and magnitudes count differ for {exp}. Skipping...")
                continue

            # Create output directory for FFT images
            fft_dir = os.path.join(self.base_path, exp, "FFT")
            os.makedirs(fft_dir, exist_ok=True)

            for i in range(magnitudes.shape[0]):  # Process each Doppler map
                img = magnitudes[i]  # Select a single magnitude map
                label = labels[i]  # Get corresponding label

                # Compute FFT and shift the zero-frequency component to the center
                fft_magnitude = np.fft.fft2(img, axes=(0, 1))
                fft_magnitude_shifted = np.fft.fftshift(fft_magnitude)

                # Convert to log scale to enhance small details
                fft_magnitude_log = np.log1p(np.abs(fft_magnitude_shifted))

                # Crop the center region
                center_x, center_y = fft_magnitude_log.shape[0] // 2, fft_magnitude_log.shape[1] // 2
                cropped_fft = fft_magnitude_log[
                    center_x - crop_size:center_x + crop_size,
                    center_y - crop_size:center_y + crop_size
                ]

                # Normalize to 0-255 for saving as PNG
                cropped_fft_normalized = cv2.normalize(
                    cropped_fft, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )

                # Resize to higher resolution
                high_res_fft = cv2.resize(
                    cropped_fft_normalized, save_resolution, interpolation=cv2.INTER_CUBIC
                )

                # Define save path with the correct format: map_x_y.png
                fft_image_path = os.path.join(fft_dir, f"map_{i}_{int(label)}.png")

                # Save as high-resolution PNG
                cv2.imwrite(fft_image_path, high_res_fft, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Lossless PNG

            print(f"✅ FFT images saved for {exp} in {fft_dir}")

        print("🎉 FFT computation and saving completed for all experiments!")

def plot_feature_per_label(labels, feature_values, feature_name="Feature", color='blue'):
    """
    Plots a scatter plot of a feature's values per label.
    Individual points are in blue, and mean values for each label are in red.

    Args:
        labels (numpy.ndarray): Array of labels corresponding to each data point.
        feature_values (numpy.ndarray): Array of feature values corresponding to labels.
        feature_name (str, optional): Name of the feature to display in the title and axis labels.

    Returns:
        None: Displays the plot.
    """
    labels = np.array(labels)
    feature_values = np.array(feature_values)

    fig, ax = plt.subplots()

    # ✅ 1. Affichage optimisé des points individuels
    ax.scatter(labels, feature_values, color=color, alpha=0.01)

    # ✅ 2. Calcul vectorisé des moyennes
    unique_labels, counts = np.unique(labels, return_counts=True)
    mean_values = np.bincount(labels, weights=feature_values) / counts

    # ✅ 3. Affichage optimisé des moyennes
    ax.scatter(unique_labels, mean_values, color='red', label=f'Mean {feature_name}', s=100, edgecolor='black')

    # ✅ 4. Mise en forme du graphique
    ax.set_ylabel(feature_name)
    ax.set_title(f'{feature_name} per Label')
    ax.set_xticks(unique_labels)
    ax.set_xticklabels(unique_labels, rotation=55)
    ax.legend()

    plt.show()



def plot_correlation_matrix(features, labels, feature_names, colors, label_name="Label"):
    """
    Computes and plots the correlation matrix between all features and the labels.
    Highlights the entire label row to emphasize correlation with the features.

    Args:
        features (numpy.ndarray): 2D array (num_samples, num_features) of feature values.
        labels (numpy.ndarray): 1D array (num_samples,) of labels.
        feature_names (list): List of feature names.
        label_name (str, optional): Name of the label column in the correlation matrix.

    Returns:
        None: Displays the correlation heatmap.
    """
    # Convert features and labels into a DataFrame
    df = pd.DataFrame(features.T, columns=feature_names)
    df[label_name] = labels  # Add labels as a column

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)

    # Highlight the label row
    y_index = df.columns.get_loc(label_name)  # Get row index of the label
    ax.add_patch(plt.Rectangle((0, y_index), len(feature_names), 1, fill=True, color='yellow', alpha=0.3, lw=0))

    # Modify tick labels with assigned colors
    for tick_label, color in zip(ax.get_xticklabels(), colors + ["black"]):  # Add black for label column
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    for tick_label, color in zip(ax.get_yticklabels(), colors + ["black"]):  # Add black for label column
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    # Add title and labels
    plt.title("Correlation Matrix of Features and Labels")
    plt.show()


def normalized_mutual_info(X, Y):
    """
    Computes the normalized mutual information between two variables.
    """
    X = X.ravel()  # S'assure que X est 1D
    Y = Y.ravel()  # S'assure que Y est 1D

    mi = mutual_info_regression(X.reshape(-1, 1), Y, discrete_features=False)[0]
    H_X = mutual_info_regression(X.reshape(-1, 1), X, discrete_features=False)[0]
    H_Y = mutual_info_regression(Y.reshape(-1, 1), Y, discrete_features=False)[0]

    return mi / max(H_X, H_Y) if max(H_X, H_Y) > 1e-6 else 0  # Évite la division par zéro

def compute_mutual_info_matrix_all(data, feature_names):
    """
    Computes the normalized mutual information matrix for all features and the label.

    Args:
        data (numpy.ndarray): 2D array (num_features+1, num_samples) containing all features and labels.
        feature_names (list): List of feature names including "Label".

    Returns:
        pd.DataFrame: Normalized mutual information matrix.
    """
    num_vars = data.shape[0]
    mi_matrix = np.zeros((num_vars, num_vars))

    # Compute mutual information for all pairs
    for i in range(num_vars):
        for j in range(i, num_vars):
            mi_matrix[i, j] = normalized_mutual_info(data[i], data[j])
            mi_matrix[j, i] = mi_matrix[i, j]  # Symmetric matrix

    return pd.DataFrame(mi_matrix, columns=feature_names, index=feature_names)

def plot_mutual_information_matrix(features, labels, feature_names, colors):
    """
    Computes and plots the normalized mutual information matrix for all features and labels.

    Args:
        features (numpy.ndarray): 2D array (num_samples, num_features) of feature values.
        labels (numpy.ndarray): 1D array (num_samples,) of labels.
        feature_names (list): List of feature names.
        colors (list): List of colors for feature labels.

    Returns:
        None: Displays the mutual information heatmap.
    """
    # Combine features and labels into a single array
    data = np.vstack([features, labels])  # Stack features and labels together
    all_feature_names = feature_names + ["Label"]

    # Compute MI matrix
    mi_df = compute_mutual_info_matrix_all(data, all_feature_names)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mi_df, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)

    # Highlight the label row
    y_index = mi_df.index.get_loc("Label")  # Get row index of the label
    ax.add_patch(plt.Rectangle((0, y_index), len(feature_names), 1, fill=True, color='yellow', alpha=0.3, lw=0))

    # Modify tick labels with assigned colors
    for tick_label, color in zip(ax.get_xticklabels(), colors + ["black"]):  # Add black for label column
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    for tick_label, color in zip(ax.get_yticklabels(), colors + ["black"]):  # Add black for label column
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    # Add title and labels
    plt.title("Normalized Mutual Information Matrix (Features & Label)")
    plt.show()


def plot_pair_feature_space(feature1, feature2, labels, feature1_name="Feature 1", feature2_name="Feature 2"):
    """
    Plots a 2D scatter plot of two features in the same space, color-coded by labels.

    Args:
        feature1 (numpy.ndarray): 1D array of values for the first feature.
        feature2 (numpy.ndarray): 1D array of values for the second feature.
        labels (numpy.ndarray): 1D array of labels for color-coding.
        feature1_name (str, optional): Name of the first feature for the x-axis.
        feature2_name (str, optional): Name of the second feature for the y-axis.

    Returns:
        None: Displays the scatter plot.
    """
    fig, ax = plt.subplots()
    unique_labels = np.unique(labels)

    # Color labels gradually from a one-color matplotlib colormap
    color_map = plt.get_cmap("YlOrRd")
    colors = color_map(np.linspace(0, 1, len(unique_labels)))

    # Plot each label with a different color
    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)
        ax.scatter(feature1[indices], feature2[indices], color=colors[i], label=f"Label {int(label)}")

    # Add labels and legend
    ax.set_xlabel(feature1_name)
    ax.set_ylabel(feature2_name)
    ax.set_title(f"{feature1_name} vs. {feature2_name}")
    ax.legend()

    plt.show()

def plot_label_distribution(labels, title="Label Distribution"):
    """
    Plots a histogram showing the distribution of labels.

    Args:
        labels (numpy.ndarray): Array of labels to plot.
        title (str, optional): Title of the histogram.

    Returns:
        None: Displays the histogram.
    """
    labels = np.array(labels)  # S'assurer que c'est un tableau numpy
    
    plt.figure(figsize=(8, 5))
    plt.hist(labels, bins=np.arange(labels.min(), labels.max() + 2) - 0.5, edgecolor='black', alpha=0.7)

    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(np.arange(labels.min(), labels.max() + 1))  # Assurer des ticks entiers

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

def define_models():
    """Définit les modèles de classification."""
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, verbose=False),
    }

def prepare_data(features, labels, N, n_components_pca=6):
    """Prépare les données en créant N classes et en appliquant normalisation + PCA."""
    discrete_labels = pd.qcut(labels, N, labels=False, duplicates="drop")

    # Split en train/validation/test (60% train, 20% val, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(features, discrete_labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Normalisation des features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # PCA pour réduire la dimensionnalité
    pca = PCA(n_components=n_components_pca)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_pca, X_val_pca, X_test_pca

def train_and_evaluate(models, X_train, X_val, X_test, y_train, y_val, y_test, X_train_pca, X_val_pca, X_test_pca, N):
    """Entraîne et évalue chaque modèle avec et sans PCA."""
    all_results = {model_name: {
        "N": [], "Without PCA - Validation acc.": [], "Without PCA - Testing acc.": [],
        "With PCA - Validation acc.": [], "With PCA - Testing acc.": [], "Random Guess": []
    } for model_name in models.keys()}

    random_guess_acc = 1 / N * 100  # Précision théorique du choix aléatoire

    for model_name, model in models.items():
        print(f"   ➤ Training {model_name}...")

        # Entraînement sans PCA
        model.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val)) * 100
        test_acc = accuracy_score(y_test, model.predict(X_test)) * 100

        # Entraînement avec PCA
        model.fit(X_train_pca, y_train)
        val_acc_pca = accuracy_score(y_val, model.predict(X_val_pca)) * 100
        test_acc_pca = accuracy_score(y_test, model.predict(X_test_pca)) * 100

        # Stocker les résultats
        all_results[model_name]["N"].append(N)
        all_results[model_name]["Without PCA - Validation acc."].append(val_acc)
        all_results[model_name]["Without PCA - Testing acc."].append(test_acc)
        all_results[model_name]["With PCA - Validation acc."].append(val_acc_pca)
        all_results[model_name]["With PCA - Testing acc."].append(test_acc_pca)
        all_results[model_name]["Random Guess"].append(random_guess_acc)

    return all_results

def plot_results(all_results):
    """Génère les courbes de classification pour chaque modèle."""
    for model_name, results in all_results.items():
        df_results = pd.DataFrame(results)

        plt.figure(figsize=(10, 6))
        plt.plot(df_results["N"], df_results["Without PCA - Validation acc."], 'b--*', label="Without PCA - Validation acc.")
        plt.plot(df_results["N"], df_results["Without PCA - Testing acc."], 'b-*', label="Without PCA - Testing acc.")
        plt.plot(df_results["N"], df_results["With PCA - Validation acc."], 'r--*', label="With PCA - Validation acc.")
        plt.plot(df_results["N"], df_results["With PCA - Testing acc."], 'r-*', label="With PCA - Testing acc.")
        plt.plot(df_results["N"], df_results["Random Guess"], 'k-*', label="Random Guess", linewidth=2)

        plt.xlabel("Number of classes $N_c$")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Classification Accuracy vs. Number of Classes ({model_name})")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 100)
        plt.show()

def run_classification_experiment(features, labels, n_components_pca=6, n_classes_range=(2, 10)):
    """Exécute l'entraînement, l'évaluation et l'affichage des performances des modèles."""
    models = define_models()
    all_results = {model_name: {"N": [], "Without PCA - Validation acc.": [], "Without PCA - Testing acc.": [],
                                "With PCA - Validation acc.": [], "With PCA - Testing acc.": [], "Random Guess": []}
                   for model_name in models.keys()}

    for N in range(n_classes_range[0], n_classes_range[1] + 1):
        print(f"🔹 Evaluating with N = {N} classes")
        X_train, X_val, X_test, y_train, y_val, y_test, X_train_pca, X_val_pca, X_test_pca = prepare_data(features, labels, N, n_components_pca)
        results = train_and_evaluate(models, X_train, X_val, X_test, y_train, y_val, y_test, X_train_pca, X_val_pca, X_test_pca, N)

        # Fusionner les résultats
        for model_name in models.keys():
            for key in all_results[model_name]:
                all_results[model_name][key].extend(results[model_name][key])

    plot_results(all_results)


def train_and_plot_classification(model, features, labels, test_size=0.2, random_state=42):
    """
    Entraîne un modèle de classification et affiche un graphique des prédictions vs. labels vrais.

    Args:
        model: Modèle de classification de sklearn (ex: RandomForestClassifier, SVC, MLPClassifier).
        features (numpy.ndarray): Matrice des caractéristiques d'entrée (X).
        labels (numpy.ndarray): Tableau des labels (y).
        test_size (float, optional): Proportion des données utilisées pour le test (default: 0.2).
        random_state (int, optional): Graine pour la reproductibilité (default: 42).

    Returns:
        model: Modèle entraîné.
        y_pred: Prédictions sur l'ensemble de test.
    """

    # 📌 1️⃣ Split en train/test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    # 📌 2️⃣ Normalisation des features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 📌 3️⃣ Entraînement du modèle
    model.fit(X_train, y_train)

    # 📌 4️⃣ Prédictions
    y_pred = model.predict(X_test)

    # 📌 5️⃣ Évaluation de la précision
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"✅ Model: {model.__class__.__name__}")
    print(f"📈 Accuracy: {accuracy:.2f}%")

    # 📌 6️⃣ Création du graphique des prédictions vs labels vrais
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="blue", alpha=0.1, label="Predicted vs True")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Ideal Fit")

    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")

    # 📌 Affichage des ticks
    unique_labels = np.unique(labels)
    plt.xticks(unique_labels)
    plt.yticks(unique_labels)

    plt.title(f"{model.__class__.__name__}: Predicted vs True Labels")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, y_pred


def train_and_plot_regression(model, features, labels, test_size=0.2, random_state=42):
    """
    Entraîne un modèle de régression et affiche un graphique des prédictions vs. valeurs réelles.

    Args:
        model: Modèle de régression de sklearn (ex: LinearRegression(), Ridge(), SVR()).
        features (numpy.ndarray): Matrice des caractéristiques d'entrée (X).
        labels (numpy.ndarray): Tableau des labels (y).
        test_size (float, optional): Proportion des données utilisées pour le test (default: 0.2).
        random_state (int, optional): Graine pour la reproductibilité (default: 42).

    Returns:
        model: Modèle entraîné.
        y_pred: Prédictions sur l'ensemble de test.
    """

    # 📌 Split en train/test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    # 📌 Normalisation des features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 📌 Entraînement du modèle
    model.fit(X_train, y_train)

    # 📌 Prédictions
    y_pred = model.predict(X_test)

    # 📌 Correction : Remettre les valeurs négatives à 0 (optionnel selon le contexte)
    y_pred = np.clip(y_pred, 0, None)

    # 📌 Évaluation du modèle
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Model: {model.__class__.__name__}")
    print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
    print(f"📈 R² Score: {r2:.2f}")

    # 📌 Affichage des résultats
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="blue", alpha=0.1, label="Predicted vs True")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Ideal Fit")
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")

    # 📌 Afficher les labels comme ticks sur X et Y
    plt.xticks(np.arange(0, np.max(labels)+1))
    plt.yticks(np.arange(0, np.max(labels)+1))

    plt.title(f"{model.__class__.__name__}: Predicted vs True Labels")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, y_pred

def train_and_evaluate_regression(model, features, labels, test_size=0.2, n_classes_range=(2, 10), random_state=42):
    """
    Entraîne un modèle de régression, fait des prédictions et évalue sa capacité à classer les données en N catégories.

    Args:
        model: Modèle de régression de sklearn (ex: LinearRegression(), Ridge(), SVR()).
        features (numpy.ndarray): Matrice des caractéristiques d'entrée (X).
        labels (numpy.ndarray): Tableau des labels (y).
        test_size (float, optional): Proportion des données utilisées pour le test (default: 0.2).
        n_classes_range (tuple, optional): Plage de valeurs pour le nombre de classes (par défaut: (2,10)).
        random_state (int, optional): Graine pour la reproductibilité (default: 42).

    Returns:
        pd.DataFrame: Tableau des résultats avec "N", "Regression Accuracy" et "Random Guess".
    """

    # 📌 1️⃣ Split des données en train/test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    # 📌 2️⃣ Normalisation des features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 📌 3️⃣ Entraînement du modèle
    model.fit(X_train, y_train)

    # 📌 4️⃣ Prédictions
    y_pred = model.predict(X_test)

    # 📌 5️⃣ Correction : Remettre les valeurs négatives à 0 si nécessaire
    y_pred = np.clip(y_pred, 0, None)

    # 📌 6️⃣ Évaluation du modèle en régression
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Model: {model.__class__.__name__}")
    print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
    print(f"📈 R² Score: {r2:.2f}")

    # 📌 7️⃣ Conversion en classification (discrétisation des labels)
    results = {"N": [], "Regression Accuracy": [], "Random Guess": []}

    for N in range(n_classes_range[0], n_classes_range[1] + 1):
        print(f"🔹 Evaluating Regression Accuracy for N = {N}")

        # Discrétisation des labels réels en N classes (quantiles)
        discrete_labels = pd.qcut(y_test, N, labels=False, duplicates="drop")

        # Discrétisation des prédictions dans les mêmes bins
        pred_bins = pd.qcut(y_pred, N, labels=False, duplicates="drop")

        # Calcul de la précision (proportion des valeurs bien classées)
        accuracy = np.mean(discrete_labels == pred_bins) * 100

        # Précision théorique d'un choix aléatoire (1/N)
        random_guess_acc = 1 / N * 100

        # Stockage des résultats
        results["N"].append(N)
        results["Regression Accuracy"].append(accuracy)
        results["Random Guess"].append(random_guess_acc)

    # 📌 8️⃣ Convertir en DataFrame
    df_results = pd.DataFrame(results)

    # 📌 9️⃣ Tracé des résultats
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["N"], df_results["Regression Accuracy"], 'b-*', label="Regression Accuracy")
    plt.plot(df_results["N"], df_results["Random Guess"], 'k-*', label="Random Guess", linewidth=2)

    plt.xlabel("Number of classes $N_c$")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Regression Accuracy vs. Number of Classes ({model.__class__.__name__})")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)  # Accuracy entre 0% et 100%
    plt.show()

    return df_results