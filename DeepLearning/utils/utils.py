"""
Remarque: La majorité des fonctions de la classe ont été générés/revues par ChatGPT et GitHub Copilot
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import yaml
from termcolor import colored
from sklearn.metrics import mean_absolute_error



class LazyImageLoader:
    """Classe pour charger les images à la demande (lazy loading), triées par numéro."""
    
    def __init__(self, directory, not_lazy=False):
        self.not_lazy = not_lazy
        self.directory = directory
        self.file_list = self._get_sorted_files()

        if self.not_lazy:
            self.file_list_loaded = self.load_all_images()

    def _get_sorted_files(self):
        """Récupère et trie les fichiers .png en fonction du numéro dans leur nom."""
        if not os.path.exists(self.directory):
            return []
        
        # Ne garde que les fichiers .png normaux (ignore ceux qui commencent par '._' ou autres fichiers cachés)
        files = [f for f in os.listdir(self.directory)
                if f.endswith(".png") and not f.startswith("._") and not f.startswith(".")]

        # Fonction de tri qui extrait uniquement le numéro du nom de fichier
        def extract_number(filename):
            match = re.search(r'map_(\d+)_', filename)  # Capture uniquement le nombre après "map_"
            return int(match.group(1)) if match else float('inf')  # Assigne une valeur élevée si pas de numéro

        return sorted(files, key=extract_number)  # Trie la liste selon le numéro

    def load_all_images(self):
        """Charge toutes les images PNG dans un tableau numpy."""
        data = []
        for file in tqdm(self.file_list, desc="Chargement des images", unit="image"):
            file_path = os.path.join(self.directory, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = np.expand_dims(img, axis=-1)
                data.append(img)
        return np.array(data)

    def load_image(self, index):
        """
        Charge une image PNG spécifique lorsqu’elle est demandée.
        Si self.not_lazy est True -> charge l'image depuis le tableau numpy chargé.
        Sinon -> charge l'image depuis le disque.
        """
        
        if self.not_lazy:
            return self.file_list_loaded[index] if 0 <= index < len(self.file_list_loaded) else None
    
        else:
            if 0 <= index < len(self.file_list):
                file_path = os.path.join(self.directory, self.file_list[index])
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Chargement en niveaux de gris
                if img is not None:
                    img = np.expand_dims(img, axis=-1)  # Ajout de la dimension du canal
                    return img
            return None  # Retourne None si l'image est introuvable ou si l'index est hors limites
            
        # Retourne None si l'image est introuvable ou si l'index est hors limites

    def __len__(self):
        """Retourne le nombre d'images disponibles."""
        return len(self.file_list)
    
    def __str__(self):
        """Affiche les noms de toutes les images disponibles."""
        return str(self.file_list)


class LazyVideoFrameLoader:
    """Classe pour charger des frames vidéo à la demande, triées par le premier numéro dans le nom (ex: frame_4_2.jpg -> 4)."""

    def __init__(self, directory, not_lazy=False):
        self.not_lazy = not_lazy
        self.directory = directory
        self.file_list = self._get_sorted_files()

        if self.not_lazy:
            self.file_list_loaded = self._load_image_files()

    def _get_sorted_files(self):
        """Récupère et trie les fichiers .jpg/.jpeg selon le premier numéro dans leur nom (ex: frame_4_2.jpg -> 4)."""
        if not os.path.exists(self.directory):
            return []

        files = [f for f in os.listdir(self.directory)
                 if f.lower().endswith((".jpg", ".jpeg")) and not f.startswith("._") and not f.startswith(".")]

        def extract_number(filename):
            match = re.search(r'frame_(\d+)_\d+', filename)
            return int(match.group(1)) if match else float('inf')

        return sorted(files, key=extract_number)

    def _load_image_files(self):
        """Charge tous les fichiers .jpg/.jpeg dans un tableau numpy (RGB)."""
        data = []
        for file in tqdm(self.file_list, desc="Chargement des frames vidéo", unit="frame"):
            file_path = os.path.join(self.directory, file)
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data.append(img)
        return np.array(data) if data else np.array([])

    def load_frame(self, index):
        """Charge une frame spécifique selon l'index trié."""
        if self.not_lazy:
            return self.file_list_loaded[index] if 0 <= index < len(self.file_list_loaded) else None
        else:
            if 0 <= index < len(self.file_list):
                file_path = os.path.join(self.directory, self.file_list[index])
                img = cv2.imread(file_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return None

    def __len__(self):
        return len(self.file_list)

    def __str__(self):
        return str(self.file_list)
    


class DataLoader:
    def __init__(self, base_path, param, exp_list=None, to_load=None):
        """
        Initialise le DataLoader sans charger les cartes radar au préalable.

        Args:
            base_path (str): Chemin de la base des expériences.
            exp_list (list, optional): Liste des expériences à charger.
            to_load (list, optional): Liste des types de données à charger immédiatement.
        """
        self.base_path = base_path
        self.cropped_radar_maps = param["DATASET"].get("CROPPED_RADAR_MAPS", False)
        print(colored(f"Cropped = {self.cropped_radar_maps}"))
        self.exp_list = exp_list if exp_list else self._discover_experiments()
        self.data = {
            "min_values": {}, "max_values": {}, "min_values2": {},
            "max_values2" : {}, "labels": {}, "magnitudes": {},
            "phases": {}, "magnitudes2": {}, "phases2": {}, "video_frames": {}
        }

        # Chargement immédiat des données (sauf les cartes radar)
        self._load_data(to_load)

    def _discover_experiments(self):
        """Automatically detects available experiments in the database."""
        return [exp for exp in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, exp))]

    def _load_data(self, to_load):
        """Charge certaines données immédiatement, sauf les images, avec barre de chargement."""
        if to_load is None:
            to_load = ["max_values", "max_values2","labels", "magnitudes", "phases"]

        # Utilisation de tqdm pour afficher une barre de progression
        for exp in tqdm(self.exp_list, desc="🔄 Chargement des données", unit="exp"):
            if "min_values" in to_load: 
                self.data["min_values"][exp] = self._load_min_values(exp)
            if "max_values" in to_load: 
                if self.cropped_radar_maps:
                    self.data["max_values"][exp] = self._load_max_values_cropped(exp)
                else:
                    self.data["max_values"][exp] = self._load_max_values(exp)
            if "max_values2" in to_load:
                self.data["max_values2"][exp] = self._load_max_values(exp)
            if "labels" in to_load: 
                self.data["labels"][exp] = self._load_labels(exp)

            if "video_frames" in to_load: 
                self.data["video_frames"][exp] = LazyVideoFrameLoader(os.path.join(self.base_path, exp, "VideoFrames"), not_lazy=False)

            if self.cropped_radar_maps:
                magnitudes_to_load = "RadarMagnitudesCropped"
                phases_to_load = "RadarPhases"
            else:
                magnitudes_to_load = "RadarMagnitudes"
                phases_to_load = "RadarPhases_NOT_AVAILABLE"

            # Utilisation du proxy pour le chargement différé des images
            self.data["magnitudes"][exp] = LazyImageLoader(os.path.join(self.base_path, exp, magnitudes_to_load), not_lazy=False)
            self.data["phases"][exp] = LazyImageLoader(os.path.join(self.base_path, exp, phases_to_load))
            self.data["magnitudes2"][exp] = LazyImageLoader(os.path.join(self.base_path, exp, "RadarMagnitudesAntenna1"))
            self.data["phases2"][exp] = LazyImageLoader(os.path.join(self.base_path, exp, "RadarPhasesAntenna1"))

    def _load_min_values(self, exp_name):
        """Loads min values from a single file."""
        min_path = os.path.join(self.base_path, exp_name, "MinValues", "min_values.npy")
        return np.load(min_path) if os.path.exists(min_path) else np.array([])

    def _load_max_values(self, exp_name):
        """Loads max values from a single file."""
        max_path = os.path.join(self.base_path, exp_name, "MaxValues", "max_values.npy")
        return np.load(max_path) if os.path.exists(max_path) else np.array([])
    
    def _load_max_values_cropped(self, exp_name):
        """Loads max values from a single file."""
        max_path = os.path.join(self.base_path, exp_name, "MaxValuesCropped", "max_values.npy")
        return np.load(max_path) if os.path.exists(max_path) else np.array([])

    def _load_max_values2(self, exp_name):
        """Loads max values from a single file."""
        max_path = os.path.join(self.base_path, exp_name, "MaxValuesAntenna1", "max_values.npy")
        return np.load(max_path) if os.path.exists(max_path) else np.array([])

    def _load_labels(self, exp_name):
        """Loads labels from a single file."""
        labels_path = os.path.join(self.base_path, exp_name, "Labels", "labels.npy")
        return np.load(labels_path) if os.path.exists(labels_path) else np.array([])

    def _load_video_frames(self, exp_name):
        """Loads all video frames for the experiment."""
        video_path = os.path.join(self.base_path, exp_name, "VideoFrames")
        return self._load_image_files(video_path)

    def get_magnitude(self, exp_name, index):
        """
        Récupère une seule image de magnitude à la demande.

        Args:
            exp_name (str): Nom de l'expérience.
            index (int): Index de l'image dans la séquence.

        Returns:
            np.ndarray: Image radar de magnitude sous forme de tableau numpy.
        """
        if exp_name in self.data["magnitudes"]:
            return self.data["magnitudes"][exp_name].load_image(index)
        return None  # Retourne None si l'expérience n'existe pas
    
    def get_magnitude2(self, exp_name, index):
        """
        Récupère une seule image de magnitude à la demande.

        Args:
            exp_name (str): Nom de l'expérience.
            index (int): Index de l'image dans la séquence.

        Returns:
            np.ndarray: Image radar de magnitude sous forme de tableau numpy.
        """
        if exp_name in self.data["magnitudes2"]:
            return self.data["magnitudes2"][exp_name].load_image(index)
        return None

    def get_number_of_magnitudes(self, exp_name):
        """
        Retourne le nombre total d'images de magnitude pour une expérience.

        Args:
            exp_name (str): Nom de l'expérience.

        Returns:
            int: Nombre d'images disponibles.
        """
        if exp_name in self.data["magnitudes"]:
            return len(self.data["magnitudes"][exp_name])
        return 0  # Si l'expérience n'existe pas ou s'il n'y a pas d'images

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

    def get_max_values2(self, exp_name=None):
        """Retrieves max values, either for a specific experiment or combined."""
        return self.data["max_values2"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["max_values2"])

    def get_labels(self, exp_name=None):
        """Retrieves labels, either for a specific experiment or combined."""
        return self.data["labels"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["labels"])

    def get_magnitudes(self, exp_name=None):
        """Retrieves magnitudes, either for a specific experiment or combined."""
        return self.data["magnitudes"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["magnitudes"])

    def get_magnitudes2(self, exp_name=None):
        """Retrieves magnitudes, either for a specific experiment or combined."""
        return self.data["magnitudes2"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["magnitudes2"])

    def get_phases(self, exp_name=None):
        """Retrieves radar phases, either for a specific experiment or combined."""
        return self.data["phases"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["phases"])

    def get_phases2(self, exp_name=None):
        """Retrieves radar phases, either for a specific experiment or combined."""
        return self.data["phases2"].get(exp_name, np.array([])) if exp_name else self._get_combined_data(self.data["phases2"])

    def get_video_frame(self, exp_name, index):
        if exp_name in self.data["video_frames"]:
            return self.data["video_frames"][exp_name].load_frame(index)
        return None  # Retourne None si l'expérience n'existe pas
       


class DopplerDataset(Dataset):
    def __init__(self, data_loader, mode="train", param=None, shuffle=True, random_seed=42, sub_sample_factor=1):
        super(DopplerDataset, self).__init__()
        self.mode = mode.lower()
        self.train_split = param["DATASET"].get("TRAIN_SPLIT", 0.8)
        self.temporal_block_size = param["DATASET"].get("TEMPORAL_BLOCK_SIZE", 200)
        self.predictionType = param["TRAINING"]["PREDICTION_TYPE"]
        if self.predictionType == "classification":
            self.nb_classes = param["DATASET"]["NB_CLASSES"]
        
        self.force_max_to_0 = param["DATASET"].get("FORCE_MAX_TO_0", False)
        self.standardize_labels = param["DATASET"].get("STANDARDIZE_LABELS", True)
        self.use_median_labels = param["DATASET"].get("USE_MEDIAN_LABELS", False)

        self.activeAntenna2 = param["DATASET"].get("ACTIVE_ANTENNA2", False)
        self.activePhase = param["DATASET"].get("ACTIVE_PHASE", False)

        self.fold_number = param["DATASET"].get("FOLD_NUMBER", 1)

        # Set numpy random seed for reproducibility
        np.random.seed(random_seed)

        self.data_loader = data_loader
        self.sub_sample_factor = sub_sample_factor
        self.exp_list = self.data_loader.exp_list

        # Nouveau paramètre
        self.use_prev_frames = param["MODEL"].get("USE_PREV_FRAMES", False)
        self.nb_prev_frames = param["MODEL"].get("NB_PREV_FRAMES", 0)
        self.time_steps = param["MODEL"].get("TIME_STEPS", 1)

        self.data_indices = self._create_file_indices()

        # Statistiques de labels
        if self.predictionType == "classification":
            self.min_label, self.max_label = self._compute_label_range()
        else:
            self.mean_label, self.std_label = self._compute_label_stats()
            print(colored(f"Label stats: mean={self.mean_label}, std={self.std_label}", "yellow"))

        # Statistiques max_values et max_values2
        self.mean_max_values = np.mean(self.data_loader.get_max_values())
        self.std_max_values = np.std(self.data_loader.get_max_values())
        self.mean_max_values2 = np.mean(self.data_loader.get_max_values2())
        self.std_max_values2 = np.std(self.data_loader.get_max_values2())

        # Split des données (3 modes différents)
        self.train_indices, self.val_indices, self.test_indices = self._split_dataset_by_exp(fold_number=self.fold_number, shuffle=shuffle) #self._split_dataset_by_blocks() #self._split_dataset(shuffle, random_seed)

        if self.mode == "train":
            self.data_indices = self.train_indices
        elif self.mode == "val":
            self.data_indices = self.val_indices
        elif self.mode == "test":
            self.data_indices = self.test_indices
        else:
            raise ValueError("Le mode doit être 'train', 'val' ou 'test'.")
        
        if self.mode == "train":
            print(colored("\n" + "="*60, "blue"))
            print(colored("             DOPPLER DATASET INITIALIZATION SUMMARY", "green", attrs=["bold"]))
            print(colored("="*60, "blue"))

            print(colored("→ Active Antenna 2", "cyan") + "         : " + colored(str(self.activeAntenna2), "green" if self.activeAntenna2 else "red"))
            print(colored("→ Active Phase", "cyan") + "             : " + colored(str(self.activePhase), "green" if self.activePhase else "red"))
            print(colored("→ Use previous frames", "cyan") + "      : " + colored(str(self.use_prev_frames), "green" if self.use_prev_frames else "red"))
            if self.use_prev_frames:
                print(colored("→ Number of previous frames", "cyan") + ": " + colored(f"{self.nb_prev_frames}", "yellow"))
                print(colored("→ Time steps", "cyan") + "               : " + colored(f"{self.time_steps}", "yellow"))

            print(colored("→ Fold number", "cyan") + "              : " + colored(f"{self.fold_number}", "yellow"))
            if self.sub_sample_factor > 1:
                print(colored("→ Sub-sample factor", "cyan") + "        : " + colored(f"{self.sub_sample_factor}", "yellow"))
            
            print(colored("→ Total experiments", "cyan") + "        : " + colored(f"{len(self.exp_list)}", "white"))
            print(colored("→ Dataset sizes (samples)", "cyan") + " :")
            print("   " + colored("• Train set", "cyan") + "             : " + colored(f"{len(self.train_indices)}", "green"))
            print("   " + colored("• Validation set", "cyan") + "        : " + colored(f"{len(self.val_indices)}", "yellow"))
            print("   " + colored("• Test set", "cyan") + "              : " + colored(f"{len(self.test_indices)}", "magenta"))
            print(colored("→ Final selected dataset", "cyan") + "   : " + colored(f"{len(self.train_indices) + len(self.val_indices) + len(self.test_indices)} samples", "white", attrs=["bold"]))
            print(colored("="*60 + "\n", "blue"))


    def _create_file_indices(self):
        data_indices = []
        for exp in self.exp_list:
            num_images = len(self.data_loader.data["magnitudes"][exp])
            # Commencer à nb_prev_frames * time_steps pour que toutes les frames précédentes soient disponibles
            for i in range(self.nb_prev_frames * self.time_steps, num_images, self.sub_sample_factor):
                data_indices.append((exp, i))  # i = frame cible
        return np.array(data_indices)

    def _split_dataset_by_exp(self, fold_number: int, shuffle: bool = True):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "folds_config.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        fold_key = f"FOLD{fold_number}"
        fold_data = config["FOLDS"][fold_key]

        train_exps = [f"NewExp{e}" for e in fold_data["TRAIN"]]
        val_exps = [f"NewExp{e}" for e in fold_data["VAL"]]
        test_exps = [f"NewExp{e}" for e in fold_data["TEST"]]

        # ✅ conversion propre de tous les éléments (au cas où tu n’aies pas encore fait ça avant)
        data_indices = [list(idx) for idx in self.data_indices]

        train_indices = [idx for idx in data_indices if idx[0] in train_exps]
        val_indices = [idx for idx in data_indices if idx[0] in val_exps]
        test_indices = [idx for idx in data_indices if idx[0] in test_exps]

        # Si shuffle est True, mélange les indices d'entrainement
        if shuffle:
            np.random.shuffle(train_indices)

        return train_indices, val_indices, test_indices

    def _split_dataset_by_blocks(self):
        train_indices, val_indices, test_indices = [], [], []
        for exp in self.exp_list:
            num_frames = len(self.data_loader.data["magnitudes"][exp])
            start_index = self.nb_prev_frames * self.time_steps
            usable_frames = list(range(0, num_frames, self.sub_sample_factor))

            block_size = self.temporal_block_size
            train_size = int(block_size * self.train_split)
            val_size = (block_size - train_size) // 2
            test_size = block_size - train_size - val_size

            if val_size <= start_index or test_size <= start_index or train_size <= start_index:
                raise ValueError("La taille des blocs est trop petite par rapport au nombre de frames précédentes.")

            i = 0
            while i < len(usable_frames):
                remaining = len(usable_frames) - i
                current_block_size = min(block_size, remaining)

                current_train_size = int(current_block_size * self.train_split)
                current_val_size = (current_block_size - current_train_size) // 2
                current_test_size = current_block_size - current_train_size - current_val_size

                block = usable_frames[i:i + current_block_size]
                block_train = block[start_index:current_train_size]
                block_val = block[current_train_size + start_index :current_train_size + current_val_size]
                block_test = block[current_train_size + current_val_size + start_index :]

                train_indices += [(exp, idx) for idx in block_train]
                val_indices += [(exp, idx) for idx in block_val]
                test_indices += [(exp, idx) for idx in block_test]

                i += current_block_size

        return np.array(train_indices), np.array(val_indices), np.array(test_indices)
    
    
    def _split_dataset_random_shuffle(self, shuffle, random_seed):
        np.random.seed(random_seed)
        indices = np.arange(len(self.data_indices))
        if shuffle:
            np.random.shuffle(indices)

        train_size = int(len(indices) * self.train_split)
        val_size = (len(indices) - train_size) // 2
        test_size = len(indices) - train_size - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        return self.data_indices[train_indices], self.data_indices[val_indices], self.data_indices[test_indices]
    
    def _compute_label_range(self):
        all_labels = self.data_loader.get_labels()
        return min(all_labels), max(all_labels)

    def _compute_label_stats(self):
        all_labels = self.data_loader.get_labels()
        return np.mean(all_labels), np.std(all_labels)

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        exp_name, image_index = self.data_indices[idx]
        image_index = int(image_index)

        img_stack = []
        max_stack = []

        labels = []

        if self.use_prev_frames:
            indices = range(image_index - self.nb_prev_frames * self.time_steps, image_index + 1, self.time_steps)
        else:
            indices = [image_index]

        for i in indices:
            img_stack.append(self._load_and_preprocess_image(exp_name, i, antenna=1, type="magnitude"))
            max_stack.append(self._get_normalized_max_value(exp_name, i, antenna=1))
            labels.append(self._load_label(exp_name, i))

            if self.activePhase:
                img_stack.append(self._load_and_preprocess_image(exp_name, i, antenna=1, type="phase"))

            if self.activeAntenna2:
                img_stack.append(self._load_and_preprocess_image(exp_name, i, antenna=2, type="magnitude"))
                max_stack.append(self._get_normalized_max_value(exp_name, i, antenna=2))

                if self.activePhase:
                    img_stack.append(self._load_and_preprocess_image(exp_name, i, antenna=2, type="phase"))

        if self.use_prev_frames or self.activeAntenna2 or self.activePhase:
            img_tensor = torch.tensor(np.stack(img_stack, axis=0), dtype=torch.float32)  # (T or 2T, H, W)
        else:
            img_tensor = torch.tensor(np.array(img_stack))

        if self.force_max_to_0:
            max_stack = [0] * len(max_stack)

        max_tensor = torch.tensor(max_stack, dtype=torch.float32)

        # Compute median of labels
        if self.use_median_labels:
            label_median = np.median(labels)
            label_tensor = torch.tensor(label_median, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(labels[-1], dtype=torch.float32)

        return img_tensor, max_tensor, label_tensor

    def get_exp_and_frame(self, idx):
        """
        Récupère le nom de l'expérience et l'index de la frame pour un index donné.

        Args:
            idx (int): Index du dataset.

        Returns:
            tuple: (nom de l'expérience, index de la frame)
        """
        return self.data_indices[idx]

    def _load_and_preprocess_image(self, exp_name, image_index, type="magnitude", antenna=1):

        if type == "magnitude":
            if antenna == 1:
                img = self.data_loader.get_magnitude(exp_name, image_index)
            else:
                img = self.data_loader.get_magnitude2(exp_name, image_index)

        if type == "phase":
            if antenna == 1:
                img = self.data_loader.get_phases(exp_name).load_image(image_index)
            else:
                img = self.data_loader.get_phases2(exp_name).load_image(image_index)

        if img is None:
            raise FileNotFoundError(f"Image index {image_index} non trouvée pour {exp_name} (antenna {antenna})")
        img = img.astype(np.float32) / 255.0

        return img[..., 0]


    def _get_normalized_max_value(self, exp_name, index, antenna=1):
        if antenna == 1:
            max_val = self.data_loader.get_max_values(exp_name)[index]
            return (max_val - self.mean_max_values) / self.std_max_values
        else:
            max_val = self.data_loader.get_max_values2(exp_name)[index]
            return (max_val - self.mean_max_values2) / self.std_max_values2


    def _load_label(self, exp_name, image_index):
        labels = self.data_loader.get_labels(exp_name)
        label = labels[image_index]

        if self.predictionType == "classification":
            label = self._convert_label_to_class(label)
            return torch.tensor(label, dtype=torch.long)
        else:
            if self.standardize_labels:
                label = (label - self.mean_label) / self.std_label
            return label

    
def plot_learning_curves(train_losses, val_losses, results_path, file_name="learning_curve",title="Learning Curves"):
    """Plot the learning curves of a training session."""
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # Os make dir
    #y_min = 0
    #y_max = 1.5
    #plt.ylim(y_min, y_max)

    os.makedirs(results_path + "/_LearningCurves", exist_ok=True)
    plt.savefig(results_path + f"/_LearningCurves/{file_name}.pdf")
    plt.show()

def plot_multiple_learning_curves(all_train_losses, all_val_losses, results_path, max_y=0.5, title="Learning Curves"):
    """
    Plot multiple learning curves with average curves overlaid in bold.
    
    Args:
        all_train_losses (List[List[float]]): List of train loss curves (each a list of losses per epoch).
        all_val_losses (List[List[float]]): List of validation loss curves (each a list of losses per epoch).
        results_path (str): Directory path to save the plot.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))

    # Convert to numpy arrays
    all_train_losses = np.array(all_train_losses)
    all_val_losses = np.array(all_val_losses)

    # Plot individual curves
    for train_curve in all_train_losses:
        plt.plot(train_curve, color='blue', alpha=0.3)
    for val_curve in all_val_losses:
        plt.plot(val_curve, color='red', alpha=0.3)

    # Plot average curves
    mean_train = np.mean(all_train_losses, axis=0)
    mean_val = np.mean(all_val_losses, axis=0)
    plt.plot(mean_train, label="Mean Train Loss", color='blue', linewidth=2.5)
    plt.plot(mean_val, label="Mean Validation Loss", color='red', linewidth=2.5)

    # Plot formatting
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, max_y)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Create directory and save figure
    os.makedirs(os.path.join(results_path, "_LearningCurves"), exist_ok=True)
    plt.savefig(os.path.join(results_path, "_LearningCurves", "learning_curve_multiple.pdf"))
    plt.close()


def compare_multiple_learning_curves(
    all_trains,
    all_vals,
    labels,
    results_path,
    max_y=0.5,
    title="Comparison of Learning Curves"
):
    """
    Compare multiple training and validation loss curves.

    Args:
        all_trains (List[List[float]]): List of training loss curves.
        all_vals (List[List[float]]): List of validation loss curves.
        labels (List[str]): List of labels for each model.
        max_y (float): Maximum y-axis value for the plot.
        results_path (str): Path to save the plot.
        title (str): Plot title.
    """
    assert len(all_trains) == len(all_vals) == len(labels), "Mismatch in number of models and labels"
    
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'brown']

    plt.figure(figsize=(12, 6))

    for i in range(len(all_trains)):
        train = all_trains[i]
        val = all_vals[i]
        color = colors[i % len(colors)]

        # Plot training and validation with distinct line styles
        plt.plot(train, label=f"{labels[i]} Train", color=color, linestyle='-', linewidth=2)
        plt.plot(val, label=f"{labels[i]} Val", color=color, linestyle='--', linewidth=2)

        # Add markers every ~10 epochs
        num_points = len(train)
        for j in range(0, num_points, max(1, num_points // 10)):
            plt.scatter(j, train[j], color=color, s=20)
            plt.scatter(j, val[j], color=color, s=20, marker='x')

    # Plot formatting
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, max_y)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save figure
    os.makedirs(os.path.join(results_path, "_LearningCurves"), exist_ok=True)
    plt.savefig(os.path.join(results_path, "_LearningCurves", "comparison_multiple_learning_curves.pdf"))
    plt.close()


def generate_prediction_video(exp_name, data_loader, results_path, save_path, output_fps=5, frame_skip=3):
    import os
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    gts = np.load(os.path.join(results_path, "_PerExperimentPlots", f"{exp_name}_gts.npy"), allow_pickle=True)
    preds = np.load(os.path.join(results_path, "_PerExperimentPlots", f"{exp_name}_preds.npy"), allow_pickle=True)
    frames_number = np.load(os.path.join(results_path, "_PerExperimentPlots", f"{exp_name}_frames.npy"), allow_pickle=True)

    assert len(gts) == len(preds) == len(frames_number), "Longueurs incompatibles"

    sample_frame = data_loader.get_video_frame(exp_name, frames_number[0])
    radar_map = data_loader.get_magnitude(exp_name, frames_number[0])

    H, W, _ = sample_frame.shape
    radar_size = H
    graph_height = 3 * 200

    combined_top_w = W + radar_size
    output_size = (combined_top_w, H + graph_height)

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, output_size)

    for i in range(0, len(frames_number), frame_skip):
        frame_idx = frames_number[i]

        frame = data_loader.get_video_frame(exp_name, frame_idx)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        t = 15
        frame_bgr = cv2.copyMakeBorder(frame_bgr, t, t, t, t, cv2.BORDER_CONSTANT, value=(255, 0, 0))

        radar_map = data_loader.get_magnitude(exp_name, frame_idx)
        radar_map_resized = cv2.resize(radar_map, (radar_size, radar_size), interpolation=cv2.INTER_NEAREST)
        radar_map_uint8 = cv2.normalize(radar_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        radar_colored = cv2.applyColorMap(radar_map_uint8, cv2.COLORMAP_VIRIDIS)
        radar_colored = cv2.copyMakeBorder(radar_colored, t, t, t, t, cv2.BORDER_CONSTANT, value=(0, 0, 255))

        # === Curseurs texte ===
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color_red = (0, 0, 255)
        color_blue = (255, 0, 0)
        thickness = 2
        y_offset = H - 20
        x_offset = 50

        cv2.putText(frame_bgr, f"YOLO prediction: {int(gts[i])}", (x_offset, y_offset), font, font_scale, color_blue, thickness)
        cv2.putText(radar_colored, f"Radar-based model prediction: {round(preds[i], 1)}", (x_offset, y_offset), font, font_scale, color_red, thickness)

        top_combined = np.hstack((frame_bgr, radar_colored))

        fig, ax = plt.subplots(figsize=(combined_top_w / 100, graph_height / 100), dpi=100)
        fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.15)
        ax.plot(gts, label="YOLO (GT)", color="red")
        ax.plot(preds, label="Prediction", color="blue")
        ax.axvline(x=i, color="red", linestyle="-", linewidth=2)

        total_duration = 60
        total_frames = len(gts)
        ax.set_xlim([0, total_frames - 1])
        ax.set_ylim([0, max(max(preds), max(gts)) + 1])
        ax.set_title("Estimation of the number of people in the scene")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("People count")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right", fontsize=16)

        ticks = np.linspace(0, total_frames - 1, total_duration // 5 + 1, dtype=int)
        labels = [str(i * 5) for i in range(len(ticks))]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        graph_width = top_combined.shape[1]
        plot_img_resized = cv2.resize(plot_img, (graph_width, graph_height))
        final_frame = np.vstack((top_combined, plot_img_resized)).astype(np.uint8)

        if final_frame.shape[1] != output_size[0] or final_frame.shape[0] != output_size[1]:
            final_frame = cv2.resize(final_frame, output_size)

        out.write(final_frame)

    out.release()
    print(f"✅ Vidéo exportée : {save_path}")

def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)

def aggregate_by_second(time, values, fps=16, method="mean"):
    seconds = np.floor(time).astype(int)
    unique_sec = np.unique(seconds)
    agg_values = []

    for sec in unique_sec:
        mask = (seconds == sec)
        if method == "mean":
            agg_values.append(values[mask].mean())

    return unique_sec, np.array(agg_values)

def plot_predictions_vs_groundtruth(results_by_experiment, save_path):

    #createFolder
    createFolder(save_path)

    for exp_name, data in results_by_experiment.items():
        frames = np.array(data["frames"], dtype=int)
        preds = np.array(data["preds"])
        gts = np.array(data["gts"])

        time = 60 * (frames - frames.min()) / (frames.max() - frames.min())
        xticks = np.linspace(time.min(), time.max(), num=13)
        ylim = (0, 25)
        # 1️⃣ Raw prediction vs ground truth
        plt.figure(figsize=(16, 5))
        plt.plot(time, gts, label="YOLO (Ground Truth)", color='blue', alpha=0.7)
        plt.plot(time, preds, label="Model Prediction", color='red', alpha=0.6)
        plt.xlabel("Time (s)")
        plt.ylabel("Number of people")
        plt.title(f"Experiment: {exp_name} — Raw Prediction vs Ground Truth")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        plt.xticks(xticks)
        plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{exp_name}_raw_pred_vs_gt.pdf")
        plt.close()
        mae_raw = mean_absolute_error(gts, preds)
        print(f"[{exp_name}] MAE - Raw: {mae_raw:.2f}")


        # 2️⃣ Aggregated (per second) prediction vs ground truth
        sec, pred_avg = aggregate_by_second(time, preds, 1)
        _, gt_avg = aggregate_by_second(time, gts, 1)

        plt.figure(figsize=(16, 5))
        plt.plot(sec, gt_avg, label="YOLO (GT) - avg/sec", linestyle='--', marker='o', color='blue')
        plt.plot(sec, pred_avg, label="Model Prediction - avg/sec", linestyle='-', marker='x', color='red')
        plt.xlabel("Time (s)")
        plt.ylabel("Average People Count")
        plt.title(f"Experiment: {exp_name} — Aggregated Prediction vs Ground Truth (per second)")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        plt.xticks(xticks)
        plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{exp_name}_agg_pred_vs_gt.pdf")
        plt.close()
        mae_agg = mean_absolute_error(gt_avg, pred_avg)
        print(f"[{exp_name}] MAE - Aggregated: {mae_agg:.2f}")


        # 3️⃣ Aggregated + Rounded prediction vs ground truth
        pred_avg_rounded = np.maximum(np.round(pred_avg), 0)
        gt_avg_rounded = np.maximum(np.round(gt_avg), 0)

        plt.figure(figsize=(16, 5))
        plt.plot(sec, gt_avg_rounded, label="YOLO (GT) - rounded", linestyle='--', marker='o', color='blue')
        plt.plot(sec, pred_avg_rounded, label="Model Prediction - rounded", linestyle='-', marker='x', color='red')
        plt.xlabel("Time (s)")
        plt.ylabel("Rounded People Count")
        plt.title(f"Experiment: {exp_name} — Rounded Aggregated Prediction vs Ground Truth")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        plt.xticks(xticks)
        plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{exp_name}_rounded_agg_pred_vs_gt.pdf")
        plt.close()
        mae_rounded = mean_absolute_error(gt_avg_rounded, pred_avg_rounded)
        print(f"[{exp_name}] MAE - Rounded: {mae_rounded:.2f}")


        # 4️⃣ Raw GT vs Aggregated Prediction + Std
        from collections import defaultdict

        # Regroupement des prédictions par seconde
        preds_per_sec = defaultdict(list)
        for t, p in zip(time, preds):
            sec_t = int(t)
            preds_per_sec[sec_t].append(p)

        sec_sorted = sorted(preds_per_sec.keys())
        pred_means = [np.mean(preds_per_sec[s]) for s in sec_sorted]
        pred_stds = [np.std(preds_per_sec[s]) for s in sec_sorted]

        plt.figure(figsize=(16, 5))
        plt.plot(time, gts, label="YOLO (GT)", color='blue', alpha=0.7)
        plt.plot(sec_sorted, pred_means, label="Model Prediction - avg/sec", color='red')
        plt.fill_between(sec_sorted,
                         np.array(pred_means) - np.array(pred_stds),
                         np.array(pred_means) + np.array(pred_stds),
                         color='red', alpha=0.2, label="Std Dev (Prediction)")
        plt.xlabel("Time (s)")
        plt.ylabel("People Count")
        plt.title(f"Experiment: {exp_name} — GT (raw) vs Aggregated Prediction + Std")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        plt.xticks(xticks)
        plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{exp_name}_rawGT_aggPred_std.pdf")
        plt.close()
        mae_rawGT_aggPred = mean_absolute_error(gts, np.interp(time, sec_sorted, pred_means))
        print(f"[{exp_name}] MAE - Raw GT vs Aggregated Pred: {mae_rawGT_aggPred:.2f}")


        # 5️⃣ Aggregated GT + Std vs Aggregated Prediction + Std
        gts_per_sec = defaultdict(list)
        for t, g in zip(time, gts):
            sec_t = int(t)
            gts_per_sec[sec_t].append(g)

        gt_means = [np.mean(gts_per_sec[s]) for s in sec_sorted]
        gt_stds = [np.std(gts_per_sec[s]) for s in sec_sorted]

        plt.figure(figsize=(11, 3))
        plt.plot(time, gts, label="YOLOv3 (GT)", color='blue')
        #plt.fill_between(sec_sorted,
        #                 np.array(gt_means) - np.array(gt_stds),
        #                 np.array(gt_means) + np.array(gt_stds),
        #                 color='blue', alpha=0.2)#, label="Std Dev (GT)")

        plt.plot(sec_sorted, pred_means, label="Model Prediction", color='red')
        plt.fill_between(sec_sorted,
                         np.array(pred_means) - np.array(pred_stds),
                         np.array(pred_means) + np.array(pred_stds),
                         color='red', alpha=0.2)#, label="Std Dev (Prediction)")

        plt.xlabel("Time (s)", fontsize=20)
        plt.ylabel("People Count", fontsize=20)
        #plt.title(f"Experiment: {exp_name} — Aggregated GT vs Aggregated Prediction + Std Dev")
        # Thick legend
        #plt.legend(fontsize=20, loc="lower right")
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        plt.xticks(xticks)
        plt.xlim((0, 60))
        plt.ylim((0, 25))
        plt.tight_layout()
        plt.savefig(f"{save_path}/{exp_name}_aggGT_aggPred_std.pdf")
        plt.close()
        mae_aggGT_aggPred = mean_absolute_error(gt_means, pred_means)
        print(f"[{exp_name}] MAE - Aggregated GT vs Aggregated Pred: {mae_aggGT_aggPred:.2f}")

