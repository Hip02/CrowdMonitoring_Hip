"""
Remarque: La majorité des fonctions de la classe ont été générés/revues par ChatGPT et GitHub Copilot
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split


class LazyImageLoader:
    """Classe pour charger les images à la demande (lazy loading), triées par numéro."""
    
    def __init__(self, directory):
        self.directory = directory
        self.file_list = self._get_sorted_files()

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

    def load_image(self, index):
        """Charge une image PNG spécifique lorsqu’elle est demandée."""
        #print(f"index type = {type(index)}, index={index}")
        if 0 <= index < len(self.file_list):
            file_path = os.path.join(self.directory, self.file_list[index])
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Chargement en niveaux de gris
            if img is not None:
                img = np.expand_dims(img, axis=-1)  # Ajout de la dimension du canal
                return img
        return None  # Retourne None si l'image est introuvable ou si l'index est hors limites
    
    def __len__(self):
        """Retourne le nombre d'images disponibles."""
        return len(self.file_list)
    
    def __str__(self):
        """Affiche les noms de toutes les images disponibles."""
        return str(self.file_list)


class DataLoader:
    def __init__(self, base_path, exp_list=None, to_load=None):
        """
        Initialise le DataLoader sans charger les cartes radar au préalable.

        Args:
            base_path (str): Chemin de la base des expériences.
            exp_list (list, optional): Liste des expériences à charger.
            to_load (list, optional): Liste des types de données à charger immédiatement.
        """
        self.base_path = base_path
        self.exp_list = exp_list if exp_list else self._discover_experiments()
        self.data = {
            "min_values": {}, "max_values": {}, "min_values2": {},
            "max_values2" : {}, "labels": {}, "magnitudes": {},
            "phases": {}, "magnitudes2": {}, "phases2": {}
        }

        # Chargement immédiat des données (sauf les cartes radar)
        self._load_data(to_load)

    def _discover_experiments(self):
        """Automatically detects available experiments in the database."""
        return [exp for exp in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, exp))]

    def _load_data(self, to_load):
        """Charge certaines données immédiatement, sauf les images, avec barre de chargement."""
        if to_load is None:
            to_load = ["max_values", "max_values2","labels", "magnitudes", "magnitudes2"]

        # Utilisation de tqdm pour afficher une barre de progression
        for exp in tqdm(self.exp_list, desc="🔄 Chargement des données", unit="exp"):
            if "min_values" in to_load: 
                self.data["min_values"][exp] = self._load_min_values(exp)
            if "max_values" in to_load: 
                self.data["max_values"][exp] = self._load_max_values(exp)
            if "max_values2" in to_load:
                self.data["max_values2"][exp] = self._load_max_values(exp)
            if "labels" in to_load: 
                self.data["labels"][exp] = self._load_labels(exp)

            # Utilisation du proxy pour le chargement différé des images
            self.data["magnitudes"][exp] = LazyImageLoader(os.path.join(self.base_path, exp, "RadarMagnitudes"))
            self.data["phases"][exp] = LazyImageLoader(os.path.join(self.base_path, exp, "RadarPhases"))
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
    
    def _load_max_values2(self, exp_name):
        """Loads max values from a single file."""
        max_path = os.path.join(self.base_path, exp_name, "MaxValuesAntenna1", "max_values.npy")
        return np.load(max_path) if os.path.exists(max_path) else np.array([])

    def _load_labels(self, exp_name):
        """Loads labels from a single file."""
        labels_path = os.path.join(self.base_path, exp_name, "Labels", "labels.npy")
        return np.load(labels_path) if os.path.exists(labels_path) else np.array([])

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


class DopplerDataset(Dataset):
    def __init__(self, data_loader, mode="train", param=None, shuffle=True, random_seed=42, sub_sample_factor=1):
        super(DopplerDataset, self).__init__()
        self.mode = mode.lower()
        self.train_split = param["DATASET"].get("TRAIN_SPLIT", 0.8)
        self.temporal_block_size = param["DATASET"].get("TEMPORAL_BLOCK_SIZE", 200)
        self.predictionType = param["TRAINING"]["PREDICTION_TYPE"]
        if self.predictionType == "classification":
            self.nb_classes = param["DATASET"]["NB_CLASSES"]
        
        self.activeAntenna2 = param["DATASET"].get("ACTIVE_ANTENNA2", False)

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

        # Statistiques max_values et max_values2
        self.mean_max_values = np.mean(self.data_loader.get_max_values())
        self.std_max_values = np.std(self.data_loader.get_max_values())
        self.mean_max_values2 = np.mean(self.data_loader.get_max_values2())
        self.std_max_values2 = np.std(self.data_loader.get_max_values2())

        # Split des données
        self.train_indices, self.val_indices, self.test_indices = self._split_dataset_by_blocks() #self._split_dataset(shuffle, random_seed)

        if self.mode == "train":
            self.data_indices = self.train_indices
        elif self.mode == "val":
            self.data_indices = self.val_indices
        elif self.mode == "test":
            self.data_indices = self.test_indices
        else:
            raise ValueError("Le mode doit être 'train', 'val' ou 'test'.")

    def _create_file_indices(self):
        data_indices = []
        for exp in self.exp_list:
            num_images = len(self.data_loader.data["magnitudes"][exp])
            # Commencer à nb_prev_frames * time_steps pour que toutes les frames précédentes soient disponibles
            for i in range(self.nb_prev_frames * self.time_steps, num_images, self.sub_sample_factor):
                data_indices.append((exp, i))  # i = frame cible
        return np.array(data_indices)

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
    """
    def _split_dataset(self, shuffle, random_seed):
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
    """
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

        if self.use_prev_frames:
            indices = range(image_index - self.nb_prev_frames * self.time_steps, image_index + 1, self.time_steps)
        else:
            indices = [image_index]

        for i in indices:
            img_stack.append(self._load_and_preprocess_image(exp_name, i, antenna=1))
            max_stack.append(self._get_normalized_max_value(exp_name, i, antenna=1))

            if self.activeAntenna2:
                img_stack.append(self._load_and_preprocess_image(exp_name, i, antenna=2))
                max_stack.append(self._get_normalized_max_value(exp_name, i, antenna=2))

        if self.use_prev_frames:
            img_tensor = torch.tensor(np.stack(img_stack, axis=0), dtype=torch.float32)  # (T or 2T, H, W)
        else:
            # Ajouter une dimension canal si nécessaire et concaténer
            img_tensor = np.concatenate(img_stack, axis=-1)  # (H, W, C)
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)

        max_tensor = torch.tensor(max_stack, dtype=torch.float32)
        label_tensor = self._load_label(exp_name, image_index)

        return img_tensor, max_tensor, label_tensor


    def _load_and_preprocess_image(self, exp_name, image_index, antenna=1):
        if antenna == 1:
            img = self.data_loader.get_magnitude(exp_name, image_index)
        else:
            img = self.data_loader.get_magnitude2(exp_name, image_index)

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
            label = (label - self.mean_label) / self.std_label
            return torch.tensor([label], dtype=torch.float32)

    
def plot_learning_curves(train_losses, val_losses, results_path, title="Learning Curves"):
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
    os.makedirs(results_path + "/_LearningCurves", exist_ok=True)
    plt.savefig(results_path + "/_LearningCurves/learning_curve.pdf")