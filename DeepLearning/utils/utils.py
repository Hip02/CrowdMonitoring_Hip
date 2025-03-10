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

    def load_image(self, index, normalize=True):
        """Charge une image PNG spécifique lorsqu’elle est demandée."""
        #print(f"index type = {type(index)}, index={index}")
        if 0 <= index < len(self.file_list):
            file_path = os.path.join(self.directory, self.file_list[index])
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Chargement en niveaux de gris
            if img is not None:
                img = np.expand_dims(img, axis=-1)  # Ajout de la dimension du canal
                if normalize:
                    img = img / 255.0
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
            "min_values": {}, "max_values": {}, "labels": {}, 
            "magnitudes": {}, "phases": {}, "video_frames": {}
        }

        # Chargement immédiat des données (sauf les cartes radar)
        self._load_data(to_load)

    def _discover_experiments(self):
        """Automatically detects available experiments in the database."""
        return [exp for exp in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, exp))]

    def _load_data(self, to_load):
        """Charge certaines données immédiatement, sauf les images, avec barre de chargement."""
        if to_load is None:
            to_load = ["max_values", "labels", "magnitudes"]

        # Utilisation de tqdm pour afficher une barre de progression
        for exp in tqdm(self.exp_list, desc="🔄 Chargement des données", unit="exp"):
            if "min_values" in to_load: 
                self.data["min_values"][exp] = self._load_min_values(exp)
            if "max_values" in to_load: 
                self.data["max_values"][exp] = self._load_max_values(exp)
            if "labels" in to_load: 
                self.data["labels"][exp] = self._load_labels(exp)

            # Utilisation du proxy pour le chargement différé des images
            self.data["magnitudes"][exp] = LazyImageLoader(os.path.join(self.base_path, exp, "RadarMagnitudes"))
            self.data["phases"][exp] = LazyImageLoader(os.path.join(self.base_path, exp, "RadarPhases"))

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



class DopplerDataset(Dataset):
    """
    Dataset PyTorch qui utilise DataLoader pour charger les Doppler Maps (512x512, 1)
    avec une valeur max associée, tout en respectant le lazy loading.

    Args:
        - data_loader (DataLoader) : Instance de DataLoader contenant les fichiers et max_values.
        - mode (str) : "train", "val" ou "test" pour sélectionner la bonne partition.
        - param (dict) : Contient "TRAIN_SPLIT" pour la répartition train/val/test.
        - shuffle (bool) : Mélanger les données.
        - random_seed (int) : Permet de garder un split stable.
    """
    
    def __init__(self, data_loader, mode="train", param=None, shuffle=True, random_seed=42, sub_sample_factor=1):
        super(DopplerDataset, self).__init__()
        self.mode = mode.lower()
        self.train_split = param["DATASET"]["TRAIN_SPLIT"] if param else 0.8
        self.predictionType = param["TRAINING"]["PREDICTION_TYPE"] if param else "regression"
        if self.predictionType == "classification":
            self.nb_classes = param["DATASET"]["NB_CLASSES"] if param else 2  # Nombre de classes par défaut : 2
        self.data_loader = data_loader
        self.sub_sample_factor = sub_sample_factor

        # Liste des expériences disponibles
        self.exp_list = self.data_loader.exp_list

        # Création d'une liste de (expérience, index) pour récupérer les images
        self.data_indices = self._create_file_indices()

        # Calcul des valeurs min et max pour la classification
        if self.predictionType == "classification":
            self.min_label, self.max_label = self._compute_label_range()

        # Split train/val/test de manière stable
        self.train_indices, self.val_indices, self.test_indices = self._split_dataset(shuffle, random_seed)

        # Sélection du bon sous-ensemble
        if self.mode == "train":
            self.data_indices = self.train_indices
        elif self.mode == "val":
            self.data_indices = self.val_indices
        elif self.mode == "test":
            self.data_indices = self.test_indices
        else:
            raise ValueError("Le mode doit être 'train', 'val' ou 'test'.")

    def _compute_label_range(self):
        """Calcule et stocke les valeurs min et max des labels."""
        all_labels = self.data_loader.get_labels()
        return min(all_labels), max(all_labels)

    def _create_file_indices(self):
        """Crée une liste (expérience, index) pour le lazy loading."""
        data_indices = []
        for exp in self.exp_list:
            num_images = len(self.data_loader.data["magnitudes"][exp])  # Nombre total d'images
            data_indices.extend([(exp, i) for i in range(0, num_images, self.sub_sample_factor)])
        return np.array(data_indices)

    def _split_dataset(self, shuffle, random_seed):
        """Sépare les indices en train/val/test de manière stable."""
        np.random.seed(random_seed)  # Fixer la seed pour rendre la séparation reproductible
        indices = np.arange(len(self.data_indices))

        if shuffle:
            np.random.shuffle(indices)  # Mélanger les indices si nécessaire

        # Définition des tailles
        train_size = int(len(indices) * self.train_split)
        val_test_size = len(indices) - train_size
        val_size = val_test_size // 2  # 50% du reste pour validation
        test_size = val_test_size - val_size  # Reste pour test

        # Découpe des indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        return self.data_indices[train_indices], self.data_indices[val_indices], self.data_indices[test_indices]

    def __len__(self):
        """Retourne la taille du dataset sélectionné."""
        return len(self.data_indices)

    def __getitem__(self, idx):
        """Charge une image Doppler, sa valeur max et son label converti en classe."""
        exp_name, image_index = self.data_indices[idx]  # Récupérer l'expérience et l'index de l'image
        image_index = int(image_index)  # Convertir en entier
        #print(f"exp_name={exp_name}, image_index={image_index}")
        # Chargement lazy de l'image Doppler
        img = self.data_loader.get_magnitude(exp_name, image_index)
        if img is None:
            raise FileNotFoundError(f"Image index {image_index} non trouvée pour {exp_name}")

        img = img.astype(np.float32) / 255.0  # Normalisation

        # Récupération de la valeur max Doppler
        max_doppler = self.data_loader.get_max_values(exp_name)
        max_value = max_doppler[image_index] if image_index < len(max_doppler) else 0.0

        # Récupération du label et conversion en classe
        labels = self.data_loader.get_labels(exp_name)
        label = labels[image_index]

        if self.predictionType == "classification":
            label = self._convert_label_to_class(label)  # Transformation en classe

        # Conversion en Tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (1, 512, 512)
        max_tensor = torch.tensor([max_value], dtype=torch.float32)  # (1,)
        if self.predictionType == "classification":
            label_tensor = torch.tensor(label, dtype=torch.long)  # Label sous forme d'entier
        else:
            label_tensor = torch.tensor([label], dtype=torch.float32) # Label sous forme de float

        return img_tensor, max_tensor, label_tensor

    def _convert_label_to_class(self, label):
        """Convertit un label en classe en fonction du nombre de classes définies."""

        # Définition des intervalles pour chaque classe
        bins = np.linspace(self.min_label, self.max_label, self.nb_classes + 1)
        
        # Attribution d'une classe en fonction des intervalles
        label_class = np.digitize(label, bins, right=True) - 1  # -1 pour avoir un index de classe commençant à 0

        # S'assurer que la classe est bien entre 0 et NB_CLASSES - 1
        label_class = max(min(label_class, self.nb_classes - 1), 0)

        return label_class