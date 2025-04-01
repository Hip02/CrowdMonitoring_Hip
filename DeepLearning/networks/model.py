import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import os
import copy

from networks.architectures.network_classification import DopplerNetClassification
from networks.architectures.network_regression import DopplerNetRegression, DopplerResNetRegression, DopplerResNet50Regression
from networks.architectures.network_regression import DebugResNet, SimpleCNN, SimpleCNN_2path
from utils.utils import DopplerDataset, plot_learning_curves
import seaborn as sns
from termcolor import colored

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim
import torchvision.transforms.functional as TF

from networks.data_augment import DataAugmentor

import time
from collections import defaultdict

# Fix seed
torch.manual_seed(42)
np.random.seed(42)

def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)

class Network_Class:
    def __init__(self, data_loader, param, resultsPath, sub_sample_factor=1):
        self.resultsPath    = resultsPath
        self.config         = param
        self.epoch          = param["TRAINING"]["EPOCH"]
        self.device         = param["TRAINING"]["DEVICE"]
        self.lr             = param["TRAINING"].get("LEARNING_RATE", None)
        self.maxlr          = param["TRAINING"].get("MAX_LEARNING_RATE", None)
        self.lr_type        = param["TRAINING"].get("LEARNING_RATE_TYPE", "constant").lower()
        self.gamma          = param["TRAINING"].get("LR_GAMMA", None)
        self.batchSize      = param["TRAINING"]["BATCH_SIZE"]
        self.predictionType = param["TRAINING"]["PREDICTION_TYPE"]
        self.resnet_type    = param["TRAINING"].get("RESNET_TYPE", "resnet18")

        if param.get("augmentation", None) is not None:
            self.data_augm = True
        else:
            self.data_augm = False

        # Data Loaders
        self.dataSetTrain = DopplerDataset(data_loader, mode='train', param=param, sub_sample_factor=sub_sample_factor)
        self.dataSetVal = DopplerDataset(data_loader, mode='val', param=param, sub_sample_factor=sub_sample_factor)
        self.dataSetTest = DopplerDataset(data_loader, mode='test', param=param, sub_sample_factor=sub_sample_factor)

        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.valDataLoader = DataLoader(self.dataSetVal, batch_size=self.batchSize, shuffle=False, num_workers=4)
        self.testDataLoader = DataLoader(self.dataSetTest, batch_size=self.batchSize, shuffle=False, num_workers=4)

        # Model & Loss
        if self.predictionType == "classification":
            self.model = DopplerNetClassification(param).to(self.device)
            self.criterion = nn.CrossEntropyLoss()
        elif self.predictionType == "regression":

            if self.resnet_type == "resnet18":
                layers = [2, 2, 2, 2]
            elif self.resnet_type == "resnet34":
                layers = [3, 4, 6, 3]
            elif self.resnet_type == "resnet_XS":
                layers = [1, 1, 1, 1]
                
            #self.model = DebugResNet(param).to(self.device)
            #self.criterion = nn.MSELoss()

            self.model = SimpleCNN_2path(param).to(self.device)
            self.criterion = nn.MSELoss()

            #self.model = DopplerResNetRegression(param, layers=layers).to(self.device)
            #self.criterion = nn.MSELoss()

        # Optimizer (initial LR is required for all schedulers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # LR Scheduler
        self.scheduler = None
        if self.lr_type == "onecycle":
            if self.maxlr is None:
                raise ValueError("MAX_LEARNING_RATE must be defined for OneCycleLR")
            steps_per_epoch = len(self.trainDataLoader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.maxlr,
                epochs=self.epoch,
                steps_per_epoch=steps_per_epoch,
                anneal_strategy="linear"
            )
        elif self.lr_type == "explr":
            if self.gamma is None:
                raise ValueError("LR_GAMMA must be defined for ExponentialLR")
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.gamma
            )
        elif self.lr_type == "constant":
            self.scheduler = None  # pas de scheduler utilisé
        else:
            raise ValueError(f"Unknown LEARNING_RATE_TYPE '{self.lr_type}'. Choose among: constant, explr, onecycle.")


        
        print(colored("\n" + "="*60, "blue"))
        print(colored("                 NETWORK INITIALIZATION SUMMARY", "green", attrs=["bold"]))
        print(colored("="*60, "blue"))

        print(colored("→ Device", "cyan") + "                  : " + colored(f"{self.device}", "white", attrs=["bold"]))
        print(colored("→ Results path", "cyan") + "            : " + colored(f"{self.resultsPath}", "yellow"))
        print(colored("→ Epochs", "cyan") + "                  : " + colored(f"{self.epoch}", "yellow"))
        print(colored("→ Learning rate", "cyan") + "           : " + colored(f"{self.lr}", "yellow"))
        print(colored("→ Batch size", "cyan") + "              : " + colored(f"{self.batchSize}", "yellow"))
        print(colored("→ Prediction type", "cyan") + "         : " + colored(f"{self.predictionType}", "green" if self.predictionType=="regression" else "magenta", attrs=["bold"]))
        print(colored("→ Data augmentation", "cyan") + "       : " + colored(str(self.data_augm), "green" if self.data_augm else "red"))

        # Scheduler info (if exists)
        if self.lr_type != "constant":
            print(colored("→ LR scheduler", "cyan") + "            : " + colored(f"{self.lr_type}", "yellow"))
            if self.lr_type == "onecycle":
                print(colored("→ Max learning rate", "cyan") + "       : " + colored(f"{self.maxlr}", "yellow"))
            elif self.lr_type == "explr":
                print(colored("→ LR gamma", "cyan") + "               : " + colored(f"{self.gamma}", "yellow"))

        print(colored("="*60, "blue") + "\n")


    def loadWeight(self):
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl', map_location=torch.device(self.device)))

    def train(self):

        max_batches_debug = 100

        best_loss = np.Inf
        val_losses = []
        train_losses = []

        # Initialiser le profiler
        profiling = defaultdict(float)
        total_batches = len(self.trainDataLoader) * self.epoch

        for i in range(self.epoch):
            self.model.train(True)
            train_loss = 0.0
            total_train_samples = 0

            total_steps = len(self.trainDataLoader)
            progress_bar = tqdm(total=total_steps, desc=f"🟢 Epoch {i+1}/{self.epoch}", unit="batch", leave=True)

            for batch_idx, (image_magnitude, max_doppler, labels) in enumerate(self.trainDataLoader):
                t0 = time.time()
                image_magnitude = image_magnitude.to(self.device)
                max_doppler = max_doppler.to(self.device)
                labels = labels.to(self.device)
                profiling["data_loading"] += time.time() - t0

                t1 = time.time()
                if self.data_augm:
                    data_augmentor = DataAugmentor(config=self.config)
                    image_magnitude = data_augmentor.apply(image_magnitude)
                profiling["augmentation"] += time.time() - t1

                if self.predictionType == "regression":
                    labels = labels.view(-1, 1)

                t2 = time.time()
                self.optimizer.zero_grad()
                outputs = self.model(image_magnitude)
                profiling["forward"] += time.time() - t2

                t3 = time.time()
                loss = self.criterion(outputs, labels)
                profiling["loss"] += time.time() - t3

                t4 = time.time()
                loss.backward()
                profiling["backward"] += time.time() - t4

                t5 = time.time()
                self.optimizer.step()
                profiling["optimizer"] += time.time() - t5

                t6 = time.time()
                #train_loss += loss.item() * labels.size(0)
                #total_train_samples += labels.size(0)
                #progress_bar.update(1)
                #progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                profiling["progress_bar"] += time.time() - t6

                if max_batches_debug is not None and batch_idx + 1 >= max_batches_debug:
                    print(f"\n🛑 Entraînement interrompu après {max_batches_debug} batchs pour debug.")
                    break


            # Profiling résumé
            print("\n🕵️ Profiling entraînement (temps cumulé par phase) :")
            total_time = sum(profiling.values())
            for k, v in profiling.items():
                print(f"{k.replace('_', ' ').capitalize():<15}: {v:.2f}s ({(v/total_time)*100:.1f}%)")


            mean_train_loss = train_loss / total_train_samples
            train_losses.append(mean_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for image_magnitude, max_doppler, labels in self.valDataLoader:
                    image_magnitude = image_magnitude.to(self.device)
                    labels = labels.to(self.device)
                    if self.predictionType == "regression":
                        labels = labels.view(-1, 1)
                    outputs = self.model(image_magnitude)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * labels.size(0)

            mean_val_loss = val_loss / total_val_samples
            val_losses.append(mean_val_loss)

            print(f"✅ Epoch {i + 1}/{self.epoch} | Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f}")

            # plot learning curves at each epoch
            plot_learning_curves(train_losses, val_losses, self.resultsPath)

            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model = copy.deepcopy(self.model)
                wghtsPath = self.resultsPath + '/_Weights/'
                createFolder(wghtsPath)
                torch.save(best_model.state_dict(), wghtsPath + '/wghts.pkl')
                print("Model saved")

        return train_losses, val_losses


    
    def test(self):
        self.model.eval()

        if self.predictionType == "regression":
            total_loss = 0.0
            total_samples = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                progress_bar = tqdm(self.testDataLoader, desc="Testing (regression)", unit="batch")
                for image_magnitude, max_doppler, labels in progress_bar:
                    image_magnitude = image_magnitude.to(self.device)
                    #max_doppler = max_doppler.to(self.device)
                    labels = labels.view(-1, 1).to(self.device)

                    ### DEBUG ###
                    outputs = self.model(image_magnitude)
                    #outputs = self.model(image_magnitude, max_doppler)
                    loss = self.criterion(outputs, labels)

                    batch_size = labels.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

                    all_preds.extend(outputs.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())

                    progress_bar.set_postfix(mse=f"{loss.item():.4f}")

            mean_loss = total_loss / total_samples
            print(f"📏 Test MSE (moyenne par échantillon): {mean_loss:.4f}")

            # Denormalization
            all_preds = np.array(all_preds) * self.dataSetTest.std_label + self.dataSetTest.mean_label
            all_labels = np.array(all_labels) * self.dataSetTest.std_label + self.dataSetTest.mean_label

            # Compute denormalized MSE
            denorm_mse = np.mean((all_preds - all_labels) ** 2)
            print(f"📏 Test MSE (denormalized): {denorm_mse:.4f}")

            # 📈 Scatter plot: predictions vs true labels
            plt.figure(figsize=(8, 6))
            plt.scatter(all_labels, all_preds, alpha=0.1)
            plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 'r--', label='Perfect prediction')
            plt.xlabel('True labels')
            plt.ylabel('Predicted labels')
            plt.title('Regression: Predictions vs True Labels')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Add denorm_mse on the plot 
            plt.text(0.95, 0.05, f"Denormalized MSE: {denorm_mse:.4f}", ha='right', va='bottom', transform=plt.gca().transAxes, color='red')

            # Os make dir
            createFolder(self.resultsPath + "/_Predictions/")
            plt.savefig(self.resultsPath + "/_Predictions/predictions.pdf")

            # Ajout 1 : arrondi à l'entier le plus proche
            rounded_preds = np.round(all_preds).astype(int)
            all_labels = np.round(all_labels).astype(int)

            # Ajout 2 : génération et affichage de la matrice de confusion
            labels = list(range(min(all_labels), max(all_labels) + 1))

            cm = confusion_matrix(all_labels, rounded_preds, labels=labels)
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', mask=(cm==0), cbar=True)
            plt.gca().invert_yaxis()
            plt.title('Confusion Matrix (Rounded Predictions)')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.tight_layout()

            # Os make dir
            createFolder(self.resultsPath + '/_ConfusionMatrix/')
            plt.savefig(self.resultsPath + '/_ConfusionMatrix/confusion_matrix.pdf')

            return mean_loss

        else:  # Classification
            correct_predictions = 0
            total_samples = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                progress_bar = tqdm(self.testDataLoader, desc="Testing (classification)", unit="batch")
                for image_magnitude, max_doppler, labels in progress_bar:
                    image_magnitude = image_magnitude.to(self.device)
                    #max_doppler = max_doppler.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(image_magnitude) #, max_doppler)
                    _, preds = torch.max(outputs, 1)

                    correct_predictions += (preds == labels).sum().item()
                    total_samples += labels.size(0)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    acc = correct_predictions / total_samples
                    progress_bar.set_postfix(acc=f"{acc:.4f}")

            accuracy = correct_predictions / total_samples
            print(f"✅ Test Accuracy: {accuracy:.4f}")

            # 📊 Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.title('Classification: Confusion Matrix')
            plt.tight_layout()
            plt.show()

        return accuracy
    def visualize_batch(self, num_images=4):
        """
        Affiche un batch d'images Doppler avec leurs labels et valeurs max Doppler.
        
        Args:
            num_images (int): Nombre d'images à afficher (doit être <= batchSize)
        """
        print("🔄 Visualizing Batch")
        # Récupérer un batch de données
        image_magnitude, max_doppler, labels = next(iter(self.trainDataLoader))
        
        # Affichage des dimensions
        print(f"Image Magnitude Shape: {image_magnitude.shape}")  # (batch_size, channels, 512, 512)
        print(f"Max Doppler Shape: {max_doppler.shape}")  # (batch_size, 1)
        print(f"Labels Shape: {labels.shape}")  # (batch_size,)

        # Sélectionner les `num_images` premières images du batch
        num_images = min(num_images, image_magnitude.shape[0])  # Éviter les dépassements

        # Convertir en format affichable
        images_to_show = image_magnitude[:num_images].cpu().numpy()
        max_doppler_to_show = max_doppler[:num_images].cpu().numpy()
        labels_to_show = labels[:num_images].cpu().numpy()

        # Affichage des images
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        if num_images == 1:
            axes = [axes]  # Convertir en liste si une seule image

        for i in range(num_images):
            denormalized_label = labels_to_show[i] * self.dataSetTrain.std_label + self.dataSetTrain.mean_label
            img = images_to_show[i, 0, :, :]  # Extraire l'image Doppler (1 canal)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Label: {labels_to_show[i]} ({int(denormalized_label)})\nMax Doppler: {max_doppler_to_show[i][0]:.7f}")
            axes[i].axis("off")

        plt.show()


    def visualize_data_augmentation(self, num_images=4):
        """
        Affiche les images originales et leurs versions augmentées côte à côte.

        Args:
            data_augmentor: instance de DataAugmentor avec méthode .apply()
            dataloader: DataLoader contenant les images d'entrée
            num_images (int): nombre de lignes (chaque ligne = original + augmentée)
        """
        print("🎨 Visualizing Data Augmentation (original vs augmented)")

        # Check if data augmentation is enabled
        if not self.data_augm:
            print("Data augmentation is not enabled in the configuration file.")
            return

        data_augmentor = DataAugmentor(config=self.config)

        # Récupérer un batch
        image_magnitude, _, _ = next(iter(self.trainDataLoader))

        # Ne pas dépasser la taille du batch
        num_images = min(num_images, image_magnitude.shape[0])

        # Sélection des images à afficher
        images_to_show = image_magnitude[:num_images]

        # Appliquer les augmentations
        augmented_images = torch.stack([data_augmentor.apply(img.unsqueeze(0)).squeeze(0) for img in images_to_show])

        # Passage en numpy
        originals = images_to_show.cpu().numpy()
        augmenteds = augmented_images.cpu().numpy()

        # Création de la figure
        fig, axes = plt.subplots(num_images, 2, figsize=(6, 3 * num_images))
        for i in range(num_images):
            # Original
            axes[i][0].imshow(originals[i][0], cmap='gray')
            axes[i][0].set_title("Original")
            axes[i][0].axis("off")

            # Augmentée
            axes[i][1].imshow(augmenteds[i][0], cmap='gray')
            axes[i][1].set_title("Augmented")
            axes[i][1].axis("off")

        plt.tight_layout()
        plt.show()