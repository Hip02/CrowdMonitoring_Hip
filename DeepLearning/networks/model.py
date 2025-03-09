import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy

from networks.architectures.network import DopplerNet
from utils.utils import DopplerDataset

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim

def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)

class Network_Class: 
    def __init__(self, data_loader, param, resultsPath, sub_sample_factor=1):
        self.resultsPath   = resultsPath
        self.epoch         = param["TRAINING"]["EPOCH"]
        self.device        = param["TRAINING"]["DEVICE"]
        self.lr            = param["TRAINING"]["LEARNING_RATE"]
        self.batchSize     = param["TRAINING"]["BATCH_SIZE"]

        self.model = DopplerNet(param).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.dataSetTrain = DopplerDataset(data_loader, mode='train', param=param, sub_sample_factor=sub_sample_factor)
        self.dataSetVal = DopplerDataset(data_loader, mode='val', param=param, sub_sample_factor=sub_sample_factor)
        self.dataSetTest = DopplerDataset(data_loader, mode='test', param=param, sub_sample_factor=sub_sample_factor)

        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=0)
        self.valDataLoader = DataLoader(self.dataSetVal, batch_size=self.batchSize, shuffle=False, num_workers=0)
        self.testDataLoader = DataLoader(self.dataSetTest, batch_size=self.batchSize, shuffle=False, num_workers=0)

        print("✅ Network Initialized")

    def loadWeight(self):
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl'))


    def train(self):
        best_loss = np.Inf
        val_losses = []
        train_losses = []

        for i in range(self.epoch):
            self.model.train(True)
            train_loss = 0.0

            # Barre de progression pour afficher l'entraînement batch par batch
            progress_bar = tqdm(self.trainDataLoader, desc=f"🟢 Epoch {i+1}/{self.epoch}", unit="batch", leave=True)

            for image_magnitude, max_doppler, labels in progress_bar:
                image_magnitude = image_magnitude.to(self.device)
                max_doppler = max_doppler.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(image_magnitude, max_doppler)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                # Mise à jour dynamique de la barre de progression avec la perte
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_losses.append(train_loss)
        
            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for (image_magnitude, max_doppler, labels) in self.valDataLoader:
                    image_magnitude = image_magnitude.to(self.device)
                    max_doppler = max_doppler.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(image_magnitude, max_doppler)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
            
            val_losses.append(val_loss)

            print(f"✅ Epoch {i + 1}/{self.epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(self.model)

                # Save the model weights
                wghtsPath  = self.resultsPath + '/_Weights/'
                createFolder(wghtsPath)
                torch.save(best_model.state_dict(), wghtsPath + '/wghts.pkl')
                print("Model saved")

        return train_losses, val_losses
    
    def test(self):
        self.model.eval()  # Mode évaluation

        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():  # Désactiver le calcul des gradients pour accélérer l'inférence
            progress_bar = tqdm(self.testDataLoader, desc="Testing", unit="batch")  # Barre de progression
            for (image_magnitude, max_doppler, labels) in progress_bar:
                image_magnitude = image_magnitude.to(self.device)
                max_doppler = max_doppler.to(self.device)
                labels = labels.to(self.device)

                # Prédictions
                outputs = self.model(image_magnitude, max_doppler)
                _, preds = torch.max(outputs, 1)

                # Calcul du nombre de prédictions correctes
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

                # Mise à jour du texte de la barre de progression
                progress_bar.set_postfix(acc=f"{correct_predictions / total_samples:.4f}")
                
        # Calcul de la précision totale
        accuracy = correct_predictions / total_samples
        print(f"Test Accuracy: {accuracy:.4f}")

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
            img = images_to_show[i, 0, :, :]  # Extraire l'image Doppler (1 canal)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Label: {labels_to_show[i]}\nMax Doppler: {max_doppler_to_show[i][0]:.7f}")
            axes[i].axis("off")

        plt.show()