import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import os
import copy

from networks.architectures.network_classification import DopplerNetClassification
from networks.architectures.network_regression import DopplerNetRegression, DopplerResNetRegression
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
        self.resultsPath    = resultsPath
        self.epoch          = param["TRAINING"]["EPOCH"]
        self.device         = param["TRAINING"]["DEVICE"]
        self.lr             = param["TRAINING"]["LEARNING_RATE"]
        self.batchSize      = param["TRAINING"]["BATCH_SIZE"]
        self.predictionType = param["TRAINING"]["PREDICTION_TYPE"]

        if self.predictionType == "classification":
            self.model = DopplerNetClassification(param).to(self.device)
            self.criterion = nn.CrossEntropyLoss()
        if self.predictionType == "regression":
            self.model = DopplerResNetRegression(param, layers=[2, 2, 2, 2]).to(self.device)
            self.criterion = nn.MSELoss()
        if self.predictionType == "regression_temporal":
            self.model = DopplerResNetRegression(param, layers=[2, 2, 2, 2]).to(self.device)
            self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.dataSetTrain = DopplerDataset(data_loader, mode='train', param=param, sub_sample_factor=sub_sample_factor)
        self.dataSetVal = DopplerDataset(data_loader, mode='val', param=param, sub_sample_factor=sub_sample_factor)
        self.dataSetTest = DopplerDataset(data_loader, mode='test', param=param, sub_sample_factor=sub_sample_factor)

        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=0)
        self.valDataLoader = DataLoader(self.dataSetVal, batch_size=self.batchSize, shuffle=False, num_workers=0)
        self.testDataLoader = DataLoader(self.dataSetTest, batch_size=self.batchSize, shuffle=False, num_workers=0)

        print("✅ Network Initialized")

    def loadWeight(self):
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl', map_location=torch.device(self.device)))


    def train(self):
        best_loss = np.Inf
        val_losses = []
        train_losses = []

        for i in range(self.epoch):
            self.model.train(True)
            train_loss = 0.0
            total_train_samples = 0

            progress_bar = tqdm(self.trainDataLoader, desc=f"🟢 Epoch {i+1}/{self.epoch}", unit="batch", leave=True)

            for image_magnitude, max_doppler, labels in progress_bar:
                image_magnitude = image_magnitude.to(self.device)
                max_doppler = max_doppler.to(self.device)

                if self.predictionType == "regression" or self.predictionType == "regression_temporal":
                    labels = labels.view(-1, 1)

                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(image_magnitude, max_doppler)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                batch_size = labels.size(0)
                train_loss += loss.item() * batch_size  # somme des pertes
                total_train_samples += batch_size

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            mean_train_loss = train_loss / total_train_samples
            train_losses.append(mean_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            total_val_samples = 0

            with torch.no_grad():
                for image_magnitude, max_doppler, labels in self.valDataLoader:
                    image_magnitude = image_magnitude.to(self.device)
                    max_doppler = max_doppler.to(self.device)

                    if self.predictionType == "regression" or self.predictionType == "regression_temporal":
                        labels = labels.view(-1, 1)

                    labels = labels.to(self.device)
                    outputs = self.model(image_magnitude, max_doppler)
                    loss = self.criterion(outputs, labels)

                    batch_size = labels.size(0)
                    val_loss += loss.item() * batch_size
                    total_val_samples += batch_size

            mean_val_loss = val_loss / total_val_samples
            val_losses.append(mean_val_loss)

            print(f"✅ Epoch {i + 1}/{self.epoch} | Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f}")

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

        if self.predictionType == "regression" or self.predictionType == "regression_temporal":
            total_loss = 0.0
            total_samples = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                progress_bar = tqdm(self.testDataLoader, desc="Testing (regression)", unit="batch")
                for image_magnitude, max_doppler, labels in progress_bar:
                    image_magnitude = image_magnitude.to(self.device)
                    max_doppler = max_doppler.to(self.device)
                    labels = labels.view(-1, 1).to(self.device)

                    outputs = self.model(image_magnitude, max_doppler)
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
            plt.scatter(all_labels, all_preds, alpha=0.6)
            plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 'r--', label='Perfect prediction')
            plt.xlabel('True labels')
            plt.ylabel('Predicted labels')
            plt.title('Regression: Predictions vs True Labels')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

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
                    max_doppler = max_doppler.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(image_magnitude, max_doppler)
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
            img = images_to_show[i, 0, :, :]  # Extraire l'image Doppler (1 canal)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Label: {labels_to_show[i]}\nMax Doppler: {max_doppler_to_show[i][0]:.7f}")
            axes[i].axis("off")

        plt.show()