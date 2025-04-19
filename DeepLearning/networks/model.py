import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import os
import copy

from networks.architectures.network_classification import DopplerNetClassification
from networks.architectures.network_regression import DopplerNetRegression, DopplerResNetRegression
from networks.architectures.network_regression import DebugResNet, SimpleCNN, SimpleCNN_2path, Homemade_CNN
from utils.utils import DopplerDataset, plot_learning_curves
import seaborn as sns
from termcolor import colored

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim
import torchvision.transforms.functional as TF
import torch.nn.functional as F


from networks.data_augment import DataAugmentor

import time
from collections import defaultdict
import random
from scipy.stats import mode

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # If you are using cuDNN, set the following flags to ensure reproducibility
    # Note: This may slow down your training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        elif method == "mode":
            agg_values.append(mode(values[mask], keepdims=False).mode)

    return unique_sec, np.array(agg_values)

def plot_predictions_vs_groundtruth(results_by_experiment, save_path):
    createFolder(save_path)

    for exp_name, data in results_by_experiment.items():
        frames = np.array(data["frames"], dtype=int)
        preds = np.array(data["preds"])
        gts = np.array(data["gts"])
        errors = preds - gts

        preds_rounded = np.maximum(np.round(preds), 0)
        gts_rounded = np.maximum(np.round(gts), 0)
        errors_rounded = preds_rounded - gts_rounded

        fps = (100/6)
        time = frames / fps
        xticks = np.linspace(time.min(), time.max(), num=10)

        # 1. Prediction vs Ground Truth (float)
        plt.figure(figsize=(12, 5))
        plt.scatter(time, gts, label="YOLO (Ground Truth)", marker='o', color='royalblue', alpha=0.7)
        plt.scatter(time, preds, label="Model Prediction", marker='x', color='darkorange', alpha=0.6)
        plt.xlabel("Time (s)")
        plt.ylabel("Number of people")
        plt.title(f"Experiment: {exp_name} — Prediction vs Ground Truth")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        plt.xticks(xticks)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{exp_name}_pred_vs_gt.pdf")
        plt.close()

        # 2. Error (float)
        plt.figure(figsize=(12, 4))
        plt.plot(time, errors, label="Prediction Error (Prediction - Ground Truth)",
                 color='crimson', marker='.', linewidth=1, alpha=0.7)
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Error")
        plt.title(f"Experiment: {exp_name} — Prediction Error over Time")
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        plt.xticks(xticks)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{exp_name}_error.pdf")
        plt.close()

        # 3. Rounded Prediction vs Ground Truth
        plt.figure(figsize=(12, 5))
        plt.scatter(time, gts_rounded, label="YOLO (Ground Truth)", marker='o', color='royalblue', alpha=0.7)
        plt.scatter(time, preds_rounded, label="Model Prediction (rounded)", marker='x', color='darkorange', alpha=0.6)
        plt.xlabel("Time (s)")
        plt.ylabel("Number of People (Rounded)")
        plt.title(f"Experiment: {exp_name} — Rounded Prediction vs Ground Truth")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        plt.xticks(xticks)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{exp_name}_rounded_pred_vs_gt.pdf")
        plt.close()

        # 4. Error (rounded)
        plt.figure(figsize=(12, 4))
        plt.plot(time, errors_rounded, label="Rounded Prediction Error",
                 color='teal', marker='.', linewidth=1, alpha=0.7)
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel("Time (s)")
        plt.ylabel("Error (rounded)")
        plt.title(f"Experiment: {exp_name} — Rounded Prediction Error over Time")
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        plt.xticks(xticks)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{exp_name}_rounded_error.pdf")
        plt.close()

        # 5. Aggregated (per second) Prediction vs Ground Truth
        sec, pred_avg = aggregate_by_second(time, preds, fps)
        _, gt_avg = aggregate_by_second(time, gts, fps)

        plt.figure(figsize=(12, 5))
        plt.plot(sec, gt_avg, label="YOLO (GT) - avg/sec", linestyle='--', marker='o')
        plt.plot(sec, pred_avg, label="Model Prediction - avg/sec", linestyle='-', marker='x')
        plt.xlabel("Time (s)")
        plt.ylabel("Avg. People Count")
        plt.title(f"Experiment: {exp_name} — Aggregated Prediction vs Ground Truth (per second)")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{exp_name}_agg_pred_vs_gt.pdf")
        plt.close()

        # 6. Heatmap of error over time
        plt.figure(figsize=(12, 1.5))
        plt.imshow([errors], cmap='coolwarm', aspect='auto', extent=[time.min(), time.max(), 0, 1])
        plt.colorbar(label="Prediction Error")
        plt.yticks([])
        plt.xlabel("Time (s)")
        plt.title(f"Experiment: {exp_name} — Error Heatmap")
        plt.tight_layout()
        plt.savefig(f"{save_path}/{exp_name}_error_heatmap.pdf")
        plt.close()

        # Save the results in raw numpy format
        np.save(f"{save_path}/{exp_name}_frames.npy",frames)
        np.save(f"{save_path}/{exp_name}_preds.npy", preds)
        np.save(f"{save_path}/{exp_name}_gts.npy", gts)
        np.save(f"{save_path}/{exp_name}_errors.npy", errors)
        np.save(f"{save_path}/{exp_name}_errors_rounded.npy", errors_rounded)
        np.save(f"{save_path}/{exp_name}_preds_rounded.npy", preds_rounded)
        np.save(f"{save_path}/{exp_name}_gts_rounded.npy", gts_rounded)


class Network_Class:
    def __init__(self, data_loader, param, resultsPath, sub_sample_factor=1):

        set_seed(42)

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
           
            self.model = DopplerResNetRegression(param, layers=layers).to(self.device)
            self.criterion = nn.MSELoss()

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
        best_loss = np.Inf
        val_losses = []
        train_losses = []

        start_time = time.time()

        for i in range(self.epoch):
            self.model.train(True)
            train_loss = 0.0
            total_train_samples = 0

            for batch_idx, (image_magnitude, max_doppler, labels) in enumerate(self.trainDataLoader):
                image_magnitude = image_magnitude.to(self.device)
                max_doppler = max_doppler.to(self.device)
                labels = labels.to(self.device)

                if self.data_augm:
                    data_augmentor = DataAugmentor(self.config, self.trainDataLoader)
                    image_magnitude, max_doppler, labels = data_augmentor.apply(image_magnitude, max_doppler, labels)

                if self.predictionType == "regression":
                    labels = labels.view(-1, 1)


                self.optimizer.zero_grad()
                outputs = self.model(image_magnitude, max_doppler)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * labels.size(0)
                total_train_samples += labels.size(0)

                # Every 10 batches, print that the training is in progress
                if batch_idx % 100 == 0:
                    print(f"🔄 Training... | Epoch {i+1}/{self.epoch} | Batch {batch_idx}/{len({self.trainDataLoader})} (time = {time.time() - start_time:.2f})")

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
                    max_doppler = max_doppler.to(self.device)
                    if self.predictionType == "regression":
                        labels = labels.view(-1, 1)
                    outputs = self.model(image_magnitude, max_doppler)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item() * labels.size(0)
                    total_val_samples += labels.size(0)

            mean_val_loss = val_loss / total_val_samples
            val_losses.append(mean_val_loss)

            # Affichage simple
            print(f"✅ Epoch {i+1} terminée | Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f}")

            # Courbes d'apprentissage
            plot_learning_curves(train_losses, val_losses, self.resultsPath)

            if True: #mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model = copy.deepcopy(self.model)
                wghtsPath = self.resultsPath + '/_Weights/'
                createFolder(wghtsPath)
                torch.save(best_model.state_dict(), wghtsPath + '/wghts.pkl')
                print("💾 Modèle sauvegardé (à chaque epoch)")

                # Save losses as numpy arrays
                np.save(os.path.join(self.resultsPath, 'train_losses.npy'), np.array(train_losses))
                np.save(os.path.join(self.resultsPath, 'val_losses.npy'), np.array(val_losses))

        return train_losses, val_losses


    
    def test(self):
        self.model.eval()

        if self.predictionType == "regression":
            total_loss = 0.0
            total_samples = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():

                worst_losses = []  # Liste de tuples : (loss, pred, label, image)
                sample_index = 0

                results_by_experience = {}

                progress_bar = tqdm(self.testDataLoader, desc="Testing (regression)", unit="batch")
                for image_magnitude, max_doppler, labels in progress_bar:
                    image_magnitude = image_magnitude.to(self.device)
                    max_doppler = max_doppler.to(self.device)
                    labels = labels.view(-1, 1).to(self.device)

                    ### DEBUG ###
                    #outputs = self.model(image_magnitude)
                    outputs = self.model(image_magnitude, max_doppler)
                    loss = self.criterion(outputs, labels)

                    # Store the worst losses
                    individual_losses = F.mse_loss(outputs, labels, reduction='none').view(-1).cpu().numpy()
                    preds = outputs.detach().cpu().numpy().flatten()
                    lbls = labels.cpu().numpy().flatten()
                    imgs = image_magnitude.cpu().numpy()

                    for i in range(len(preds)):
                        index_i = sample_index + i  # index local dans le batch
                        # Récupération du nom d'expérience et de la frame
                        exp_name, frame_idx = self.testDataLoader.dataset.get_exp_and_frame(index_i)
                        pred_denorm = preds[i] * self.dataSetTest.std_label + self.dataSetTest.mean_label
                        label_denorm = lbls[i] * self.dataSetTest.std_label + self.dataSetTest.mean_label

                        if exp_name not in results_by_experience:
                            results_by_experience[exp_name] = {
                                'frames': [],
                                'preds': [],
                                'gts': []
                            }

                        results_by_experience[exp_name]['frames'].append(frame_idx)
                        results_by_experience[exp_name]['preds'].append(pred_denorm)
                        results_by_experience[exp_name]['gts'].append(label_denorm)

                        loss_i = individual_losses[i]
                        case_index = sample_index + i  # index global dans le dataset
                        case = (loss_i, preds[i], lbls[i], imgs[i], case_index)
                        if len(worst_losses) < 5:
                            worst_losses.append(case)
                            worst_losses.sort(reverse=True, key=lambda x: x[0])
                        elif loss_i > worst_losses[-1][0]:
                            worst_losses[-1] = case
                            worst_losses.sort(reverse=True, key=lambda x: x[0])

                    sample_index += len(preds)

                    
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
            print(f"📈 Predictions vs True Labels Plotted")

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
            print(f"📊 Confusion Matrix Plotted")

            # 📊 Barplot des erreurs (%) par label entier
            errors_per_label = {}
            total_per_label = {}

            for true, pred in zip(all_labels, rounded_preds):
                total_per_label[true] = total_per_label.get(true, 0) + 1
                if true != pred:
                    errors_per_label[true] = errors_per_label.get(true, 0) + 1

            labels_sorted = sorted(total_per_label.keys())
            error_rates = [(errors_per_label.get(label, 0) / total_per_label[label]) * 100 for label in labels_sorted]

            plt.figure(figsize=(10, 6))
            sns.barplot(x=labels_sorted, y=error_rates, color="salmon")
            plt.ylabel("Error (%)")
            plt.xlabel("Label")
            plt.title("Error rate per label (%)")
            plt.grid(True)
            plt.tight_layout()
            createFolder(self.resultsPath + "/_ErrorAnalysis/")
            plt.savefig(self.resultsPath + "/_ErrorAnalysis/error_per_label.pdf")
            print(f"📊 Error Rate per Label Plotted")

            # 📉 Analyse de calibration (erreur moyenne vs vrai label)
            calibration_bias = {}
            count_per_label = {}

            for true, pred in zip(all_labels, all_preds):
                calibration_bias[true] = calibration_bias.get(true, 0) + (pred - true)
                count_per_label[true] = count_per_label.get(true, 0) + 1

            bias_per_label = [calibration_bias[l] / count_per_label[l] for l in labels_sorted]

            plt.figure(figsize=(10, 6))
            sns.barplot(x=labels_sorted, y=bias_per_label, color="skyblue")
            plt.axhline(0, color='black', linestyle='--')
            plt.ylabel("Average bias (predicted - true)")
            plt.xlabel("Label")
            plt.title("Calibration analysis")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.resultsPath + "/_ErrorAnalysis/calibration_bias.pdf")
            print(f"📊 Calibration Analysis Plotted")

            # 📊 Distribution des erreurs absolues
            errors = np.abs(all_preds - all_labels)

            plt.figure(figsize=(8, 6))
            sns.histplot(errors, bins=30, kde=True, color="purple")
            plt.xlabel("Absolute Error")
            plt.ylabel("Number of samples")
            plt.title("Distribution of absolute errors")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.resultsPath + "/_ErrorAnalysis/error_distribution.pdf")
            print(f"📊 Error Distribution Plotted")

            # Create directory for failure cases
            createFolder(self.resultsPath + "/_FailureCases/")

            # 📷 Visualisation des pires échecs avec informations d'expérience/frame
            for idx, (loss_i, pred_i, label_i, img_i, index_i) in enumerate(worst_losses):

                # Récupération du nom d'expérience et de la frame
                exp_name, frame_idx = self.testDataLoader.dataset.get_exp_and_frame(index_i)

                # Denormalize
                pred_i = pred_i * self.dataSetTest.std_label + self.dataSetTest.mean_label
                label_i = label_i * self.dataSetTest.std_label + self.dataSetTest.mean_label

                plt.figure(figsize=(6, 5))
                plt.imshow(img_i[0], cmap='viridis')
                plt.title(f"Failure Case #{idx+1}\nExp: {exp_name}, Frame: {frame_idx}\nTrue: {label_i:.1f} | Pred: {pred_i:.1f}")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"{self.resultsPath}/_FailureCases/case_{idx+1}.pdf")
                plt.close()

            
                print(f"📷 Failure case #{idx+1} visualized")

            
            plot_predictions_vs_groundtruth(
                results_by_experience,
                save_path=self.resultsPath + "/_PerExperimentPlots/"
            )
            print("📊 Prediction vs Ground Truth Plots per Experiment saved.")



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

                    outputs = self.model(image_magnitude, max_doppler) #, max_doppler)
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


    def visualize_data_augmentation(self, num_images=4, idx=0):
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

        data_augmentor = DataAugmentor(self.config, self.trainDataLoader)

        # Récupérer un batch
        image_magnitude, max_doppler, label = next(iter(self.trainDataLoader))

        # Ne pas dépasser la taille du batch
        num_images = min(num_images, image_magnitude.shape[0])

        # Sélection des images à afficher
        images_to_show = image_magnitude[:num_images]
        max_doppler_to_show = max_doppler[:num_images]
        labels_to_show = label[:num_images]

        # Appliquer les augmentations
        augmented_images, augmented_max_doppler, augmented_labels = data_augmentor.apply(images_to_show, max_doppler_to_show, labels_to_show)

        # Passage en numpy
        originals = images_to_show.cpu().numpy()
        augmenteds = augmented_images.cpu().numpy()
        max_dopplers = max_doppler_to_show.cpu().numpy()
        labels_to_show = labels_to_show.cpu().numpy()
        augmented_max_dopplers = augmented_max_doppler.cpu().numpy()
        augmented_labels_to_show = augmented_labels.cpu().numpy()

        # Création de la figure
        fig, axes = plt.subplots(num_images, 2, figsize=(10, 3 * num_images))
        for i in range(num_images):
            # Original
            axes[i][0].imshow(originals[i][0], cmap='gray')
            axes[i][0].set_title(f"Original - Label = {labels_to_show[i]:.2f} ({int(labels_to_show[i] * self.dataSetTrain.std_label + self.dataSetTrain.mean_label)} people)")
            axes[i][0].axis("off")

            # Augmentée
            axes[i][1].imshow(augmenteds[i][0], cmap='gray')
            axes[i][1].set_title(f"Augmented - Label = {augmented_labels_to_show[i]:.2f} ({int(augmented_labels_to_show[i] * self.dataSetTrain.std_label + self.dataSetTrain.mean_label)} people)")
            axes[i][1].axis("off")

        plt.tight_layout()
        dir_to_create = self.resultsPath + "/_Augmentation/"
        createFolder(dir_to_create)
        plt.savefig("" + self.resultsPath + f"/_Augmentation/augmented_images{idx}.pdf")