import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
#                                                                            #
# CLASS DESCRIBING AN IMPROVED CNN ARCHITECTURE FOR DOPPLER MAPS (512x512)   #
#                                                                            #
##############################################################################

class DopplerNetClassification(nn.Module):
    def __init__(self, param):
        super(DopplerNetClassification, self).__init__()
        self.nb_channels = param["MODEL"]["NB_CHANNELS"]
        self.num_classes = param["DATASET"]["NB_CLASSES"]  # Nombre de classes de sortie

        # Feature extraction (Réduction plus rapide avec des MaxPooling 4x4)
        self.conv1 = nn.Conv2d(in_channels=self.nb_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)  # Réduit 512 → 128 directement

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)  # Réduit 128 → 32

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Réduit 32 → 16

        # Fully connected layers
        self.flatten_dim = (16 * 16 * 128)  # Dimension réduite drastiquement
        self.fc1 = nn.Linear(self.flatten_dim + 1, 64)  # Réduction du nombre de neurones
        self.fc2 = nn.Linear(64, self.num_classes)

    def forward(self, x, max_doppler):
        """
        Args:
            x: Tensor de taille (batch, 1, 512, 512) -> Doppler Maps
            max_doppler: Tensor de taille (batch, 1) -> Valeur maximale de la Doppler Map
        
        Returns:
            output: Prédictions du modèle
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, start_dim=1)  # Mise en vecteur

        # Fusion avec la valeur max de la Doppler Map
        max_doppler = max_doppler.view(-1, 1)  # S'assurer que max_doppler a la bonne forme (batch, 1)
        x = torch.cat((x, max_doppler), dim=1)

        x = F.relu(self.fc1(x))
        output = self.fc2(x)

        return output