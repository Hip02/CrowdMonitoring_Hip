import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
#                                                                            #
# CLASS DESCRIBING AN IMPROVED CNN ARCHITECTURE FOR DOPPLER MAPS (512x512)   #
#                                                                            #
##############################################################################

class DopplerNetRegression(nn.Module):
    def __init__(self, param):
        super(DopplerNetRegression, self).__init__()
        self.nb_channels = param["MODEL"]["NB_CHANNELS"]  # Utilisé comme nb de canaux de sortie de la 1re couche

        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.nb_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)  # 512 → 128

        self.conv2 = nn.Conv2d(in_channels=self.nb_channels, out_channels=2*self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)  # 128 → 32

        self.conv3 = nn.Conv2d(in_channels=2*self.nb_channels, out_channels=4*self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32 → 16

        # Fully connected layers
        self.flatten_dim = 16 * 16 * 128
        self.fc1 = nn.Linear(self.flatten_dim + 1, 64)
        self.fc2 = nn.Linear(64, 1)  # 1 neurone en sortie pour la régression

    def forward(self, x, max_doppler):
        """
        Args:
            x: Tensor de taille (batch, 1, 512, 512) → Doppler Maps
            max_doppler: Tensor de taille (batch, 1) → Valeur max de la Doppler Map
        
        Returns:
            output: Prédiction unique du modèle (régression)
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, start_dim=1)
        max_doppler = max_doppler.view(-1, 1)
        x = torch.cat((x, max_doppler), dim=1)

        x = F.relu(self.fc1(x))
        output = self.fc2(x)

        return output