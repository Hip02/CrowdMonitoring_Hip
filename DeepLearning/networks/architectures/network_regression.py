import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
#                                                                            #
# CLASS DESCRIBING AN IMPROVED CNN ARCHITECTURE FOR DOPPLER MAPS (512x512)   #
#                                                                            #
##############################################################################


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class DopplerResNetRegression(nn.Module):
    def __init__(self, param, layers, block=ResidualBlock):
        super(DopplerResNetRegression, self).__init__()
        self.inplanes = param["MODEL"]["NB_CHANNELS"]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 1 canal en entrée
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # ↓ taille x4 ici
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # peu importe la taille finale, ça fera [B,512,1,1]

        # ⚠️ On augmente la taille d'entrée du FC : 512 (features) + 1 (max_doppler)
        self.fc = nn.Linear(512 + 1, 1)  # régression → une seule sortie

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, max_doppler):
        x = self.conv1(x)       # → [B, 64, 256, 256]
        x = self.maxpool(x)     # → [B, 64, 128, 128]
        x = self.layer0(x)      # → reste identique
        x = self.layer1(x)      # → stride=2 → ↓ taille
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)     # → [B, 512, 1, 1]
        x = torch.flatten(x, 1) # → [B, 512]

        if len(max_doppler.shape) == 1:
            max_doppler = max_doppler.unsqueeze(1)  # [B, 1]

        x = torch.cat([x, max_doppler], dim=1)  # [B, 513]
        
        x = self.fc(x)          # → [B, 1]
        return x

""""
#### REG_5 ####
class DopplerNetRegressionTemporal(nn.Module):
    def __init__(self, param):
        super(DopplerNetRegressionTemporal, self).__init__()
        H = 512
        W = 512
        self.nb_channels = param["MODEL"]["NB_CHANNELS"]
        self.nb_previous_frames = param["MODEL"]["NB_PREV_FRAMES"]
        self.total_input_channels = self.nb_previous_frames + 1

        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels=self.total_input_channels, out_channels=self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.nb_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv2d(in_channels=self.nb_channels, out_channels=2 * self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * self.nb_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv2d(in_channels=2 * self.nb_channels, out_channels=4 * self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * self.nb_channels)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers (simplifiées)
        self.flatten_dim = (4 * self.nb_channels) * (H // 32) * (W // 32)
        self.fc1 = nn.Linear(self.flatten_dim + self.total_input_channels, 128)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, max_doppler):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, start_dim=1)  # (batch, flatten_dim)
        x = torch.cat((x, max_doppler), dim=1)  # (batch, flatten_dim + total_input_channels)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        output = self.fc2(x)

        return output
"""


#### REG_4 ####
"""
class DopplerNetRegressionTemporal(nn.Module):
    def __init__(self, param):
        super(DopplerNetRegressionTemporal, self).__init__()
        H = 512
        W = 512
        self.nb_channels = param["MODEL"]["NB_CHANNELS"]
        self.nb_previous_frames = param["MODEL"]["NB_PREV_FRAMES"]  # ← param YAML

        self.total_input_channels = self.nb_previous_frames + 1  # Séquence complète

        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels=self.total_input_channels, out_channels=self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.nb_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv2d(in_channels=self.nb_channels, out_channels=2 * self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * self.nb_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv2d(in_channels=2 * self.nb_channels, out_channels=4 * self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * self.nb_channels)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.flatten_dim = (4 * self.nb_channels) * (H // 32) * (W // 32)
        self.fc1 = nn.Linear(self.flatten_dim + self.total_input_channels, 256)  # ⚠ ici aussi on utilise total_input_channels
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, max_doppler):
        # x: (batch, total_input_channels, H, W)
        # max_doppler: (batch, total_input_channels)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, start_dim=1)  # shape: (batch, flatten_dim)

        # ⚠ ici : max_doppler est déjà (batch, total_input_channels), donc PAS de view()
        x = torch.cat((x, max_doppler), dim=1)  # shape: (batch, flatten_dim + total_input_channels)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        output = self.fc4(x)

        return output
"""

#### REG_3 ####
class DopplerNetRegression(nn.Module):
    def __init__(self, param):
        super(DopplerNetRegression, self).__init__()
        H = 512
        W = 512
        self.nb_channels = param["MODEL"]["NB_CHANNELS"]

        # Feature extraction
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.nb_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)  # 512 → 128

        self.conv2 = nn.Conv2d(in_channels=self.nb_channels, out_channels=2*self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(2*self.nb_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)  # 128 → 32

        self.conv3 = nn.Conv2d(in_channels=2*self.nb_channels, out_channels=4*self.nb_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4*self.nb_channels)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32 → 16

        # Fully connected layers
        self.flatten_dim = (4 * self.nb_channels) * (H // 32) * (W // 32)

        self.fc1 = nn.Linear(self.flatten_dim + 1, 256)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x, max_doppler):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, start_dim=1)
        max_doppler = max_doppler.view(-1, 1)
        x = torch.cat((x, max_doppler), dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        output = self.fc4(x)

        return output