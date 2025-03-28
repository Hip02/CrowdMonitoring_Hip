import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

##############################################################################
#                                                                            #
# CLASS DESCRIBING AN IMPROVED CNN ARCHITECTURE FOR DOPPLER MAPS (512x512)   #
#                                                                            #
##############################################################################


class DebugResNet(nn.Module):
    def __init__(self, param):

        self.pretrained = param["MODEL"].get("PRETRAINED", False)
        self.in_channels = param["MODEL"].get("NB_CHANNELS", 1)

        super(DebugResNet, self).__init__()

        self.model = torchvision.models.resnet18(pretrained=self.pretrained)

        self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation=1):
        super(ResidualBlock, self).__init__()
        padding = dilation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class DopplerResNetRegression(nn.Module):
    def __init__(self, param, layers, block=ResidualBlock):
        super(DopplerResNetRegression, self).__init__()
        nb_input_frames = param["MODEL"].get("NB_PREV_FRAMES", 0) + 1
        if param["DATASET"].get("ACTIVE_ANTENNA2", False):
            nb_input_frames += 1
        
        number_of_max = nb_input_frames

        if param["DATASET"].get("ACTIVE_PHASE", False):
            nb_input_frames += 1
        
        if param["DATASET"].get("ACTIVE_ANTENNA2", False) and param["DATASET"].get("ACTIVE_PHASE", False):
            nb_input_frames += 1

        out_channels_conv1 = param["MODEL"]["NB_CHANNELS"]
        self.use_atrous_conv1 = param["MODEL"].get("USE_ATROUS_CONV1", False)
        self.atrous_dilation_conv1 = param["MODEL"].get("ATROUS_DILATION_CONV1", 2)
        self.use_dilation = param["MODEL"].get("USE_DILATION", False)
        self.dilation = 2 if self.use_dilation else 1

        # Conv1: standard + atrous (optionnel)
        if self.use_atrous_conv1:
            self.conv1_std = nn.Sequential(
                nn.Conv2d(nb_input_frames, out_channels_conv1, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(out_channels_conv1),
                nn.ReLU(inplace=True)
            )
            self.conv1_dilated = nn.Sequential(
                nn.Conv2d(nb_input_frames, out_channels_conv1, kernel_size=3, stride=2,
                          padding=self.atrous_dilation_conv1, dilation=self.atrous_dilation_conv1),
                nn.BatchNorm2d(out_channels_conv1),
                nn.ReLU(inplace=True)
            )
            conv1_out_channels = 2 * out_channels_conv1
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(nb_input_frames, out_channels_conv1, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(out_channels_conv1),
                nn.ReLU(inplace=True)
            )
            conv1_out_channels = out_channels_conv1

        self.inplanes = conv1_out_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers (avec dilation uniforme si activée)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1, dilation=self.dilation)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2, dilation=self.dilation)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2, dilation=self.dilation)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2, dilation=self.dilation)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 + number_of_max, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        first_dilation = dilation if stride == 1 else 1

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )

        layers = [block(self.inplanes, planes, stride, downsample, dilation=first_dilation)]
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, max_doppler):
        if self.use_atrous_conv1:
            x_std = self.conv1_std(x)
            x_dil = self.conv1_dilated(x)
            x = torch.cat([x_std, x_dil], dim=1)
        else:
            x = self.conv1(x)

        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if len(max_doppler.shape) == 1:
            max_doppler = max_doppler.unsqueeze(1)
        x = torch.cat([x, max_doppler], dim=1)
        return self.fc(x)
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        width = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, stride=stride,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class DopplerResNet50Regression(nn.Module):
    def __init__(self, param, layers=[3,4,6,3], block=Bottleneck):
        super(DopplerResNet50Regression, self).__init__()
        nb_input_frames = param["MODEL"].get("NB_PREV_FRAMES", 0) + 1
        out_channels_conv1 = param["MODEL"]["NB_CHANNELS"]
        self.use_atrous_conv1 = param["MODEL"].get("USE_ATROUS_CONV1", False)
        self.atrous_dilation_conv1 = param["MODEL"].get("ATROUS_DILATION_CONV1", 2)
        self.use_dilation = param["MODEL"].get("USE_DILATION", False)
        self.dilation = 2 if self.use_dilation else 1

        # Conv1 standard + atrous
        if self.use_atrous_conv1:
            self.conv1_std = nn.Sequential(
                nn.Conv2d(nb_input_frames, out_channels_conv1, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(out_channels_conv1),
                nn.ReLU(inplace=True)
            )
            self.conv1_dilated = nn.Sequential(
                nn.Conv2d(nb_input_frames, out_channels_conv1, kernel_size=3, stride=2,
                          padding=self.atrous_dilation_conv1, dilation=self.atrous_dilation_conv1, bias=False),
                nn.BatchNorm2d(out_channels_conv1),
                nn.ReLU(inplace=True)
            )
            conv1_out_channels = 2 * out_channels_conv1
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(nb_input_frames, out_channels_conv1, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(out_channels_conv1),
                nn.ReLU(inplace=True)
            )
            conv1_out_channels = out_channels_conv1

        self.inplanes = conv1_out_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 layers
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1, dilation=self.dilation)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2, dilation=self.dilation)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2, dilation=self.dilation)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2, dilation=self.dilation)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion + nb_input_frames, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        first_dilation = dilation if stride == 1 else 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, downsample, dilation=first_dilation)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, max_doppler):
        if self.use_atrous_conv1:
            x_std = self.conv1_std(x)
            x_dil = self.conv1_dilated(x)
            x = torch.cat([x_std, x_dil], dim=1)
        else:
            x = self.conv1(x)

        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if len(max_doppler.shape) == 1:
            max_doppler = max_doppler.unsqueeze(1)
        x = torch.cat([x, max_doppler], dim=1)
        return self.fc(x)

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