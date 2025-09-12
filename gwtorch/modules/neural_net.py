import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    def __init__(self, input_channlels=3, dropout=0.2):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(input_channlels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, dropout=0.2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)  # <- need to calculate this
        self.fc2 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)   # 224 -> 112
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)   # 112 -> 56
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNN_Meta_Model(nn.Module):
    def __init__(self, input_features=3, hidden_features=32, output_features=3):
        super(CNN_Model, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features

        self.conv1 = nn.Conv2d(self.input_features, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Placeholder for dynamic FC layer
        self.fc1 = None
        self.fc2 = nn.Linear(256, self.output_features)  # output stays fixed

    def forward(self, x):
        x = F.relu(self.conv1(x))     # -> (B, 32, H, W)
        x = self.pool(x)              # -> (B, 32, H/2, W/2)

        x = F.relu(self.conv2(x))     # -> (B, 64, H/2, W/2)
        x = self.pool(x)              # -> (B, 64, H/4, W/4)

        x = F.relu(self.conv3(x))     # -> (B, 128, H/4, W/4)
        x = self.pool(x)              # -> (B, 128, H/8, W/8)

        x = x.view(x.size(0), -1)     # flatten

        # Dynamically create fc1 based on input feature map
        if self.fc1 is None:
            in_features = x.size(1)
            self.fc1 = nn.Linear(in_features, 256).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x