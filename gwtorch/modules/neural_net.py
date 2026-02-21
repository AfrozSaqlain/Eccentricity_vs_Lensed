import torch
import torch.nn as nn
import torch.nn.functional as F
from linformer import Linformer
from vit_pytorch import ViT as VanillaViT
from vit_pytorch.efficient import ViT as EfficientViT
from vit_pytorch.cvt import CvT as CvTModel

class CNN_Model(nn.Module):
    def __init__(self, input_channels=3, dropout=0.2):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
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


class CNNBinaryModel(nn.Module):
    def __init__(self, input_channels=3, dropout=0.2):
        super(CNNBinaryModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 2)
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


class SimpleCNNBinary(nn.Module):
    def __init__(self, input_channels=3, dropout=0.2):
        super(SimpleCNNBinary, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_vit(num_channels, num_classes=3, dropout=0.2):
    return VanillaViT(
        image_size=224,
        patch_size=32,
        num_classes=num_classes,
        dim=128,
        depth=12,
        heads=8,
        mlp_dim=256,
        dropout=dropout,
        emb_dropout=dropout,
        channels=num_channels,
    )


def create_vit_linear(num_channels, num_classes=3, dropout=0.2):
    transformer = Linformer(
        dim=128,
        seq_len=50,
        depth=12,
        heads=8,
        k=64,
        dropout=dropout,
    )
    return EfficientViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=num_classes,
        transformer=transformer,
        channels=num_channels,
    )


def create_cvt(num_channels, num_classes=3, dropout=0.2):
    return CvTModel(
        num_classes=num_classes,
        channels=num_channels,
        s1_emb_dim=64,
        s1_emb_kernel=7,
        s1_emb_stride=4,
        s1_proj_kernel=3,
        s1_kv_proj_stride=2,
        s1_heads=1,
        s1_depth=1,
        s1_mlp_mult=4,
        s2_emb_dim=192,
        s2_emb_kernel=3,
        s2_emb_stride=2,
        s2_proj_kernel=3,
        s2_kv_proj_stride=2,
        s2_heads=3,
        s2_depth=2,
        s2_mlp_mult=4,
        s3_emb_dim=384,
        s3_emb_kernel=3,
        s3_emb_stride=2,
        s3_proj_kernel=3,
        s3_kv_proj_stride=2,
        s3_heads=4,
        s3_depth=10,
        s3_mlp_mult=4,
        dropout=dropout,
    )


def create_transformer_model(name, num_channels, device, num_classes=3, dropout=0.2):
    if name == "ViT":
        model = create_vit(num_channels=num_channels, num_classes=num_classes, dropout=dropout)
    elif name == "ViTLinear":
        model = create_vit_linear(num_channels=num_channels, num_classes=num_classes, dropout=dropout)
    elif name == "CvT":
        model = create_cvt(num_channels=num_channels, num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown transformer model '{name}'. Expected one of: ViT, ViTLinear, CvT.")
    return model.to(device)
