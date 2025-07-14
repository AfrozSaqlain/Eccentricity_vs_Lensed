import os

# os.makedirs('../models', exist_ok = True)
# os.makedirs('../results', exist_ok=True)
# os.makedirs('../results/cnn_results', exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import glob
from pathlib import Path
from PIL import Image

from modules.general_utils import compute_roc_auc_with_misclassifications, plot_roc_curves, plot_confusion_matrix, save_misclassified_files_to_dict

RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 3e-4
GAMMA = 0.7

results_dir = Path('../results/cnn_results')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_dir = '../data_2/data/test'

test_list = glob.glob(os.path.join(test_dir, '*.png'))

print(f"Test Data: {len(test_list)}")

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

class GWDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        
        img_transformed = self.transform(img)
        img_transformed = img_transformed[:3, :, :]

        label_str = img_path.split("/")[-1].split("_")[0]
        if label_str == "eccentric":
            label = 1
        elif label_str == "unlensed":
            label = 2
        else:
            label = 0 

        # Extract file number for tracking misclassifications
        file_number = img_path.split("/")[-1].split("_")[1]

        return img_transformed, label, file_number, img_path
    
test_data = GWDataset(test_list, transform=test_transforms)

# Custom collate function to handle the additional file_number and img_path
def custom_collate_fn(batch):
    images, labels, file_numbers, img_paths = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels, file_numbers, img_paths

test_loader = DataLoader(dataset = test_data, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model0 = CNN_Model().to(device)

model_path = '../models/cnn_model0.pth'

if os.path.exists(model_path):
    model0.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
else:
    print(f"No pre-trained model found at {model_path}. Please specify a trained model.")


class_names=["Lensed", "Eccentric", "Unlensed"]

fpr, tpr, roc_auc, labels, predictions, _, misclassified = compute_roc_auc_with_misclassifications(model=model0, data_loader=test_loader, device=device)

roc_fig = plot_roc_curves(fpr=fpr, tpr=tpr, roc_auc=roc_auc, class_names=class_names, title_suffix=" (Test Set)", results_dir=results_dir)

plot_confusion_matrix(y_true=labels, y_pred=predictions, class_names=class_names, title_suffix=" (Test Set)", results_dir=results_dir)

save_misclassified_files_to_dict(misclassified, results_dir / "misclassified_test.pkl")