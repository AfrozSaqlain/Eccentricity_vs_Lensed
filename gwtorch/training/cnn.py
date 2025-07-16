import os
import sys

os.makedirs('../models', exist_ok = True)
os.makedirs('../results', exist_ok=True)
os.makedirs('../results/cnn_results', exist_ok=True)

# sys.stdout = open("../results/cnn_results/log.out", "w")
# sys.stderr = open("../results/cnn_results/error.err", "w")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from PIL import Image
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from gwtorch.modules.general_utils import compute_roc_auc_with_misclassifications, plot_roc_curves, plot_confusion_matrix, plot_training_curves
from gwtorch.modules.neural_net import CNN_Model

RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 3e-4
GAMMA = 0.7

results_dir = Path('../results/cnn_results')
os.makedirs(results_dir / 'Plots', exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dir = '../data/train'
test_dir = '../data/test'

train_list = glob.glob(os.path.join(train_dir,'*.png'))
test_list = glob.glob(os.path.join(test_dir, '*.png'))

print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

labels = [path.split('/')[-1].split('_')[0] for path in train_list]

train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          shuffle=True,
                                          random_state=RANDOM_SEED)

print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

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
    
train_data = GWDataset(train_list, transform=train_transforms)
valid_data = GWDataset(valid_list, transform=test_transforms)
test_data = GWDataset(test_list, transform=test_transforms)

# Custom collate function to handle the additional file_number and img_path
def custom_collate_fn(batch):
    images, labels, file_numbers, img_paths = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels, file_numbers, img_paths

train_loader = DataLoader(dataset = train_data, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
valid_loader = DataLoader(dataset = valid_data, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(dataset = test_data, num_workers=os.cpu_count(), batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

print(f"Train Dataset Length: {len(train_data)}, Train Dataloader Length: {len(train_loader)}")

print(f"Validation Dataset Length: {len(valid_data)}, Validation Dataloader Length: {len(valid_loader)}")

model0 = CNN_Model().to(device)

model_path = '../models/cnn_model0.pth'

if os.path.exists(model_path):
    model0.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
else:
    print(f"No pre-trained model found at {model_path}, starting training from scratch.")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model0.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=GAMMA)


def train_model(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, epochs=EPOCHS):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, file_numbers, img_paths in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_correct = 0
            valid_total = 0
            
            for val_images, val_labels, file_numbers, img_paths in valid_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                v_loss = loss_fn(val_outputs, val_labels)
                
                valid_loss += v_loss.item() * val_images.size(0)
                _, v_predicted = torch.max(val_outputs.data, 1)
                valid_total += val_labels.size(0)
                valid_correct += (v_predicted == val_labels).sum().item()
            
            epoch_valid_loss = valid_loss / len(valid_loader.dataset)
            epoch_valid_acc = valid_correct / valid_total
            val_losses.append(epoch_valid_loss)
            val_accuracies.append(epoch_valid_acc)

            print(f"Validation Loss: {epoch_valid_loss:.4f}, Validation Accuracy: {epoch_valid_acc:.4f}")
        
        scheduler.step()
    
    return train_losses, val_losses, train_accuracies, val_accuracies


train_losses, val_losses, train_accuracies, val_accuracies = train_model(model0, train_loader, valid_loader, loss_fn, optimizer, scheduler, epochs=EPOCHS)

class_names=["Lensed", "Eccentric", "Unlensed"]


fpr, tpr, roc_auc, labels, predictions, _, misclassified = compute_roc_auc_with_misclassifications(model=model0, data_loader=test_loader, device=device)

roc_fig = plot_roc_curves(fpr=fpr, tpr=tpr, roc_auc=roc_auc, class_names=class_names, title_suffix=" (Test Set)", results_dir=results_dir)

plot_confusion_matrix(y_true=labels, y_pred=predictions, class_names=class_names, title_suffix=" (Test Set)", results_dir=results_dir)

plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, results_dir)

torch.save(model0.state_dict(), model_path)

# sys.stdout.close()
# sys.stderr.close()