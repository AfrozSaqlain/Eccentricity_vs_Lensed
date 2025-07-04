from __future__ import print_function

import glob
import os
import sys
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report

from vit_pytorch.efficient import ViT
from vit_pytorch.cvt import CvT

from modules.general_utils import compute_roc_auc_with_misclassifications, plot_roc_curves, plot_confusion_matrix, print_misclassified_files, save_misclassified_files_to_txt, plot_training_curves

model_selection = str(sys.argv[1])

# sys.stdout = open("log.out", "w")
# sys.stderr = open("error.err", "w")

batch_size = 512
epochs = 100
lr = 3e-5
gamma = 0.7
seed = 42

print("Code is running")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"The device is {device}")

os.makedirs('../results', exist_ok=True)
os.makedirs('../models', exist_ok=True)

train_dir = '../data/train'
test_dir = '../data/test'

results_dir = Path('../results')

train_list = glob.glob(os.path.join(train_dir,'*.png'))
test_list = glob.glob(os.path.join(test_dir, '*.png'))

print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

labels = [path.split('/')[-1].split('_')[0] for path in train_list]

train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          shuffle=True,
                                          random_state=seed)

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
valid_data = GWDataset(valid_list, transform=val_transforms)
test_data = GWDataset(test_list, transform=test_transforms)

# Custom collate function to handle the additional file_number and img_path
def custom_collate_fn(batch):
    images, labels, file_numbers, img_paths = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels, file_numbers, img_paths

train_loader = DataLoader(dataset = train_data, num_workers=os.cpu_count(), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
valid_loader = DataLoader(dataset = valid_data, num_workers=os.cpu_count(), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(dataset = test_data, num_workers=os.cpu_count(), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

print(f"Train Dataset Length: {len(train_data)}, Test Dataloader Length: {len(train_loader)}")
print(f"Validation Dataset Length: {len(valid_data)}, Validation Dataloader Length: {len(valid_loader)}")

if model_selection == 'CvT':
    model =  CvT(num_classes = 3,
                s1_emb_dim = 64,
                s1_emb_kernel = 7,
                s1_emb_stride = 4,
                s1_proj_kernel = 3,
                s1_kv_proj_stride = 2,
                s1_heads = 1,
                s1_depth = 1,
                s1_mlp_mult = 4,
                s2_emb_dim = 192,
                s2_emb_kernel = 3,
                s2_emb_stride = 2,
                s2_proj_kernel = 3,
                s2_kv_proj_stride = 2,
                s2_heads = 3,
                s2_depth = 2,
                s2_mlp_mult = 4,
                s3_emb_dim = 384,
                s3_emb_kernel = 3,
                s3_emb_stride = 2,
                s3_proj_kernel = 3,
                s3_kv_proj_stride = 2,
                s3_heads = 4,
                s3_depth = 10,
                s3_mlp_mult = 4,
                dropout = 0.3).to(device)
else:
    efficient_transformer = Linformer(
        dim=128,
        seq_len=49+1,
        depth=12,
        heads=8,
        k=64,
        dropout=0.2,)

    model = ViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=3,
        transformer=efficient_transformer,
        channels=3,
    ).to(device)

model_path = "../models/model_alpha.pth"

if os.path.exists(model_path):
    print("Pre-trained model found. Training continued...")
    model.load_state_dict(torch.load(model_path))
else:
    print("Pre-trained model not found. Training from beginning...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    
    for data, label, file_numbers, img_paths in train_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    
    train_losses.append(epoch_loss.item())
    train_accuracies.append(epoch_accuracy.item())
    
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label, file_numbers, img_paths in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)
    
    val_losses.append(epoch_val_loss.item())
    val_accuracies.append(epoch_val_accuracy.item())
    
    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")

# Save the trained model
torch.save(model.state_dict(), '../models/model_alpha.pth')
print("Model saved successfully!")

# ==============================================================================
# ROC CURVE AND AUC COMPUTATION WITH MISCLASSIFICATION TRACKING
# ==============================================================================

# Compute ROC curves and AUC for validation set
print("\n" + "="*50)
print("COMPUTING ROC CURVES AND AUC SCORES WITH MISCLASSIFICATION ANALYSIS")
print("="*50)

class_names = ['Lensed', 'Eccentric', 'Unlensed']

# Validation set ROC with misclassification tracking
print("Computing ROC curves for validation set...")
val_fpr, val_tpr, val_roc_auc, val_labels, val_predictions, _, val_misclassified = compute_roc_auc_with_misclassifications(
    model, valid_loader, device, class_names
)

# Test set ROC with misclassification tracking
print("Computing ROC curves for test set...")
test_fpr, test_tpr, test_roc_auc, test_labels, test_predictions, _, test_misclassified = compute_roc_auc_with_misclassifications(
    model, test_loader, device, class_names
)

# Print misclassified files
print_misclassified_files(val_misclassified, "Validation Set")
print_misclassified_files(test_misclassified, "Test Set")

# Save misclassified files to text files
save_misclassified_files_to_txt(val_misclassified, "Validation Set", results_dir / "misclassified_validation.txt")
save_misclassified_files_to_txt(test_misclassified, "Test Set", results_dir / "misclassified_test.txt")

# Print AUC scores
print("\nAUC Scores:")
print("-" * 30)
print("Validation Set:")
for i, class_name in enumerate(class_names):
    if i in val_roc_auc:
        print(f"  {class_name}: {val_roc_auc[i]:.4f}")
print(f"  Macro-average: {val_roc_auc['macro']:.4f}")

print("\nTest Set:")
for i, class_name in enumerate(class_names):
    if i in test_roc_auc:
        print(f"  {class_name}: {test_roc_auc[i]:.4f}")
print(f"  Macro-average: {test_roc_auc['macro']:.4f}")

# Plot ROC curves
print("\nGenerating ROC curve plots...")

# Validation ROC curves
val_fig = plot_roc_curves(val_fpr, val_tpr, val_roc_auc, class_names, " (Validation Set)", results_dir=results_dir)

# Test ROC curves
test_fig = plot_roc_curves(test_fpr, test_tpr, test_roc_auc, class_names, " (Test Set)", results_dir=results_dir)

# Plot confusion matrices
print("Generating confusion matrices...")

# Validation confusion matrix
val_cm_fig = plot_confusion_matrix(val_labels, val_predictions, class_names, " (Validation Set)", results_dir=results_dir)

# Test confusion matrix
test_cm_fig = plot_confusion_matrix(test_labels, test_predictions, class_names, " (Test Set)", results_dir=results_dir)

# Print classification reports
print("\nClassification Reports:")
print("-" * 40)
print("Validation Set:")
print(classification_report(val_labels, val_predictions, target_names=class_names))

print("\nTest Set:")
print(classification_report(test_labels, test_predictions, target_names=class_names))

# Plot training curves
plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, results_dir)

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("Files saved:")
print("- ROC_Curves_Validation.png")
print("- ROC_Curves_Test.png") 
print("- Confusion_Matrix_Validation.png")
print("- Confusion_Matrix_Test.png")
print("- Training_curves.png")
print("- model.pth")
print("- misclassified_validation.txt")
print("- misclassified_test.txt")
print(f"\nMisclassification Summary:")
print(f"Validation Set: {len(val_misclassified)} misclassified files")
print(f"Test Set: {len(test_misclassified)} misclassified files")