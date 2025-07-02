from __future__ import print_function

import glob
import os
import sys
import random

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
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
import torch.nn.functional as F

from vit_pytorch.efficient import ViT
from vit_pytorch.cvt import CvT

model_selection = str(sys.argv[1])

# sys.stdout = open("log.out", "w")
# sys.stderr = open("error.err", "w")

batch_size = 512
epochs = 20
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

train_dir = './data_tmp/train'
test_dir = './data_tmp/test'

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

        return img_transformed, label

train_data = GWDataset(train_list, transform=train_transforms)
valid_data = GWDataset(valid_list, transform=test_transforms)
test_data = GWDataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

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

file_path = "./model.pth"

if os.path.exists(file_path):
    print("Pre-trained model found. Training continued...")
    model_path = 'model.pth'
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
    
    for data, label in train_loader:
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
        for data, label in valid_loader:
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
torch.save(model.state_dict(), 'model.pth')
print("Model saved successfully!")

# ==============================================================================
# ROC CURVE AND AUC COMPUTATION
# ==============================================================================

def compute_roc_auc(model, data_loader, device, class_names=['Class 0', 'Eccentric', 'Unlensed']):
    """
    Compute ROC curves and AUC scores for multi-class classification
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Convert labels to binary format for ROC computation
    label_binarizer = LabelBinarizer()
    y_true_binary = label_binarizer.fit_transform(all_labels)
    
    # Handle case where we have only 2 classes (binary classification)
    if y_true_binary.shape[1] == 1:
        y_true_binary = np.hstack([1 - y_true_binary, y_true_binary])
    
    n_classes = y_true_binary.shape[1]
    
    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], all_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc, all_labels, all_predictions, class_names

def plot_roc_curves(fpr, tpr, roc_auc, class_names, title_suffix=""):
    """
    Plot ROC curves for multi-class classification
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        if i in fpr:
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                    label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    # Plot macro-average ROC curve
    plt.plot(fpr["macro"], tpr["macro"], color='black', linestyle='--', lw=2,
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - Multi-class Classification{title_suffix}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix=""):
    """
    Plot confusion matrix with counts and percentages
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations with both count and percentage
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = cm_percentage[i, j]
            row.append(f"{count}\n({percentage:.1f}%)")
        annotations.append(row)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix{title_suffix}', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    return plt.gcf()

# Compute ROC curves and AUC for validation set
print("\n" + "="*50)
print("COMPUTING ROC CURVES AND AUC SCORES")
print("="*50)

class_names = ['Class 0', 'Eccentric', 'Unlensed']

# Validation set ROC
print("Computing ROC curves for validation set...")
val_fpr, val_tpr, val_roc_auc, val_labels, val_predictions, _ = compute_roc_auc(
    model, valid_loader, device, class_names
)

# Test set ROC (if you want to evaluate on test set)
print("Computing ROC curves for test set...")
test_fpr, test_tpr, test_roc_auc, test_labels, test_predictions, _ = compute_roc_auc(
    model, test_loader, device, class_names
)

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
val_fig = plot_roc_curves(val_fpr, val_tpr, val_roc_auc, class_names, " (Validation Set)")
plt.savefig('roc_curves_validation.png', dpi=300, bbox_inches='tight')
plt.show()

# Test ROC curves
test_fig = plot_roc_curves(test_fpr, test_tpr, test_roc_auc, class_names, " (Test Set)")
plt.savefig('roc_curves_test.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot confusion matrices
print("Generating confusion matrices...")

# Validation confusion matrix
val_cm_fig = plot_confusion_matrix(val_labels, val_predictions, class_names, " (Validation Set)")
plt.savefig('confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
plt.show()

# Test confusion matrix
test_cm_fig = plot_confusion_matrix(test_labels, test_predictions, class_names, " (Test Set)")
plt.savefig('confusion_matrix_test.png', dpi=300, bbox_inches='tight')
plt.show()

# Print classification reports
print("\nClassification Reports:")
print("-" * 40)
print("Validation Set:")
print(classification_report(val_labels, val_predictions, target_names=class_names))

print("\nTest Set:")
print(classification_report(test_labels, test_predictions, target_names=class_names))

# Plot training curves
print("Generating training curves...")
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
epochs_range = range(1, len(train_losses) + 1)
plt.plot(epochs_range, train_losses, 'b-', label='Train Loss')
plt.plot(epochs_range, val_losses, 'r-', label='Val Loss')
plt.fill_between(epochs_range, train_losses, alpha=0.3, color='blue')
plt.fill_between(epochs_range, val_losses, alpha=0.3, color='red')
plt.title('Loss Progression')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("Files saved:")
print("- roc_curves_validation.png")
print("- roc_curves_test.png") 
print("- confusion_matrix_validation.png")
print("- confusion_matrix_test.png")
print("- training_curves.png")
print("- model.pth")