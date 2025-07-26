import os
import sys
import glob
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler

from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from gwtorch.modules.general_utils import (
    compute_roc_auc_with_misclassifications,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_training_curves,
    save_misclassified_files_to_dict
)
from gwtorch.modules.neural_net import CNN_Model

# ----------------- Constants -----------------
RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCHS = 20
LR = 3e-4
GAMMA = 0.7
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_names = ["Eccentric", "Lensed"]  # Binary classification

# ----------------- Dataset Class -----------------
class GWDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")  # Ensures 3 channels
        img = self.transform(img)
        label_str = img_path.split("/")[-1].split("_")[0]
        # Binary classification: eccentric=0, lensed=1
        label = {"eccentric": 0, "lensed": 1}[label_str]
        file_number = img_path.split("/")[-1].split("_")[1]
        return img, label, file_number, img_path

# ----------------- Utility Functions -----------------
def custom_collate_fn(batch):
    images, labels, file_numbers, img_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), file_numbers, img_paths

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN model for binary GW classification (eccentric vs lensed)")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--gamma', type=float, default=GAMMA)
    parser.add_argument('--model-path', type=str, default='./models/cnn_model_binary.pth')
    return parser.parse_args()

def setup_data_loaders(train_dir, test_dir, batch_size):
    # Load only eccentric and lensed images
    all_train = glob.glob(os.path.join(train_dir, '*.png'))
    all_test = glob.glob(os.path.join(test_dir, '*.png'))
    
    train_list = []
    test_list = []
    
    # Filter for only eccentric and lensed images
    for path in all_train:
        label = path.split('/')[-1].split('_')[0]
        if label in ['eccentric', 'lensed']:
            train_list.append(path)
    
    for path in all_test:
        label = path.split('/')[-1].split('_')[0]
        if label in ['eccentric', 'lensed']:
            test_list.append(path)

    labels = [path.split('/')[-1].split('_')[0] for path in train_list]
    train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, shuffle=True, random_state=RANDOM_SEED)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    train_data = GWDataset(train_list, transform)
    valid_data = GWDataset(valid_list, transform)
    test_data = GWDataset(test_list, transform)

    # Compute class weights for training data (binary classification)
    label_to_idx = {"eccentric": 0, "lensed": 1}
    train_labels = [path.split('/')[-1].split('_')[0] for path in train_list]
    label_counts = Counter(train_labels)
    class_counts = [label_counts['eccentric'], label_counts['lensed']]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()

    print(f"Training samples - Eccentric: {label_counts['eccentric']}, Lensed: {label_counts['lensed']}")
    print(f"Total training samples: {len(train_list)}")
    print(f"Validation samples: {len(valid_list)}")
    print(f"Test samples: {len(test_list)}")

    # Assign sample weights for sampler
    sample_weights = [class_weights[label_to_idx[label]] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    # Use regular shuffling or no shuffling for val/test
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return train_loader, valid_loader, test_loader, class_weights

def train_model(model, train_loader, valid_loader, loss_fn, optimizer, scheduler, epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / total)
        train_accuracies.append(correct / total)
        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Accuracy = {train_accuracies[-1]:.4f}")

        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels, _, _ in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_losses.append(val_loss / total)
        val_accuracies.append(correct / total)
        print(f"          Validation Loss = {val_losses[-1]:.4f}, Accuracy = {val_accuracies[-1]:.4f}")
        scheduler.step()

    return train_losses, val_losses, train_accuracies, val_accuracies

# ----------------- Main -----------------
def main():
    args = parse_args()

    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results/cnn_results/Plots', exist_ok=True)
    results_dir = Path('./results/cnn_results')

    print(f"Using device: {device}")
    print("Training binary CNN classifier: Eccentric vs Lensed")

    train_loader, valid_loader, test_loader, class_weights = setup_data_loaders('./data/train', './data/test', args.batch_size)

    # Note: You may need to modify CNN_Model to have 2 output classes instead of 3
    # If CNN_Model doesn't support changing num_classes, you might need to modify it
    model = CNN_Model().to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Binary model loaded from {args.model_path}")
    else:
        print(f"No pre-trained binary model found. Training from scratch.")

    weights = class_weights.detach().clone().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.gamma)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, valid_loader, loss_fn, optimizer, scheduler, args.epochs
    )

    print("Evaluating on validation and test set...")

    val_fpr, val_tpr, val_roc_auc, val_labels, val_preds, _, val_misclassified = compute_roc_auc_with_misclassifications(model, valid_loader, device, class_names)
    test_fpr, test_tpr, test_roc_auc, test_labels, test_preds, _, test_misclassified = compute_roc_auc_with_misclassifications(model, test_loader, device, class_names)

    plot_roc_curves(val_fpr, val_tpr, val_roc_auc, class_names, " (Validation Set - Binary)", results_dir)
    plot_confusion_matrix(val_labels, val_preds, class_names, " (Validation Set - Binary)", results_dir)
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, results_dir)
    save_misclassified_files_to_dict(val_misclassified, results_dir / "misclassified_val_binary.pkl")

    plot_roc_curves(test_fpr, test_tpr, test_roc_auc, class_names, " (Test Set - Binary)", results_dir)
    plot_confusion_matrix(test_labels, test_preds, class_names, " (Test Set - Binary)", results_dir)
    save_misclassified_files_to_dict(test_misclassified, results_dir / "misclassified_test_binary.pkl")

    print("Validation Classification Report (Binary):")
    val_report = classification_report(val_labels, val_preds, target_names=class_names)
    print(val_report)
    val_report_path = results_dir / "val_classification_report_binary.txt"
    with open(val_report_path, "w") as f:
        f.write("Validation Classification Report (Binary - Eccentric vs Lensed):\n")
        f.write(val_report)

    print("\nTest Classification Report (Binary):")
    test_report = classification_report(test_labels, test_preds, target_names=class_names)
    print(test_report)
    test_report_path = results_dir / "test_classification_report_binary.txt"
    with open(test_report_path, "w") as f:
        f.write("Test Classification Report (Binary - Eccentric vs Lensed):\n")
        f.write(test_report)

    print(f"\nBinary CNN training complete. Summary:")
    print(f"Validation misclassifications: {len(val_misclassified)}")
    print(f"Test misclassifications: {len(test_misclassified)}")

    torch.save(model.state_dict(), args.model_path)
    print(f"Binary CNN model saved to {args.model_path}")

if __name__ == "__main__":
    main()