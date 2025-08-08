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

import random
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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ----------------- Constants -----------------
RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCHS = 20
LR = 3e-4
GAMMA = 0.7
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_names = ["Lensed", "Eccentric", "Quasi-Circular Unlensed"]

# ----------------- Dataset Class -----------------
class GWDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # Load .npy file
        arr = np.load(file_path)  # shape: (H, W, C)

        # Transpose to (H, W, C) only if needed
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]  # Convert to (H, W, 1)

        arr = arr.astype(np.float32)

        # Apply torchvision transforms
        
        arr = self.transform(arr)  # Will convert to tensor inside transforms if needed

        # Extract label and metadata
        label_str = file_path.split("/")[-1].split("_")[0]
        label = {'lensed': 0, 'eccentric': 1, 'unlensed': 2}[label_str]
        file_number = file_path.split("/")[-1].split("_")[1].split(".")[0]

        return arr, label, file_number, file_path

# ----------------- Utility Functions -----------------
def custom_collate_fn(batch):
    images, labels, file_numbers, img_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), file_numbers, img_paths

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN model for GW classification")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--gamma', type=float, default=GAMMA)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-path', type=str, default='./models/cnn_model0.pth')
    parser.add_argument('--num-channels', type=int, default=3, help="Defines what is the input channels of the model. Default is 3 for signals from 3 detectors.")
    return parser.parse_args()

def setup_data_loaders(train_dir, test_dir, batch_size, seed=42):
    # all_files = glob.glob(os.path.join(train_dir, '*.npy')) + glob.glob(os.path.join(test_dir, '*.npy'))
    all_files = glob.glob('./data/train/*.npy') + glob.glob('./data/test/*.npy')
    all_labels = [path.split('/')[-1].split('_')[0] for path in all_files]

    # Stratified split: 70% train, 15% val, 15% test
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.3, stratify=all_labels, random_state=seed
    )

    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    # Define transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x).permute(2, 0, 1)),
        transforms.Resize((224, 224)),
    ])
    
    train_data = GWDataset(train_files, transform)
    val_data   = GWDataset(val_files, transform)
    test_data  = GWDataset(test_files, transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader

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

    seed_everything(args.seed)

    train_loader, valid_loader, test_loader = setup_data_loaders('./data/train', './data/test', args.batch_size)

    model = CNN_Model(input_channlels=args.num_channels).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
    else:
        print(f"No pre-trained model found. Training from scratch.")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.gamma)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, valid_loader, loss_fn, optimizer, scheduler, args.epochs
    )

    print("Evaluating on validation and test set...")

    val_fpr, val_tpr, val_roc_auc, val_labels, val_preds, _, val_misclassified = compute_roc_auc_with_misclassifications(model, valid_loader, device, class_names)
    test_fpr, test_tpr, test_roc_auc, test_labels, test_preds, _, test_misclassified = compute_roc_auc_with_misclassifications(model, test_loader, device, class_names)

    plot_roc_curves(val_fpr, val_tpr, val_roc_auc, class_names, " (Validation Set)", results_dir)
    plot_confusion_matrix(val_labels, val_preds, class_names, " (Validation Set)", results_dir)
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, results_dir)
    save_misclassified_files_to_dict(val_misclassified, results_dir / "misclassified_val.pkl")

    plot_roc_curves(test_fpr, test_tpr, test_roc_auc, class_names, " (Test Set)", results_dir)
    plot_confusion_matrix(test_labels, test_preds, class_names, " (Test Set)", results_dir)
    save_misclassified_files_to_dict(test_misclassified, results_dir / "misclassified_test.pkl")

    print("Validation Classification Report:")
    val_report = classification_report(val_labels, val_preds, target_names=class_names)
    print(val_report)
    val_report_path = results_dir / "val_classification_report.txt"
    with open(val_report_path, "w") as f:
        f.write("Validation Classification Report:\n")
        f.write(val_report)

    print("\nTest Classification Report:")
    test_report = classification_report(test_labels, test_preds, target_names=class_names)
    print(test_report)
    test_report_path = results_dir / "test_classification_report.txt"
    with open(test_report_path, "w") as f:
        f.write("Test Classification Report:\n")
        f.write(test_report)

    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main()
