import os
import sys
import glob
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

from gwtorch.modules.general_utils import (
    compute_roc_auc_with_misclassifications,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_training_curves,
)
from gwtorch.modules.neural_net import CNN_Model

# ----------------- Constants -----------------
RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 1
LR = 3e-4
GAMMA = 0.7
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_names = ["Lensed", "Eccentric", "Unlensed"]

# ----------------- Dataset Class -----------------
class GWDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img = self.transform(img)[:3, :, :]
        label_str = img_path.split("/")[-1].split("_")[0]
        label = {"lensed": 0, "eccentric": 1, "unlensed": 2}.get(label_str, 0)
        file_number = img_path.split("/")[-1].split("_")[1]
        return img, label, file_number, img_path

# ----------------- Utility Functions -----------------
def custom_collate_fn(batch):
    images, labels, file_numbers, img_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), file_numbers, img_paths

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN model for GW classification")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--gamma', type=float, default=GAMMA)
    parser.add_argument('--model_path', type=str, default='./models/cnn_model0.pth')
    return parser.parse_args()

def setup_data_loaders(train_dir, test_dir, batch_size):
    train_list = glob.glob(os.path.join(train_dir, '*.png'))
    test_list = glob.glob(os.path.join(test_dir, '*.png'))

    labels = [path.split('/')[-1].split('_')[0] for path in train_list]
    train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, shuffle=True, random_state=RANDOM_SEED)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_data = GWDataset(train_list, transform)
    valid_data = GWDataset(valid_list, transform)
    test_data = GWDataset(test_list, transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return train_loader, valid_loader, test_loader

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

    train_loader, valid_loader, test_loader = setup_data_loaders('./data/train', './data/test', args.batch_size)

    model = CNN_Model().to(device)
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

    print("Evaluating on test set...")
    fpr, tpr, roc_auc, labels, predictions, _, misclassified = compute_roc_auc_with_misclassifications(
        model=model, data_loader=test_loader, device=device
    )

    plot_roc_curves(fpr, tpr, roc_auc, class_names, " (Test Set)", results_dir)
    plot_confusion_matrix(labels, predictions, class_names, " (Test Set)", results_dir)
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, results_dir)

    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main()
