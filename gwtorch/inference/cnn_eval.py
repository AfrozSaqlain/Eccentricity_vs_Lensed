import os
import glob
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import argparse

import numpy as np

from gwtorch.modules.general_utils import (
    compute_roc_auc_with_misclassifications,
    plot_roc_curves,
    plot_confusion_matrix,
    save_misclassified_files_to_dict
)
from gwtorch.modules.neural_net import CNN_Model

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CNN model on test data")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth)")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the directory with test data")
    parser.add_argument("--results_dir", type=str, default="./results/cnn_results", help="Directory to store evaluation results")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for DataLoader")

    args = parser.parse_args()

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_test_data(test_dir, transform):
    test_list = glob.glob(os.path.join(test_dir, '*.png'))
    print(f"Test Data: {len(test_list)}")
    return GWDataset(test_list, transform=transform)

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

class GWDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

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

        file_number = img_path.split("/")[-1].split("_")[1]
        return img_transformed, label, file_number, img_path

def custom_collate_fn(batch):
    images, labels, file_numbers, img_paths = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels, file_numbers, img_paths

def load_model(model_path, device):
    model = CNN_Model().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"No pre-trained model found at {model_path}.")
    return model

def evaluate(model, test_loader, device, class_names, results_dir):
    fpr, tpr, roc_auc, labels, predictions, _, misclassified = compute_roc_auc_with_misclassifications(
        model=model, data_loader=test_loader, device=device, class_names=class_names
    )

    roc_fig = plot_roc_curves(fpr=fpr, tpr=tpr, roc_auc=roc_auc, class_names=class_names, title_suffix=" (Test Set)", results_dir=results_dir)
    plot_confusion_matrix(y_true=labels, y_pred=predictions, class_names=class_names, title_suffix=" (Test Set)", results_dir=results_dir)
    save_misclassified_files_to_dict(misclassified, results_dir / "misclassified_test.pkl")

def main():
    results_dir = Path('./results/cnn_results')
    test_dir = './data/test'
    model_path = './models/cnn_model0.pth'
    class_names = ["Lensed", "Eccentric", "Unlensed"]

    os.makedirs('./results', exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = get_device()
    transform = get_transforms()
    test_data = prepare_test_data(test_dir, transform)

    test_loader = DataLoader(
        dataset=test_data,
        num_workers=os.cpu_count(),
        batch_size=128,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    model = load_model(model_path, device)
    evaluate(model, test_loader, device, class_names, results_dir)

if __name__ == '__main__':
    main()
