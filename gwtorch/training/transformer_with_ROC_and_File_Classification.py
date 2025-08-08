import argparse
import os
import numpy as np
import random
import glob
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from linformer import Linformer
from vit_pytorch.efficient import ViT
from vit_pytorch.cvt import CvT

from gwtorch.modules.general_utils import (
    compute_roc_auc_with_misclassifications,
    plot_roc_curves,
    plot_confusion_matrix,
    print_misclassified_files,
    save_misclassified_files_to_txt,
    plot_training_curves,
)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_directories():
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results/transformer_results/Plots', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer for GW classification")
    parser.add_argument('--model', type=str, default='ViT', choices=['CvT', 'ViT'], help='Model to train')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-path', type=str, default='model_alpha.pth', help="Model will be saved in models directory with extention as .pth")
    parser.add_argument('--num-channels', type=int, default=3, help="Defines what is the input channels of the model. Default is 3 for signals from 3 detectors.")
    return parser.parse_args()

# class GWDataset(Dataset):
#     def __init__(self, file_list, transform=None):
#         self.file_list = file_list
#         self.transform = transform

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         img_path = self.file_list[idx]
#         # img = Image.open(img_path)
#         # img_transformed = self.transform(img)[:3, :, :]
#         img = Image.open(img_path)#.convert("RGB")  
#         img = self.transform(img)

#         label_str = img_path.split("/")[-1].split("_")[0]
#         label = {'lensed': 0, 'eccentric': 1, 'unlensed': 2}[label_str]
#         file_number = img_path.split("/")[-1].split("_")[1]

#         return img, label, file_number, img_path


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

def custom_collate_fn(batch):
    images, labels, file_numbers, img_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), file_numbers, img_paths

# def load_data(seed, batch_size):
#     train_list = glob.glob('./data/train/*.npy')
#     test_list = glob.glob('./data/test/*.npy')
#     labels = [path.split('/')[-1].split('_')[0] for path in train_list]

#     train_list, valid_list = train_test_split(
#         train_list, test_size=0.2, stratify=labels, shuffle=True, random_state=seed
#     )

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((224, 224)),
#         # transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
#     ])

#     train_data = GWDataset(train_list, transform)
#     valid_data = GWDataset(valid_list, transform)
#     test_data  = GWDataset(test_list,  transform)

#     label_to_idx = {"lensed": 0, "eccentric": 1, "unlensed": 2}
#     train_labels = [path.split('/')[-1].split('_')[0] for path in train_list]
#     label_counts = Counter(train_labels)
#     class_counts = [label_counts['lensed'], label_counts['eccentric'], label_counts['unlensed']]
#     class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
#     class_weights = class_weights / class_weights.sum()

#     sample_weights = [class_weights[label_to_idx[label]] for label in train_labels]
#     sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

#     train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler,
#                               num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

#     valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
#                               num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

#     test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False,
#                               num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

#     return train_loader, valid_loader, test_loader, class_weights


def load_all_data(seed=42, batch_size=32):
    # Load all files from both train and test folders
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

    # Create datasets
    train_data = GWDataset(train_files, transform)
    val_data   = GWDataset(val_files, transform)
    test_data  = GWDataset(test_files, transform)

    # Create loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader

def get_model(name, num_channels, device):
    if name == 'CvT':
        return CvT(num_classes=3, s1_emb_dim=64, s1_emb_kernel=7, s1_emb_stride=4, s1_proj_kernel=3,
                   s1_kv_proj_stride=2, s1_heads=1, s1_depth=1, s1_mlp_mult=4, s2_emb_dim=192,
                   s2_emb_kernel=3, s2_emb_stride=2, s2_proj_kernel=3, s2_kv_proj_stride=2, s2_heads=3,
                   s2_depth=2, s2_mlp_mult=4, s3_emb_dim=384, s3_emb_kernel=3, s3_emb_stride=2,
                   s3_proj_kernel=3, s3_kv_proj_stride=2, s3_heads=4, s3_depth=10, s3_mlp_mult=4,
                   dropout=0.4).to(device)
    else:
        transformer = Linformer(dim=128, seq_len=50, depth=12, heads=8, k=64, dropout=0.4)
        return ViT(dim=128, image_size=224, patch_size=32, num_classes=3,
                   transformer=transformer, channels=num_channels).to(device)

def train_model(model, train_loader, valid_loader, device, epochs, lr, gamma): #class_weights, device, epochs, lr, gamma):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=gamma)
    # weights = class_weights.detach().clone().to(device)
    criterion = nn.CrossEntropyLoss()#weight=weights)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, running_acc = 0, 0
        for data, labels, *_ in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            acc = (output.argmax(1) == labels).float().mean()
            running_loss += loss.item()
            running_acc += acc.item()

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(running_acc / len(train_loader))

        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for data, labels, *_ in valid_loader:
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = criterion(output, labels)
                acc = (output.argmax(1) == labels).float().mean()
                val_loss += loss.item()
                val_acc += acc.item()

        val_losses.append(val_loss / len(valid_loader))
        val_accs.append(val_acc / len(valid_loader))

        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Acc={train_accs[-1]:.4f}; Val Loss={val_losses[-1]:.4f}, Acc={val_accs[-1]:.4f}")

    return train_losses, val_losses, train_accs, val_accs

def evaluate_and_plot(model, valid_loader, test_loader, device):
    class_names = ['Lensed', 'Eccentric', ' Quasi-Circular Unlensed']
    results_dir = Path('./results/transformer_results')

    val_fpr, val_tpr, val_roc_auc, val_labels, val_preds, _, val_misclassified = compute_roc_auc_with_misclassifications(model, valid_loader, device, class_names)
    test_fpr, test_tpr, test_roc_auc, test_labels, test_preds, _, test_misclassified = compute_roc_auc_with_misclassifications(model, test_loader, device, class_names)

    print_misclassified_files(val_misclassified, "Validation Set")
    print_misclassified_files(test_misclassified, "Test Set")
    save_misclassified_files_to_txt(val_misclassified, "Validation", results_dir / "misclassified_validation.txt")
    save_misclassified_files_to_txt(test_misclassified, "Test", results_dir / "misclassified_test.txt")

    plot_roc_curves(val_fpr, val_tpr, val_roc_auc, class_names, " (Validation Set)", results_dir)
    plot_roc_curves(test_fpr, test_tpr, test_roc_auc, class_names, " (Test Set)", results_dir)
    plot_confusion_matrix(val_labels, val_preds, class_names, " (Validation Set)", results_dir)
    plot_confusion_matrix(test_labels, test_preds, class_names, " (Test Set)", results_dir)

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

    return val_misclassified, test_misclassified

def main():
    args = parse_args()
    setup_directories()
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    # train_loader, valid_loader, test_loader, class_weights = load_data(args.seed, args.batch_size)
    train_loader, valid_loader, test_loader = load_all_data(args.seed, args.batch_size)
    model = get_model(args.model, args.num_channels, device)

    if os.path.exists(f'./models/{args.model_path}'):
        print("Pretrained model found. Loading...")
        model.load_state_dict(torch.load(f'./models/{args.model_path}'))
    else:
        print("No pretrained model found. Training from scratch...")

    # train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, valid_loader, class_weights, device, args.epochs, args.lr, args.gamma)
    train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, valid_loader, device, args.epochs, args.lr, args.gamma)
    val_misclassified, test_misclassified = evaluate_and_plot(model, valid_loader, test_loader, device)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, Path('./results/transformer_results'))

    print("\nTraining complete. Summary:")
    print(f"Validation misclassifications: {len(val_misclassified)}")
    print(f"Test misclassifications: {len(test_misclassified)}")

    torch.save(model.state_dict(), f'./models/{args.model_path}')
    print(f"Model saved to ./models/{args.model_path}")

if __name__ == "__main__":
    main()