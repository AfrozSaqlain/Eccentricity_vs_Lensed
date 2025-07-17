import argparse
import os
import random
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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
    parser.add_argument('--model', type=str, default='CvT', choices=['CvT', 'ViT'], help='Model to train')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

class GWDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)[:3, :, :]

        label_str = img_path.split("/")[-1].split("_")[0]
        label = {'lensed': 0, 'eccentric': 1, 'unlensed': 2}[label_str]
        file_number = img_path.split("/")[-1].split("_")[1]

        return img_transformed, label, file_number, img_path

def custom_collate_fn(batch):
    images, labels, file_numbers, img_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), file_numbers, img_paths

def load_data(seed, batch_size):
    train_list = glob.glob('../../data/train/*.png')
    test_list = glob.glob('../../data/test/*.png')
    labels = [path.split('/')[-1].split('_')[0] for path in train_list]

    train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, shuffle=True, random_state=seed)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_data = GWDataset(train_list, transform)
    valid_data = GWDataset(valid_list, transform)
    test_data = GWDataset(test_list, transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return train_loader, valid_loader, test_loader

def get_model(name, device):
    if name == 'CvT':
        return CvT(num_classes=3, s1_emb_dim=64, s1_emb_kernel=7, s1_emb_stride=4, s1_proj_kernel=3,
                   s1_kv_proj_stride=2, s1_heads=1, s1_depth=1, s1_mlp_mult=4, s2_emb_dim=192,
                   s2_emb_kernel=3, s2_emb_stride=2, s2_proj_kernel=3, s2_kv_proj_stride=2, s2_heads=3,
                   s2_depth=2, s2_mlp_mult=4, s3_emb_dim=384, s3_emb_kernel=3, s3_emb_stride=2,
                   s3_proj_kernel=3, s3_kv_proj_stride=2, s3_heads=4, s3_depth=10, s3_mlp_mult=4,
                   dropout=0.3).to(device)
    else:
        transformer = Linformer(dim=128, seq_len=50, depth=12, heads=8, k=64, dropout=0.2)
        return ViT(dim=128, image_size=224, patch_size=32, num_classes=3,
                   transformer=transformer, channels=3).to(device)

def train_model(model, train_loader, valid_loader, device, epochs, lr, gamma):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

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

    torch.save(model.state_dict(), './models/model_alpha.pth')
    return train_losses, val_losses, train_accs, val_accs

def evaluate_and_plot(model, valid_loader, test_loader, device):
    class_names = ['Lensed', 'Eccentric', 'Unlensed']
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
    print(classification_report(val_labels, val_preds, target_names=class_names))
    print("\nTest Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    return val_misclassified, test_misclassified

def main():
    args = parse_args()
    setup_directories()
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    train_loader, valid_loader, test_loader = load_data(args.seed, args.batch_size)
    model = get_model(args.model, device)

    if os.path.exists('./models/model_alpha.pth'):
        print("Pretrained model found. Loading...")
        model.load_state_dict(torch.load('./models/model_alpha.pth'))
    else:
        print("No pretrained model found. Training from scratch...")

    train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, valid_loader, device, args.epochs, args.lr, args.gamma)
    val_misclassified, test_misclassified = evaluate_and_plot(model, valid_loader, test_loader, device)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, Path('./results/transformer_results'))

    print("\nTraining complete. Summary:")
    print(f"Validation misclassifications: {len(val_misclassified)}")
    print(f"Test misclassifications: {len(test_misclassified)}")

if __name__ == "__main__":
    main()