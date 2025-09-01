import os
import glob
import argparse
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import classification_report

from linformer import Linformer
from vit_pytorch.cvt import CvT
from vit_pytorch.efficient import ViT

from gwtorch.modules.general_utils import (
    compute_roc_auc_with_misclassifications,
    plot_roc_curves,
    plot_confusion_matrix,
    print_misclassified_files,
    save_misclassified_files_to_txt,
)

def setup_directories(results_dir, model_name):
    os.makedirs(f'./{results_dir}/{model_name}_results/Plots', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Test Transformer for GW classification")
    parser.add_argument('--model', type=str, default='ViT', choices=['CvT', 'ViT'], help='Model to train')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--model-path', type=str, default='model_alpha.pth', help="Model will be saved in models directory with extention as .pth")
    parser.add_argument('--path-name', type=str, default='data', help='Path to the dataset directory. Default is "data".')
    parser.add_argument('--results-dir', type=str, default='inference_results', help='Path to save the results of the inference made by the model.')
    return parser.parse_args()


class GWDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        arr = np.load(file_path)  # shape: (H, W, C)

        # Transpose to (H, W, C) only if needed
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]  # Convert to (H, W, 1)

        arr = arr.astype(np.float32)
        arr = torch.from_numpy(arr).float()
        
        arr = self.transform(arr)

        label_str = file_path.split("/")[2].lower()
        label = {'lensed': 0, 'eccentric': 1, 'unlensed': 2}[label_str]

        return arr, label, file_path

def custom_collate_fn(batch):
    images, labels, img_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), img_paths

def custom_collate_fn(batch):
    if len(batch[0]) == 2:  # (data, label)
        images, labels = zip(*batch)
        img_paths = [None] * len(batch)
        weights = None
    elif len(batch[0]) == 3:  # (data, label, path)
        images, labels, img_paths = zip(*batch)
        weights = None
    elif len(batch[0]) == 4:  # (data, label, dummy, path)
        images, labels, weights, img_paths = zip(*batch)
    else:
        raise ValueError(f"Unexpected sample format: {len(batch[0])} elements")

    return torch.stack(images), torch.tensor(labels), weights, img_paths


def load_all_data(path_name='data', batch_size=32):

    all_files = glob.glob(f'./{path_name}/*/*.npy')
    all_labels = [path.split('/')[2] for path in all_files]


    transform = transforms.Compose([
        # transforms.Lambda(lambda x: torch.tensor(x).permute(2, 0, 1)),
        transforms.Resize((224, 224)),
    ])

    test_data = GWDataset(all_files, transform)

    num_channels = test_data[0][0].shape[0] if isinstance(test_data[0][0], torch.Tensor) else 3

    # Create loaders

    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                              num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return num_channels, test_loader

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
    

def evaluate_and_plot(model, test_loader, device, results_dir, model_name='ViT'):
    class_names = ['Lensed', 'Eccentric', 'Quasi-Circular Unlensed']
    results_dir = Path(f'./{results_dir}/{model_name}_results')

    
    test_fpr, test_tpr, test_roc_auc, test_labels, test_preds, _, test_misclassified = compute_roc_auc_with_misclassifications(model, test_loader, device, class_names)

    print_misclassified_files(test_misclassified, "Test Set")

    save_misclassified_files_to_txt(test_misclassified, "Test", results_dir / "misclassified_test.txt")


    plot_roc_curves(test_fpr, test_tpr, test_roc_auc, class_names, " (Test Set)", results_dir)

    plot_confusion_matrix(test_labels, test_preds, class_names, " (Test Set)", results_dir)

    print("\nTest Classification Report:")
    test_report = classification_report(test_labels, test_preds, target_names=class_names)
    print(test_report)
    test_report_path = results_dir / "test_classification_report.txt"
    with open(test_report_path, "w") as f:
        f.write("Test Classification Report:\n")
        f.write(test_report)

    return test_misclassified

def main():
    args = parse_args()
    setup_directories(results_dir=args.results_dir, model_name=args.model)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    num_channels, test_loader = load_all_data(path_name=args.path_name, batch_size=args.batch_size)
    model = get_model(args.model, num_channels=num_channels, device=device)

    if os.path.exists(f'./models/{args.model_path}'):
        model.load_state_dict(torch.load(f'./models/{args.model_path}'))
        print(f"Model loaded from ./model/{args.model_path}")
    else:
        print("No pretrained model found. Training from scratch...")

    test_misclassified = evaluate_and_plot(model, test_loader, device, args.results_dir, model_name=args.model)

    print("\nTraining complete. Summary:")
    print(f"Test misclassifications: {len(test_misclassified)}")

if __name__ == "__main__":
    main()