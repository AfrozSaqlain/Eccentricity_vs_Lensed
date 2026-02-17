import os
import glob
import argparse
import numpy as np
import pandas as pd # <-- NEW: Added pandas dependency
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

from linformer import Linformer
from vit_pytorch.cvt import CvT
from vit_pytorch.efficient import ViT

from gwtorch.modules.general_utils import (
    print_misclassified_files,
    save_misclassified_files_to_txt,
    get_file_prefix,
    plot_roc_curves
)

from gwtorch.modules.ml_utils import (
    seed_everything,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Test Transformer for GW classification")
    parser.add_argument('--model', type=str, default='ViT', choices=['CvT', 'ViT'], help='Model to train')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--model-path', type=str, default='vit_model.pth', help="Path to the trained model (.pth file)")
    parser.add_argument('--path-name', type=str, default='data', help='Path to the dataset directory.')
    parser.add_argument('--results-dir', type=str, default='inference_results', help='Path to save inference results.')
    parser.add_argument('--meta-csv', type=str, default='./results/data_parameters.csv', help='Path to the metadata CSV file.')
    parser.add_argument('--lnB-min', type=float, default=None, help='Minimum lnB value for confusion matrix filtering.')
    parser.add_argument('--lnB-max', type=float, default=None, help='Maximum lnB value for confusion matrix filtering.')
    return parser.parse_args()


# <-- MODIFIED: Using your provided GWDataset class -->
class GWDataset(Dataset):
    def __init__(self, file_list, transform=None, meta_csv='./results/data_parameters.csv'):
        self.file_list = file_list
        self.transform = transform
        try:
            self.df = pd.read_csv(meta_csv)
            # set index if sample column exists
            if 'sample' in self.df.columns:
                self.df = self.df.set_index('sample')
        except FileNotFoundError:
            print(f"Warning: Metadata file not found at {meta_csv}. lnB values will be 0.")
            self.df = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # Assumes loaded array is [C, H, W]
        arr = np.load(file_path).astype(np.float32)
        arr = torch.from_numpy(arr)

        if self.transform:
            arr = self.transform(arr)

        label_str = file_path.split("/")[-2].lower() # Safer way to get class folder
        label = {'lensed': 0, 'eccentric': 1, 'unlensed': 2}[label_str]

        ln_B = 0.0
        if self.df is not None:
            try:
                # Assumes filename is like '.../123.npy'
                file_number = int(os.path.basename(file_path).split('.')[0])

                if file_number in self.df.index:
                    if label_str == 'lensed' and 'lens_lnB' in self.df.columns:
                        ln_B = self.df.loc[file_number, 'lens_lnB']
                    elif label_str == 'eccentric' and 'ecc_lnB' in self.df.columns:
                        ln_B = self.df.loc[file_number, 'ecc_lnB']
                    elif label_str == 'unlensed':
                        # Assign a high value for unlensed if needed for filtering
                        ln_B = 100
            except (ValueError, KeyError) as e:
                # Handle cases where filename isn't a number or key is missing
                pass

        return arr, label, ln_B, file_path


# <-- MODIFIED: Updated collate function to handle ln_B -->
def custom_collate_fn(batch):
    images, labels, lnBs, img_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), lnBs, img_paths


def load_all_data(path_name='data', batch_size=32, meta_csv='./results/data_parameters.csv'):
    all_files = glob.glob(f'{path_name}/*/*.npy')

    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
    ])

    test_data = GWDataset(all_files, transform=transform, meta_csv=meta_csv)

    num_channels = test_data[0][0].shape[0]

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
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
    else: # ViT
        transformer = Linformer(dim=128, seq_len=50, depth=12, heads=8, k=64, dropout=0.4)
        return ViT(dim=128, image_size=224, patch_size=32, num_classes=3,
                   transformer=transformer, channels=num_channels).to(device)


# <-- MODIFIED: Function now collects and returns lnB values -->
def compute_roc_auc_with_misclassifications(model, data_loader, device, class_names):
    model.eval()
    all_labels, all_predictions, all_probabilities, all_img_paths, all_lnBs = [], [], [], [], []

    with torch.no_grad():
        for data, labels, lnBs, img_paths in data_loader: # <-- Updated loop unpacking
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_img_paths.extend(img_paths)
            all_lnBs.extend(lnBs) # <-- Collect lnB values

    all_labels, all_predictions = np.array(all_labels), np.array(all_predictions)
    all_probabilities, all_lnBs = np.array(all_probabilities), np.array(all_lnBs)

    # Find misclassified samples
    misclassified_mask = all_labels != all_predictions
    misclassified_indices = np.where(misclassified_mask)[0]
    misclassified_info = [{
        'file_path': all_img_paths[idx],
        'true_class': class_names[all_labels[idx]],
        'predicted_class': class_names[all_predictions[idx]],
    } for idx in misclassified_indices]

    # ROC computation (remains the same)
    label_binarizer = LabelBinarizer().fit(all_labels)
    y_true_binary = label_binarizer.transform(all_labels)
    n_classes = y_true_binary.shape[1]
    
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], all_probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = sum(np.interp(all_fpr, fpr[i], tpr[i]) for i in range(n_classes)) / n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc, all_labels, all_predictions, misclassified_info, all_lnBs


def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix="", results_dir=None):
    if len(y_true) == 0:
        print(f"Skipping confusion matrix for '{title_suffix}': No data points after filtering.")
        return

    prefix = get_file_prefix(results_dir) if results_dir else ''
    title = title_suffix.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = np.divide(cm.astype('float') * 100, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    annotations = [[f"{cm_percentage[i, j]:.1f}%" for j in range(cm.shape[1])] for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=annotations, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(rf'Confusion Matrix{title_suffix}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()

    if results_dir:
        results_dir = Path(results_dir)
        os.makedirs(results_dir / 'Plots', exist_ok=True)
        fig.savefig(results_dir / f'Plots/{prefix}_Confusion_Matrix{title}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# <-- MODIFIED: Function now accepts args to perform filtering -->
def evaluate_and_plot(model, test_loader, device, results_dir, model_name, args):
    class_names = ['Lensed', 'Eccentric', 'Quasi-Circular Unlensed']
    results_dir = Path(f'./{results_dir}/{model_name}_results')

    test_fpr, test_tpr, test_roc_auc, test_labels, test_preds, test_misclassified, test_lnBs = \
        compute_roc_auc_with_misclassifications(model, test_loader, device, class_names)

    plot_roc_curves(test_fpr, test_tpr, test_roc_auc, class_names, " (Test Set)", results_dir)

    # Plot unfiltered confusion matrix first
    plot_confusion_matrix(test_labels, test_preds, class_names, " (Test Set, All Data)", results_dir)

    # --- NEW: Filtering Logic based on args ---
    if args.lnB_min is not None or args.lnB_max is not None:
        mask = np.ones_like(test_lnBs, dtype=bool)
        filter_parts = []

        if args.lnB_min is not None:
            mask &= (test_lnBs >= args.lnB_min)
            filter_parts.append(f'lnB >= {args.lnB_min}')

        if args.lnB_max is not None:
            mask &= (test_lnBs <= args.lnB_max)
            filter_parts.append(f'lnB <= {args.lnB_max}')

        filtered_labels = test_labels[mask]
        filtered_preds = test_preds[mask]

        filter_str = ' and '.join(filter_parts)
        title_suffix = f" (Test Set, Filtered by {filter_str})"

        print("-" * 50)
        print(f"Plotting filtered confusion matrix for: {filter_str}")
        print(f"{len(filtered_labels)} out of {len(test_labels)} samples remaining.")
        print("-" * 50)

        plot_confusion_matrix(filtered_labels, filtered_preds, class_names, title_suffix, results_dir)
    
    # Classification report on all data
    print("\nTest Classification Report (on all test data):")
    report = classification_report(test_labels, test_preds, target_names=class_names, zero_division=0)
    print(report)
    with open(results_dir / "test_classification_report.txt", "w") as f:
        f.write("Test Classification Report:\n" + report)


def main():
    args = parse_args()
    os.makedirs(f'./{args.results_dir}/{args.model}_results/Plots', exist_ok=True)
    seed_everything(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    num_channels, test_loader = load_all_data(
        path_name=args.path_name,
        batch_size=args.batch_size,
        meta_csv=args.meta_csv # <-- Pass meta_csv path
    )
    model = get_model(args.model, num_channels=num_channels, device=device)

    model_full_path = f'./models/{args.model_path}'
    if os.path.exists(model_full_path):
        model.load_state_dict(torch.load(model_full_path, map_location=device))
        print(f"Model loaded from {model_full_path}")
    else:
        print(f"Error: No model found at {model_full_path}. Please provide a valid model path.")
        return

    evaluate_and_plot(model, test_loader, device, args.results_dir, args.model, args)


if __name__ == "__main__":
    main()