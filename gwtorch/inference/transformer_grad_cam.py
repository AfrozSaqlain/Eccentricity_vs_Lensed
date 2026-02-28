import os
import glob
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# --- Grad-CAM Imports ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


from gwtorch.modules.ml_utils import seed_everything, get_model


# ================= ARGUMENTS =================
def parse_args():
    parser = argparse.ArgumentParser(description="Test Transformer for GW classification with Grad-CAM")
    parser.add_argument('--model', type=str, default='ViT', choices=['CvT', 'ViT', 'ViTLinear'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model-path', type=str, default='vit_model.pth')
    parser.add_argument('--path-name', type=str, default='data')
    parser.add_argument('--results-dir', type=str, default='inference_results')
    parser.add_argument('--num-classes', type=int, choices=[2, 3], default=3)
    return parser.parse_args()


# ================= DATASET =================
class GWDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        try:
            arr = torch.from_numpy(np.load(file_path).astype(np.float32))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros((1, 224, 224)), 0, file_path

        if self.transform:
            arr = self.transform(arr)

        label_str = file_path.split("/")[-2].lower()
        label_map = {
            'lensed': 0, 
            'eccentric': 1, 
            'unlensed': 2, 
            'quasi-circular unlensed': 2
        }
        label = label_map.get(label_str, 2)

        return arr, label, file_path


def custom_collate_fn(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), paths


def load_all_data(path_name='data', batch_size=32, num_classes=3):
    all_files = glob.glob(f'{path_name}/*/*.npy')
    if num_classes == 2:
        all_files = [p for p in all_files if Path(p).parent.name.lower() in {'lensed', 'eccentric'}]
    if not all_files:
        print(f"No files found in {path_name}")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
    ])

    dataset = GWDataset(all_files, transform=transform)
    sample_data = dataset[0][0]
    num_channels = sample_data.shape[0]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return num_channels, loader


# ================= GRAD-CAM HELPERS =================
def vit_reshape_transform(tensor, height=7, width=7):
    result = tensor[:, 1:, :] 
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def visualize_gradcam(model, loader, device, class_names, results_dir, model_name):
    print("\nGenerating Grad-CAM maps...")
    
    # 1. Select Target Layer
    if model_name in {'ViT', 'ViTLinear'}:
        target_layers = [model.transformer]
        reshape_transform = vit_reshape_transform
    elif model_name == 'CvT':
        target_layers = [model.layers[-1]] 
        reshape_transform = None 
    else:
        print(f"Unknown model type {model_name} for Grad-CAM")
        return

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    correct_samples = {i: [] for i in range(len(class_names))}
    
    model.eval()
    found_all = False
    
    for data, labels, paths in loader:
        if found_all: break
        
        data = data.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            
        indices_to_process = []
        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = preds[i].item()
            
            if true_label == pred_label and len(correct_samples[true_label]) < 3:
                indices_to_process.append(i)
                correct_samples[true_label].append(None) 

        if indices_to_process:
            input_tensor = data[indices_to_process]
            targets = [ClassifierOutputTarget(labels[idx].item()) for idx in indices_to_process]

            # Generate Grayscale CAM
            grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
            
            for local_idx, batch_idx in enumerate(indices_to_process):
                true_label = labels[batch_idx].item()
                
                # --- PREPARE IMAGE ---
                # 1. Convert to numpy (H, W, C)
                orig_tensor = data[batch_idx].cpu().permute(1, 2, 0).numpy()
                
                # 2. Normalize to 0-1
                orig_norm = (orig_tensor - orig_tensor.min()) / (orig_tensor.max() - orig_tensor.min() + 1e-8)
                
                # 3. Handle Channels (Fix for 4-channel input)
                if orig_norm.shape[2] != 3:
                    # Average channels to get grayscale intensity, then stack to RGB
                    orig_mean = np.mean(orig_norm, axis=2) 
                    orig_rgb = np.stack([orig_mean]*3, axis=2)
                else:
                    orig_rgb = orig_norm

                # 4. Get corresponding CAM mask
                cam_mask = grayscale_cams[local_idx, :]

                # --- MIRRORING (Vertical Flip) ---
                # Flip the image and the mask upside down so low frequencies appear at the bottom
                orig_rgb = np.flipud(orig_rgb)
                cam_mask = np.flipud(cam_mask)
                
                # 5. Blend
                visualization = show_cam_on_image(orig_rgb, cam_mask, use_rgb=True)
                
                # Store
                none_idx = correct_samples[true_label].index(None)
                correct_samples[true_label][none_idx] = (orig_rgb, visualization, paths[batch_idx])

        if all(len(v) == 3 and None not in v for v in correct_samples.values()):
            found_all = True

    # Save to Disk
    save_dir = Path(results_dir) / "GradCAM_Maps"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving Grad-CAM images to {save_dir}...")
    
    for cls_idx, samples in correct_samples.items():
        for i, item in enumerate(samples):
            if item is None: continue 
            
            orig, viz, fpath = item
            fname = Path(fpath).stem
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            
            ax[0].imshow(orig)
            ax[0].set_title(f"{class_names[cls_idx]} - Original")
            ax[0].axis('off')
            
            ax[1].imshow(viz)
            ax[1].set_title(f"Grad-CAM")
            ax[1].axis('off')
            
            plt.suptitle(f"File: {fname}")
            plt.tight_layout()
            plt.savefig(save_dir / f"{class_names[cls_idx]}_{i}_{fname}.png", dpi=150)
            plt.close(fig)
            
    print("Done.\n")


# ================= EVALUATION =================
def evaluate(model, loader, device, results_dir, model_name, class_names):
    results_dir = Path(f'./{results_dir}/{model_name}_results')
    os.makedirs(results_dir, exist_ok=True)

    model.eval()
    all_labels, all_preds = [], []

    print(f"Running Inference on {device}...")
    
    with torch.no_grad():
        for data, labels, _ in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names,
                                zero_division=0))

    visualize_gradcam(
        model,
        loader,
        device,
        class_names,
        results_dir,
        model_name
    )


# ================= MAIN =================
def main():
    args = parse_args()
    seed_everything(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # 1. Load Data
    num_channels, loader = load_all_data(
        path_name=args.path_name,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
    )

    # 2. Initialize Model
    model = get_model(args.model, num_channels, device, num_classes=args.num_classes)
    
    # 3. Load Weights
    model_path = f'./models/{args.model_path}'
    if not os.path.exists(model_path):
        model_path = args.model_path
    
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            return
    else:
        print(f"Warning: Model path {model_path} not found. Running with random weights.")

    # 4. Run
    class_names = ['Lensed', 'Eccentric'] if args.num_classes == 2 else ['Lensed', 'Eccentric', 'Quasi-Circular Unlensed']
    evaluate(model, loader, device, args.results_dir, args.model, class_names)


if __name__ == "__main__":
    main()
