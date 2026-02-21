import os
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from gwtorch.modules.ml_utils import seed_everything, get_model
from gwtorch.modules.data_loading import safe_num_workers
from gwtorch.modules.pathing import class_from_file_path


CLASS_NAMES = ['Lensed', 'Eccentric', 'Quasi-Circular Unlensed']
LABEL_MAP = {
    'lensed': 0,
    'eccentric': 1,
    'unlensed': 2,
    'quasi-circular unlensed': 2,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM maps for correctly classified CNN samples (one per class).')
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN', 'SimpleCNN'], help='CNN architecture to evaluate')
    parser.add_argument('--model-path', type=str, default='cnn_model.pth', help='Model file name inside models directory')
    parser.add_argument('--path-name', type=str, default='data', help='Path to dataset directory containing class folders')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for finding correctly classified samples')
    parser.add_argument('--samples-per-class', type=int, default=1, help='Number of correctly classified samples per class to generate Grad-CAM for')
    parser.add_argument('--results-dir', type=str, default='inference_results', help='Base directory for output artifacts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--flip-x', dest='flip_x', action='store_true', help='Flip images along x-axis before any transforms.')
    parser.add_argument('--no-flip-x', dest='flip_x', action='store_false', help='Disable x-axis flip.')
    parser.set_defaults(flip_x=False)
    return parser.parse_args()


class GWDataset(Dataset):
    def __init__(self, file_list, transform=None, flip_x=True):
        self.file_list = file_list
        self.transform = transform
        self.flip_x = flip_x

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        arr = np.load(file_path).astype(np.float32)
        if self.flip_x:
            # CHW -> flip height (vertical) to mirror along x-axis.
            arr = np.flip(arr, axis=1).copy()
        arr = torch.from_numpy(arr)

        if self.transform:
            arr = self.transform(arr)

        label_str = class_from_file_path(file_path).lower()
        if label_str not in LABEL_MAP:
            raise ValueError(f'Unknown class folder: {label_str} ({file_path})')
        label = LABEL_MAP[label_str]

        return arr, label, file_path


def custom_collate_fn(batch):
    images, labels, file_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), file_paths


def load_data(path_name='data', batch_size=128, flip_x=True):
    all_files = sorted(glob.glob(f'{path_name}/*/*.npy'))
    if not all_files:
        raise FileNotFoundError(f'No .npy files found under {path_name}/*/*.npy')

    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
    ])

    dataset = GWDataset(all_files, transform=transform, flip_x=flip_x)
    num_channels = dataset[0][0].shape[0]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_num_workers(),
        collate_fn=custom_collate_fn,
    )
    return num_channels, loader


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor, target_class):
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)
        score = output[:, target_class].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(0).squeeze(0)

        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        probs = F.softmax(output, dim=1)
        pred = int(output.argmax(dim=1).item())
        confidence = float(probs[0, pred].item())

        return cam.detach().cpu().numpy(), pred, confidence


def select_correct_samples(model, loader, device, samples_per_class=1):
    found = {i: [] for i in range(3)}

    model.eval()
    with torch.no_grad():
        for images, labels, file_paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                label = int(labels[i].item())
                pred = int(preds[i].item())

                if label == pred and len(found[label]) < samples_per_class:
                    found[label].append({
                        'image': images[i].detach().cpu(),
                        'label': label,
                        'pred': pred,
                        'file_path': file_paths[i],
                    })

                if all(len(v) >= samples_per_class for v in found.values()):
                    return found

    return found


def tensor_to_rgb(image_tensor):
    img = image_tensor.numpy()
    img = np.transpose(img, (1, 2, 0))

    img_min = img.min()
    img_max = img.max()
    img = (img - img_min) / (img_max - img_min + 1e-8)

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] != 3:
        gray = img.mean(axis=2, keepdims=True)
        img = np.repeat(gray, 3, axis=2)

    return img


def overlay_cam_on_image(rgb_image, cam_map, alpha=0.45):
    heatmap = plt.get_cmap('jet')(cam_map)[..., :3]
    overlay = (1 - alpha) * rgb_image + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return heatmap, overlay


def save_gradcam_figure(rgb_image, cam_map, overlay, out_path, class_name, file_path, pred_conf):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(rgb_image)
    axes[0].set_title('Input')
    axes[0].axis('off')

    axes[1].imshow(cam_map, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    fig.suptitle(f'{class_name} | pred conf: {pred_conf:.3f} | {Path(file_path).name}', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    num_channels, loader = load_data(path_name=args.path_name, batch_size=args.batch_size, flip_x=args.flip_x)

    model = get_model(name=args.model, num_channels=num_channels, device=device)
    model_file = Path('./models') / args.model_path
    if not model_file.exists():
        raise FileNotFoundError(f'No pre-trained model found at {model_file}.')
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # Match Grad-CAM target layer to architecture depth.
    if args.model == 'CNN':
        target_layer = model.conv3
    else:
        target_layer = model.conv2

    if args.samples_per_class < 1:
        raise ValueError('--samples-per-class must be >= 1')

    selected = select_correct_samples(
        model=model,
        loader=loader,
        device=device,
        samples_per_class=args.samples_per_class,
    )
    if all(len(v) == 0 for v in selected.values()):
        raise RuntimeError('No correctly classified samples were found in the provided dataset.')

    output_dir = Path(args.results_dir) / f'{args.model}_results' / 'GradCAM_Maps'
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_generator = GradCAM(model, target_layer)
    try:
        for class_idx in range(3):
            if len(selected[class_idx]) == 0:
                print(f'Warning: No correctly classified sample found for class {CLASS_NAMES[class_idx]}.')
                continue

            for sample_idx, sample in enumerate(selected[class_idx], start=1):
                image_tensor = sample['image'].unsqueeze(0).to(device)

                cam_map, pred_idx, pred_conf = cam_generator.generate(image_tensor, sample['pred'])
                rgb = tensor_to_rgb(sample['image'])
                _, overlay = overlay_cam_on_image(rgb, cam_map)

                out_file = output_dir / (
                    f'class_{class_idx}_{CLASS_NAMES[class_idx].replace(" ", "_")}'
                    f'_sample{sample_idx}_{Path(sample["file_path"]).stem}.png'
                )
                save_gradcam_figure(
                    rgb_image=rgb,
                    cam_map=cam_map,
                    overlay=overlay,
                    out_path=out_file,
                    class_name=CLASS_NAMES[class_idx],
                    file_path=sample['file_path'],
                    pred_conf=pred_conf,
                )
                print(f'Saved: {out_file}')
    finally:
        cam_generator.remove_hooks()


if __name__ == '__main__':
    main()
