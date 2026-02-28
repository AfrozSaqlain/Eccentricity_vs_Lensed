import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from gwtorch.modules.ml_utils import get_model, seed_everything


CLASS_NAMES = ["Lensed", "Eccentric", "Quasi-Circular Unlensed"]
LABEL_MAP = {"lensed": 0, "eccentric": 1, "unlensed": 2, "quasi-circular unlensed": 2}


def get_class_names(num_classes):
    return ["Lensed", "Eccentric"] if num_classes == 2 else CLASS_NAMES


def get_label_map(num_classes):
    return {"lensed": 0, "eccentric": 1} if num_classes == 2 else LABEL_MAP


def class_from_file_path(file_path):
    return Path(file_path).parent.name


def discover_npy_files(path_name):
    return sorted(glob.glob(f"{path_name}/*/*.npy"))


def safe_num_workers():
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def default_transform_224(antialias=False):
    return transforms.Compose([
        transforms.Resize((224, 224), antialias=antialias),
    ])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate transformer attention maps for correctly classified GW samples."
    )
    parser.add_argument("--model", type=str, default="ViT", choices=["ViT", "ViTLinear"])
    parser.add_argument("--model-path", type=str, default="vit_model.pth")
    parser.add_argument("--path-name", type=str, default="data", help="Dataset root with class subfolders.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-class", type=int, default=1, help="Correctly classified samples per class.")
    parser.add_argument("--results-dir", type=str, default="inference_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha in [0,1]")
    parser.add_argument("--num-classes", type=int, choices=[2, 3], default=3)
    parser.add_argument("--flip-x", dest="flip_x", action="store_true", help="Flip input along x-axis before plotting")
    parser.add_argument("--no-flip-x", dest="flip_x", action="store_false", help="Disable x-axis flip")
    parser.set_defaults(flip_x=False)
    return parser.parse_args()


class GWDataset(Dataset):
    def __init__(self, file_list, label_map, transform=None, flip_x=True):
        self.file_list = file_list
        self.label_map = label_map
        self.transform = transform
        self.flip_x = flip_x

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        arr = np.load(file_path).astype(np.float32)
        if self.flip_x:
            arr = np.flip(arr, axis=1).copy()
        arr = torch.from_numpy(arr)

        if self.transform:
            arr = self.transform(arr)

        label_str = class_from_file_path(file_path).lower()
        if label_str not in self.label_map:
            raise ValueError(f"Unknown class folder: {label_str} ({file_path})")
        label = self.label_map[label_str]
        return arr, label, file_path


def custom_collate_fn(batch):
    images, labels, file_paths = zip(*batch)
    return torch.stack(images), torch.tensor(labels), file_paths


def find_last_attention_module(model):
    candidates = []
    for module in model.modules():
        if hasattr(module, "to_q") and hasattr(module, "to_k"):
            candidates.append(module)
        elif hasattr(module, "to_qkv"):
            candidates.append(module)

    if not candidates:
        raise RuntimeError(
            "Could not find a compatible attention module. Use model=ViT or ViTLinear from this project."
        )
    return candidates[-1]


def compute_attention_from_module(module, tokens):
    heads = int(getattr(module, "heads", 8))

    if hasattr(module, "to_q") and hasattr(module, "to_k"):
        q = module.to_q(tokens)
        k = module.to_k(tokens)
    elif hasattr(module, "to_qkv"):
        qkv = module.to_qkv(tokens)
        q, k, _ = qkv.chunk(3, dim=-1)
    else:
        raise RuntimeError("Selected module does not expose query/key projections.")

    bsz, seq_len, _ = q.shape
    q = q.view(bsz, seq_len, heads, -1).transpose(1, 2)
    k = k.view(bsz, seq_len, heads, -1).transpose(1, 2)

    scale = getattr(module, "scale", q.shape[-1] ** -0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = torch.softmax(scores, dim=-1)

    cls_to_patches = attn[:, :, 0, 1:].mean(dim=1)
    return cls_to_patches


def attention_to_heatmap(cls_attention, image_size=224):
    patch_count = cls_attention.shape[-1]
    side = int(patch_count ** 0.5)
    if side * side != patch_count:
        raise RuntimeError(
            f"Attention patches are not square (got {patch_count}). This can happen with projected-attention variants."
        )

    heatmap = cls_attention.reshape(1, 1, side, side)
    heatmap = F.interpolate(heatmap, size=(image_size, image_size), mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze().detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def to_rgb(image_tensor):
    image = image_tensor.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] != 3:
        gray = image.mean(axis=2, keepdims=True)
        image = np.repeat(gray, 3, axis=2)
    return image


def save_attention_figure(rgb_image, heatmap, overlay, output_path, sample_name, true_label_name, pred_label_name, confidence):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(rgb_image)
    axes[0].set_title("Input GW")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Transformer Attention")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"{sample_name} | true: {true_label_name} | pred: {pred_label_name} ({confidence:.3f})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be >= 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    files = discover_npy_files(args.path_name)
    if args.num_classes == 2:
        files = [p for p in files if class_from_file_path(p).lower() in {"lensed", "eccentric"}]
    if not files:
        raise FileNotFoundError(f"No .npy files found under {args.path_name}/*/*.npy")

    transform = default_transform_224(antialias=False)
    class_names = get_class_names(args.num_classes)
    label_map = get_label_map(args.num_classes)
    dataset = GWDataset(files, label_map=label_map, transform=transform, flip_x=args.flip_x)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=safe_num_workers(),
        collate_fn=custom_collate_fn,
    )

    num_channels = dataset[0][0].shape[0]
    model = get_model(name=args.model, num_channels=num_channels, device=device, num_classes=args.num_classes)

    model_file = Path(args.model_path)
    if not model_file.exists():
        model_file = Path("./models") / args.model_path
    if not model_file.exists():
        raise FileNotFoundError(f"No pre-trained model found at {args.model_path} or {Path('./models') / args.model_path}.")
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    attn_module = find_last_attention_module(model)
    captured_tokens = {}

    def hook_fn(_module, module_input, _module_output):
        captured_tokens["tokens"] = module_input[0]

    handle = attn_module.register_forward_hook(hook_fn)

    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    output_dir = Path(args.results_dir) / f"{args.model}_results" / "Attention_Maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    collected = {i: 0 for i in range(len(class_names))}
    target = args.samples_per_class

    try:
        for images, labels, file_paths in loader:
            if all(collected[c] >= target for c in collected):
                break

            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                tokens_batch = captured_tokens.get("tokens")

            if tokens_batch is None:
                raise RuntimeError("Failed to capture transformer tokens for attention computation.")

            for i in range(images.shape[0]):
                true_idx = int(labels[i].item())
                pred_idx = int(preds[i].item())

                if true_idx != pred_idx:
                    continue
                if collected[true_idx] >= target:
                    continue

                cls_attention = compute_attention_from_module(attn_module, tokens_batch[i:i + 1])
                heatmap = attention_to_heatmap(cls_attention, image_size=images.shape[-1])

                rgb_image = to_rgb(images[i].detach().cpu())
                jet = plt.get_cmap("jet")(heatmap)[..., :3]
                overlay = np.clip((1 - alpha) * rgb_image + alpha * jet, 0.0, 1.0)

                sample_path = Path(file_paths[i])
                sample_idx = collected[true_idx] + 1
                output_png = output_dir / f"{class_names[true_idx]}_{sample_idx}_{sample_path.stem}.png"

                save_attention_figure(
                    rgb_image=rgb_image,
                    heatmap=heatmap,
                    overlay=overlay,
                    output_path=output_png,
                    sample_name=sample_path.name,
                    true_label_name=class_names[true_idx],
                    pred_label_name=class_names[pred_idx],
                    confidence=float(probs[i, pred_idx].item()),
                )

                collected[true_idx] += 1
                print(f"Saved attention overlay: {output_png}")

                if all(collected[c] >= target for c in collected):
                    break
    finally:
        handle.remove()

    print(
            "Completed attention generation for correctly classified samples: "
            + ", ".join([f"{class_names[k]}={v}" for k, v in collected.items()])
        )


if __name__ == "__main__":
    main()
