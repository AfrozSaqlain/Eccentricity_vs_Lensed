import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from gwtorch.modules.neural_net import CNN_Model
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from gwtorch.modules.general_utils import (
    compute_roc_auc_with_misclassifications,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_training_curves,
)

def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix="", results_dir=None):
    """
    Plot confusion matrix with counts and percentages.
    
    Args:
        y_true (List[int]): True class labels.
        y_pred (List[int]): Predicted class labels.
        class_names (List[str]): List of class label names.
        title_suffix (str): Optional suffix for plot title.
        results_dir (Path or str): Directory to save plot. If None, does not save.
    """
    # Handle title naming
    if "(" in title_suffix:
        try:
            title = title_suffix.split(" Set")[0].split("(")[1]
        except IndexError:
            title = "Confusion_Matrix"
    else:
        title = "Confusion_Matrix"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # Format annotations with count + percentage
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_percentage[i, j]
            annotations[i, j] = f"{count}\n({percent:.1f}%)"

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix{title_suffix}', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    # Save if requested
    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / f"Confusion_Matrix_{title}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()

# --- Models ---
lensed_unlensed_model = CNN_Model(input_features=3, hidden_features=32, output_features=2)
eccentric_quasi_circular_model = CNN_Model(input_features=3, hidden_features=32, output_features=2)


# --- MetaModel ---
class MetaModel(nn.Module):
    def __init__(self, lensed_unlensed_model, eccentric_quasi_circular_model):
        super(MetaModel, self).__init__()
        self.lens_model = lensed_unlensed_model
        self.eccentric_quasi_circular_model = eccentric_quasi_circular_model

    def forward(self, x):
        lens_output = self.lens_model(x)
        lens_probs = F.softmax(lens_output, dim=1)
        lens_pred = torch.argmax(lens_probs, dim=1)

        final_logits = []
        for i, pred in enumerate(lens_pred):
            if pred == 0:
                # Class 0: Lensed → one-hot [1, 0, 0]
                final_logits.append(torch.tensor([1.0, 0.0, 0.0], device=x.device))
            else:
                morph_out = self.eccentric_quasi_circular_model(x[i].unsqueeze(0))
                morph_probs = F.softmax(morph_out, dim=1).squeeze()
                # morph_probs[0] = eccentric → label 1
                # morph_probs[1] = quasi-circular → label 2
                class1_prob = morph_probs[0].item()
                class2_prob = morph_probs[1].item()
                final_logits.append(torch.tensor([0.0, class1_prob, class2_prob], device=x.device))

        return torch.stack(final_logits)


# --- Dataset ---
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


# --- Load Data ---
def load_data(seed, batch_size):
    train_list = glob.glob('./data/train/*.png')
    test_list = glob.glob('./data/test/*.png')
    labels = [path.split('/')[-1].split('_')[0] for path in train_list]

    train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, shuffle=True, random_state=seed)

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    train_data = GWDataset(train_list, transform)
    valid_data = GWDataset(valid_list, transform)
    test_data = GWDataset(test_list, transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), collate_fn=custom_collate_fn)

    return train_loader, valid_loader, test_loader, train_data, valid_data, transform


# --- Training Function ---
def train_submodel(model, train_loader, valid_loader, optimizer, criterion, device, epochs=10, label_map=None):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        for images, labels, _, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            if label_map:
                labels = torch.tensor([label_map[l.item()] for l in labels]).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = train_correct / total
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Train Accuracy: {acc:.4f}")

        model.eval()
        valid_correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _, _ in valid_loader:
                images, labels = images.to(device), labels.to(device)
                if label_map:
                    labels = torch.tensor([label_map[l.item()] for l in labels]).to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                valid_correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = valid_correct / total
        print(f"           Valid Accuracy: {val_acc:.4f}")


# --- Evaluate MetaModel ---
def evaluate_meta_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    wrong_preds = []

    with torch.no_grad():
        for images, labels, file_numbers, img_paths in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i, (p, t) in enumerate(zip(preds, labels)):
                if p != t:
                    wrong_preds.append((img_paths[i], int(p), int(t)))

    acc = correct / total
    print(f"[MetaModel] Test Accuracy: {acc:.4f}")
    return wrong_preds


# --- Main Execution ---
if __name__ == "__main__":
    seed = 42
    batch_size = 64
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader, train_data, valid_data, transform = load_data(seed, batch_size)

    # --- Train lens classifier ---
    lens_label_map = {0: 0, 1: 1, 2: 1}
    lens_optim = optim.Adam(lensed_unlensed_model.parameters(), lr=1e-3)
    lens_criterion = nn.CrossEntropyLoss()
    train_submodel(lensed_unlensed_model, train_loader, valid_loader, lens_optim, lens_criterion, device, epochs, label_map=lens_label_map)

    # --- Filter data for morphology classifier ---
    def filter_by_label(dataset, keep_labels):
        return [f for f in dataset.file_list if any(lbl in f for lbl in keep_labels)]

    train_list_unlensed = filter_by_label(train_data, ["eccentric", "unlensed"])
    valid_list_unlensed = filter_by_label(valid_data, ["eccentric", "unlensed"])

    train_loader_unlensed = DataLoader(GWDataset(train_list_unlensed, transform), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    valid_loader_unlensed = DataLoader(GWDataset(valid_list_unlensed, transform), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    morph_label_map = {1: 0, 2: 1}  # eccentric → 0, quasi → 1
    morph_optim = optim.Adam(eccentric_quasi_circular_model.parameters(), lr=1e-3)
    morph_criterion = nn.CrossEntropyLoss()

    train_submodel(eccentric_quasi_circular_model, train_loader_unlensed, valid_loader_unlensed, morph_optim, morph_criterion, device, epochs, label_map=morph_label_map)

    # --- Evaluate Meta Model ---
    meta_model = MetaModel(lensed_unlensed_model, eccentric_quasi_circular_model).to(device)
    wrong_preds = evaluate_meta_model(meta_model, test_loader, device)

    print("Evaluating on test set...")
    fpr, tpr, roc_auc, labels, predictions, _, misclassified = compute_roc_auc_with_misclassifications(
        model=meta_model, data_loader=test_loader, device=device
    )

    class_names = ['Lensed', 'Eccentric', 'Unlensed']

    plot_roc_curves(fpr, tpr, roc_auc, class_names, " (Test Set)")

    plot_confusion_matrix(labels, predictions, class_names, " (Test Set)")

    # --- Save models ---
    torch.save(lensed_unlensed_model.state_dict(), "lens_model.pth")
    torch.save(eccentric_quasi_circular_model.state_dict(), "morph_model.pth")

    # --- Load models (if needed) ---
    # lensed_unlensed_model.load_state_dict(torch.load("lens_model.pth"))
    # eccentric_quasi_circular_model.load_state_dict(torch.load("morph_model.pth"))
