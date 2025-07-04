from __future__ import print_function

import glob
import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

from vit_pytorch.efficient import ViT
from vit_pytorch.cvt import CvT

model_selection = str(sys.argv[1])

# sys.stdout = open("log.out", "w")
# sys.stderr = open("error.err", "w")

batch_size = 512
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42

print("Code is running")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)


# device = 'cuda' if torch.cuda.is_available() else "cpu"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"The device is {device}")


train_dir = './data/train'
test_dir = './data/test'


train_list = glob.glob(os.path.join(train_dir,'*.png'))
test_list = glob.glob(os.path.join(test_dir, '*.png'))



print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")




labels = [path.split('/')[-1].split('_')[0] for path in train_list]



train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          shuffle=True,
                                          random_state=seed)




print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")




train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)




class GWDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        
        # Apply transformations if provided
        img_transformed = self.transform(img)
        img_transformed = img_transformed[:3, :, :]  # Ensure correct format for channels

        # Determine the label based on filename
        label_str = img_path.split("/")[-1].split("_")[0]
        if label_str == "eccentric":
            label = 1
        elif label_str == "unlensed":
            label = 2
        else:
            label = 0 

        return img_transformed, label





train_data = GWDataset(train_list, transform=train_transforms)
valid_data = GWDataset(valid_list, transform=test_transforms)
test_data = GWDataset(test_list, transform=test_transforms)



train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)




print(f"Train Dataset Length: {len(train_data)}, Test Dataloader Length: {len(train_loader)}")

print(f"Validation Dataset Length: {len(valid_data)}, Validation Dataloader Length: {len(valid_loader)}")



if model_selection == 'CvT':
    model =  CvT(num_classes = 3,
                s1_emb_dim = 64,        # stage 1 - dimension
                s1_emb_kernel = 7,      # stage 1 - conv kernel
                s1_emb_stride = 4,      # stage 1 - conv stride
                s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
                s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
                s1_heads = 1,           # stage 1 - heads
                s1_depth = 1,           # stage 1 - depth
                s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
                s2_emb_dim = 192,       # stage 2 - (same as above)
                s2_emb_kernel = 3,
                s2_emb_stride = 2,
                s2_proj_kernel = 3,
                s2_kv_proj_stride = 2,
                s2_heads = 3,
                s2_depth = 2,
                s2_mlp_mult = 4,
                s3_emb_dim = 384,       # stage 3 - (same as above)
                s3_emb_kernel = 3,
                s3_emb_stride = 2,
                s3_proj_kernel = 3,
                s3_kv_proj_stride = 2,
                s3_heads = 4,
                s3_depth = 10,
                s3_mlp_mult = 4,
                dropout = 0.3).to(device)
else:
    efficient_transformer = Linformer(
        dim=128,
        seq_len=49+1,  # 7x7 patches + 1 cls-token
        depth=12,
        heads=8,
        k=64,
        dropout=0.2,)

    model = ViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=3,
        transformer=efficient_transformer,
        channels=3,
    ).to(device)


################################################################################################

file_path = "./model.pth"

if os.path.exists(file_path):
    print("Pre-trained model found. Training continued...")
    model_path = 'model.pth'
    model.load_state_dict(torch.load(model_path))
else:
    print("Pre-trained model not found. Training from beginning...")

################################################################################################

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    
    train_losses.append(epoch_loss.item())
    train_accuracies.append(epoch_accuracy.item())
    
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)
    
    val_losses.append(epoch_val_loss.item())
    val_accuracies.append(epoch_val_accuracy.item())
    
    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")


plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.savefig('Loss_Curve.png')


with torch.no_grad():
    model.eval()
    all_preds = []
    all_labels = []

    # Inference loop
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        
        preds = torch.argmax(output, dim=1).cpu().numpy()
        labels = label.cpu().numpy()

        # Map numeric labels to class names
        class_mapping = {0: 'Lensed', 1: 'Eccentric', 2: 'Unlensed'}
        preds = [class_mapping[p] for p in preds]
        labels = [class_mapping[l] for l in labels]

        all_preds.extend(preds)
        all_labels.extend(labels)

    # Compute the confusion matrix
    cm_labels = ['Lensed', 'Eccentric', 'Unlensed']
    cm = confusion_matrix(all_labels, all_preds, labels=cm_labels)
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0) * 100

    # Annotate the confusion matrix with counts and percentages
    annotations = np.array([[f"{count}\n({percent:.2f}%)" for count, percent in zip(row, row_perc)] 
                             for row, row_perc in zip(cm, cm_percentage)])

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix: Lensed, Eccentric, Unlensed')
    plt.savefig("Confusion_matrix.png")

    # # Compute ROC and AUC for each class (One-vs-Rest approach)
    # all_labels_numeric = np.array([list(class_mapping.keys())[list(class_mapping.values()).index(label)] for label in all_labels])
    # all_preds_numeric = np.array([list(class_mapping.keys())[list(class_mapping.values()).index(pred)] for pred in all_preds])

    # # Convert multi-class labels to One-vs-Rest format for ROC
    # binarizer = LabelBinarizer()
    # all_labels_binarized = binarizer.fit_transform(all_labels_numeric)
    # all_preds_binarized = binarizer.transform(all_preds_numeric)

    # plt.figure(figsize=(8, 6))
    # for i, class_name in enumerate(cm_labels):
    #     fpr, tpr, _ = roc_curve(all_labels_binarized[:, i], all_preds_binarized[:, i])
    #     roc_auc = auc(fpr, tpr)
    #     plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

    # plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve (Multiclass)')
    # plt.legend(loc='lower right')
    # plt.savefig("ROC_AUC_curve.png")

torch.save(model.state_dict(), f"model.pth")

# sys.stdout.close()
# sys.stderr.close()
