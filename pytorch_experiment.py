import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
from PIL import Image
import hashlib

print("Starting the script", flush=True)

# Custom Dataset for image data
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the load_image function
def load_image(image_path):
    """Load an image from a file path."""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}", flush=True)
        raise

# Define the transformations and augmentations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3)),
    transforms.GaussianBlur(kernel_size=3),
    transforms.Resize((2500, 2500)),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)], p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading dataset", flush=True)

# Load dataset from CSV
data_file = "er_status_no_white.csv"
df = pd.read_csv(data_file)

# Extract the required columns
image_paths = df['image_path'].tolist()
labels = df['er_status_by_ihc'].tolist()
samples = df['sample'].tolist()

print(f"Loaded {len(image_paths)} images with labels.", flush=True)

# Create a hash-based split function
def hash_split(samples, train_ratio=0.7, val_ratio=0.1):
    train_indices, val_indices, test_indices = [], [], []
    for idx, sample in enumerate(samples):
        sample_hash = int(hashlib.md5(sample.encode('utf-8')).hexdigest(), 16)
        split_value = sample_hash % 100

        if split_value < train_ratio * 100:
            train_indices.append(idx)
        elif split_value < (train_ratio + val_ratio) * 100:
            val_indices.append(idx)
        else:
            test_indices.append(idx)
    return train_indices, val_indices, test_indices

print("Splitting dataset", flush=True)

# Apply hash-based splitting
train_indices, val_indices, test_indices = hash_split(samples)

# Create datasets for each split
dataset = CustomImageDataset(image_paths, labels, transform=augmentation_transforms)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

print("Defining model", flush=True)

# Define model
model = models.resnet50()
weights_path = "resnet50-0676ba61.pth"
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)

device = torch.device('cpu')
model = model.to(device)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Metrics function
def compute_metrics(labels, preds, loss):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    roc_auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else np.nan
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp)
    return {
        "Accuracy": accuracy,
        "Loss": loss,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc,
        "Specificity": specificity
    }

print("Starting training", flush=True)

# Training loop
epochs = 50
early_stop_patience = 10
best_val_acc = 0
early_stop_counter = 0
metrics_log = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}", flush=True)
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (outputs.squeeze() > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_metrics = compute_metrics(all_labels, all_preds, train_loss)
    print(f"Train Metrics: {train_metrics}", flush=True)

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
            preds = (outputs.squeeze() > 0.5).long()
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_metrics = compute_metrics(val_labels, val_preds, val_loss)
    print(f"Validation Metrics: {val_metrics}", flush=True)

    metrics_log.append({
        "Epoch": epoch + 1,
        "Train": train_metrics,
        "Validation": val_metrics
    })

    scheduler.step(val_metrics["Accuracy"])

    if val_metrics["Accuracy"] > best_val_acc:
        best_val_acc = val_metrics["Accuracy"]
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved.", flush=True)
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.", flush=True)
            break

print("Evaluating on test data", flush=True)

# Test evaluation
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_preds, test_labels = [], []
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).float()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        test_loss += loss.item()
        preds = (outputs.squeeze() > 0.5).long()
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader)
test_metrics = compute_metrics(test_labels, test_preds, test_loss)
print(f"Test Metrics: {test_metrics}", flush=True)

# Save metrics to a file
print("Saving metrics to a file", flush=True)
output_file = "pytorch_h_outputs.txt"
with open(output_file, "w") as f:
    f.write("Training, Validation, and Test Metrics:\n\n")
    for entry in metrics_log:
        f.write(f"Epoch {entry['Epoch']}:\n")
        f.write(f"Train Metrics: {entry['Train']}\n")
        f.write(f"Validation Metrics: {entry['Validation']}\n")
    f.write(f"Test Metrics: {test_metrics}\n")

print("Script completed successfully", flush=True)
