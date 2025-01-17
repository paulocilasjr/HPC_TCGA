import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
import hashlib

# Paths and constants
CSV_FILE = "er_status_no_white.csv"
OUTPUT_METRICS_FILE = "pytorch_h_mil.txt"
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset for MIL
class ERStatusDataset(Dataset):
    def __init__(self, data, transform=None, bag_size=5):
        self.data = data
        self.transform = transform
        self.bag_size = bag_size

    def __len__(self):
        return len(self.data) // self.bag_size  # Number of bags

    def __getitem__(self, idx):
        """
        Creating a bag of instances
        """
        bag_start_idx = idx * self.bag_size
        bag_end_idx = (idx + 1) * self.bag_size
        bag_data = self.data.iloc[bag_start_idx:bag_end_idx]

        images = []
        labels = []

        # Loading all images in the bag and their labels
        for i, row in bag_data.iterrows():
            img_path = row['image_path']
            label = row['er_status_by_ihc']
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
            labels.append(label)

        # Stack all images into a tensor (shape: [bag_size, channels, height, width])
        images = torch.stack(images)

        labels = torch.tensor(labels, dtype=torch.float32)

        # Bag-level label (if any instance has label 1, the bag level will be 1)
        bag_label = torch.tensor(int(any(labels == 1)), dtype=torch.float32)

        return images, bag_label  # Return images for the bag and the bag label

# Load the dataset
data = pd.read_csv(CSV_FILE)

# Normalize hash values to a range between 0 and 1
def normalized_hash_function(sample_id):
    return int(hashlib.sha256(sample_id.encode('utf-8')).hexdigest(), 16) % (10**8) / (10**8)

# Deterine split indices based on normalized hash values
train_indices = []
val_indices = []
test_indices = []

for index, row in data.iterrows():
    sample_hash = normalized_hash_function(row['sample'])
    if sample_hash < 0.7:
        train_indices.append(index)
    elif sample_hash < 0.8:
        val_indices.append(index)
    else:
        test_indices.append(index)

# Create splits
train_data = data.loc[train_indices]
val_data = data.loc[val_indices]
test_data = data.loc[test_indices]

# Ensure there are no empty splits
if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
    raise ValueError("One of the splits has no data")

# Transforms 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets and Dataloaders
train_dataset = ERStatusDataset(data=train_data, transform=transform)
val_dataset = ERStatusDataset(data=val_data, transform=transform)
test_dataset = ERStatusDataset(data=test_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Debugging Information
print(f"Training samples: {len(train_data)}", flush=True)
print(f"Validation samples: {len(val_data)}", flush=True)
print(f"Testing samples: {len(test_data)}", flush=True)

# Define the model
model = models.resnet50(weights=None)
weights_path = "resnet50-0676ba61.pth"
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
model = model.to(DEVICE)

# Loss and Optmizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation functions
def calculate_metrics(outputs, labels):
    predictions = (outputs > 0.5).float()
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    roc_auc = roc_auc_score(labels, outputs)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    print(f"confusion matrix: tn = {tn}, fp = {fp}, fn = {fn}, tp = {tp}")
    specificity = tn / (tn + fp)
    return acc, precision, recall, roc_auc, specificity

def evaluate(loader, model):
    model.eval()
    all_outputs = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Reshape images: Combine batch size and bag size dimensions
            batch_size, bag_size, channels, height, width = images.shape
            images = images.view(-1, channels, height, width)  # Flatten batch and bag dimensions

            # Forward pass
            outputs = model(images)

            # Reshape outputs back to [batch_size, bag_size]
            outputs = outputs.view(batch_size, bag_size)

            # Aggregate predictions per bag (e.g., mean across bag size)
            bag_outputs = outputs.mean(dim=1)  # Aggregate predictions over the bag dimension

            # Compute loss
            loss = criterion(bag_outputs.squeeze(), labels)
            total_loss += loss.item()

            # Store outputs and labels for metric calculations
            all_outputs.extend(bag_outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    acc, precision, recall, roc_auc, specificity = calculate_metrics(
        torch.tensor(all_outputs), torch.tensor(all_labels)
    )
    return total_loss / len(loader), acc, precision, recall, roc_auc, specificity

def write_metrics_to_file(file_path, epoch, phase, metrics):
    with open(file_path, "a") as f:
        f.write(
            f"Epoch {epoch} - {phase}: "
            f"Loss: {metrics[0]:.4f}, Accuracy: {metrics[1]:.4f}, Precision: {metrics[2]:.4f}, "
            f"Recall: {metrics[3]:.4f}, ROC-AUC: {metrics[4]:.4f}, Specificity: {metrics[5]:.4f}\n"
        )

# Training Loop
for epoch in range(EPOCHS):
    print(f"epoch: {epoch}", flush=True)
    # Training
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # Reshape images: Combine batch size and bag size dimensions
        batch_size, bag_size, channels, height, width = images.shape
        images = images.view(-1, channels, height, width)  # Flatten batch and bag sizes

        # Get predictions from the model
        outputs = model(images)

        # Reshape the outputs back to the original batch size and bag size
        outputs = outputs.view(batch_size, bag_size, -1)  # Shape: [batch_size, bag_size, output_size]

        # Aggregate predictions per bag (mean, sum, or max)
        bag_outputs = outputs.mean(dim=1)  # Aggregating over the bag size (5 images per bag)

        # Compute loss between bag-level predictions and labels
        loss = criterion(bag_outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_metrics = evaluate(train_loader, model)
    val_metrics = evaluate(val_loader, model)

    # Write metrics to file
    write_metrics_to_file(OUTPUT_METRICS_FILE, epoch + 1, "Train", (train_loss, *train_metrics))
    write_metrics_to_file(OUTPUT_METRICS_FILE, epoch + 1, "Validation", val_metrics)

# Test Evaluation
print("testing begin", flush=True)
test_metrics = evaluate(test_loader, model)
write_metrics_to_file(OUTPUT_METRICS_FILE, EPOCHS, "Test", test_metrics)

print("Training complete. Metrics saved to", OUTPUT_METRICS_FILE)

