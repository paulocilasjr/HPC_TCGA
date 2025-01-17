import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix

# Paths and constants
CSV_FILE = "er_status_no_white.csv"
OUTPUT_METRICS_FILE = "pytorch_h_3.txt"
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset
class ERStatusDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['er_status_by_ihc']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# Load the dataset
data = pd.read_csv(CSV_FILE)

# Split the data into train, val, and test (70%, 10%, 20%)
train_data, test_data = train_test_split(data, test_size=0.30, stratify=data['er_status_by_ihc'], random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.67, stratify=test_data['er_status_by_ihc'], random_state=42)

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

# Define the Model
model = models.resnet50(pretrained=True)  # Use ResNet-50 instead of ResNet-18
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
model = model.to(DEVICE)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation Functions
def calculate_metrics(outputs, labels):
    predictions = (outputs > 0.5).float()
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    roc_auc = roc_auc_score(labels, outputs)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
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
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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
    # Training
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
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
test_metrics = evaluate(test_loader, model)
write_metrics_to_file(OUTPUT_METRICS_FILE, EPOCHS, "Test", test_metrics)

print("Training complete. Metrics saved to", OUTPUT_METRICS_FILE)

