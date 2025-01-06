import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler

# Updated Paths and Constants
OUTPUT_DIR = "/share/lab_goecks/TCGA_deep_learning/now_h_experiment/embeddings_processing/embeddings"
OUTPUT_CSV = "image_embeddings_mil.csv"
IMG_SIZE = (224, 224)  # Image size for Ludwig

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Data
data = pd.read_csv("./../er_status_no_white.csv")

# Assign Split Randomly at the Sample Level
np.random.seed(42)  # Set a seed for reproducibility
unique_samples = data["sample"].unique()
num_samples = len(unique_samples)
print(f"this is the length of unique samples: {num_samples}", flush=True)

# Generate random splits for samples
train_count = int(0.7 * num_samples)
val_count = int(0.1 * num_samples)

# Randomly shuffle the samples
shuffled_samples = np.random.permutation(unique_samples)

# Assign splits
train_samples = shuffled_samples[:train_count]
val_samples = shuffled_samples[train_count:train_count + val_count]
test_samples = shuffled_samples[train_count + val_count:]
print(f'assign splits: train_samples = {train_samples}, val_samples = {val_samples}, test_samples = {test_samples}', flush=True)

# Map samples to splits
sample_to_split = {sample: 0 for sample in train_samples}  # Train: 0
sample_to_split.update({sample: 1 for sample in val_samples})  # Validation: 1
sample_to_split.update({sample: 2 for sample in test_samples})  # Test: 2
print("finished map samples to splits", flush=True)

# Add split column to the DataFrame
data["split"] = data["sample"].map(sample_to_split)
print(f'data[split] = {data["split"]}', flush=True)

# ResNet Model Setup for Embedding Extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=False).to(device)
weights_path = "./../resnet50-0676ba61.pth"
resnet.load_state_dict(torch.load(weights_path, map_location=device))
resnet.eval()

# Preprocessing Transformations
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to Create ResNet Embeddings from Image
def extract_resnet_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = resnet(image)
    
    # Remove the classification head (last layer) and use the features from earlier layers
    embedding = embedding.squeeze().cpu().numpy()  # Convert tensor to NumPy array
    return embedding

# Function to Aggregate Embeddings Using Max Pooling
def aggregate_embeddings_with_max_pooling(embeddings):
    """
    Aggregates embeddings using max pooling.
    
    Args:
    embeddings (np.array): Embeddings array of shape (num_images, embedding_dim)
    
    Returns:
    np.array: The aggregated embedding.
    """
    print("inside aggregate_embeddings_with_max_pooling function", flush=True)
    
    # Apply max pooling across the embeddings (along the first axis)
    max_pooled_embedding = np.max(embeddings, axis=0)
    return max_pooled_embedding

# Function to Save Aggregated Embedding as Vector for Ludwig
def save_embedding_as_vector(embedding, output_path):
    """
    Saves the aggregated embedding as a whitespace-separated vector
    suitable for Ludwig, where each number is a part of the vector.

    Args:
    embedding (np.array): The aggregated embedding, shape (embedding_dim,).
    output_path (str): The path where the embedding vector will be saved.
    """
    # Ensure no NaN values in the embedding
    embedding = np.nan_to_num(embedding)  # Replace NaNs with zero

    # Save the embedding as a whitespace-separated vector
    np.savetxt(output_path, embedding, delimiter=" ", fmt="%.6f")

# Function to Create Bags from Split Data
def create_bags_from_split(split_data, split):
    print("inside create bags from split function", flush=True)
    bags = []
    images = []

    # Collect all images for the current split
    print(f'collect all images for the current split', flush=True)
    for _, row in split_data.iterrows():
        image_path = row["image_path"]
        embedding = extract_resnet_embedding(image_path)
        images.append((embedding, row["er_status_by_ihc"], row["sample"]))

    # Create bags of 5 images
    num_images = len(images)
    np.random.shuffle(images)  # Shuffle images to ensure randomness

    # Create full-sized bags (5 images per bag)
    print(f'create full-sized bags')
    for i in range(0, num_images, 5):
        bag_images = images[i:i+5]
        bag_image_embeddings = [x[0] for x in bag_images]
        bag_labels = [x[1] for x in bag_images]
        bag_samples = [x[2] for x in bag_images]

        # Aggregate images into a single embedding using max pooling
        aggregated_embedding = aggregate_embeddings_with_max_pooling(np.array(bag_image_embeddings))

        # Save the aggregated embedding as a vector (instead of an image)
        vector_path = os.path.join(OUTPUT_DIR, f"bag_{len(bags)}.txt")
        save_embedding_as_vector(aggregated_embedding, vector_path)

        # Bag-level label (if any image has label 1, the bag label is 1)
        bag_label = int(any(np.array(bag_labels) == 1))

        # Use the split from the first sample in the bag (assuming all images in the bag are from the same split)
        split = sample_to_split[bag_samples[0]]

        # Add the bag information to the records
        bags.append({
            "vector_path": vector_path,
            "bag_label": bag_label,
            "split": split,
        })
    print("done with bags", flush=True)
    return bags

# Create Bags for All Splits
all_bags = []

# Create bags for train, validation, and test splits
for split, split_samples in [(0, train_samples), (1, val_samples), (2, test_samples)]:
    print(f"create bags for: {split}", flush=True)
    # Get data corresponding to the current split
    split_data = data[data["sample"].isin(split_samples)]
    split_bags = create_bags_from_split(split_data, split)
    all_bags.extend(split_bags)

# Save Metadata to CSV
metadata_df = pd.DataFrame(all_bags)
metadata_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved metadata to {OUTPUT_CSV}")

