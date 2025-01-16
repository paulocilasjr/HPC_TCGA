import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler

dataset_array = ["image_embeddings_mil_13.csv", "image_embeddings_mil_14.csv"]


# Updated Paths and Constants
OUTPUT_CSV = "image_embeddings_mil_6.csv"
IMG_SIZE = (224, 224)  # Image size for Ludwig
EMBEDDING_SIZE = 1000  # Fixed size for all embeddings, you can adjust this based on the model

# Load Data
def LoadDataset (dataset_type):
    if dataset_type == "all_data":
        dataset = pd.read_csv("./../er_status_all_data.csv")
    if dataset_type == "no_white":
        dataset = pd.read_csv("./../er_status_no_white.csv")

def UniqueSamples (load_data):

    # Assign Split Randomly at the Sample Level
    np.random.seed(42)  # Set a seed for reproducibility
    unique_samples = load_data["sample"].unique()
    num_samples = len(unique_samples)
    print(f"this is the length of unique samples: {num_samples}", flush=True)
    return unique_samples, num_samples 

def SplitData (unique_samples, num_samples, data):
    # Generate random splits for samples
    train_count = int(0.7 * num_samples)
    val_count = int(0.1 * num_samples)

    # Randomly shuffle the samples
    shuffled_samples = np.random.permutation(unique_samples)

    # Assign splits
    train_samples = shuffled_samples[:train_count]
    val_samples = shuffled_samples[train_count:train_count + val_count]
    test_samples = shuffled_samples[train_count + val_count:]

    # Map samples to splits
    sample_to_split = {sample: 0 for sample in train_samples}  # Train: 0
    sample_to_split.update({sample: 1 for sample in val_samples})  # Validation: 1
    sample_to_split.update({sample: 2 for sample in test_samples})  # Test: 2

    # Add split column to the DataFrame
    data["split"] = data["sample"].map(sample_to_split)

    return data, train_samples, val_samples, test_samples 

def ModelExtractor(Model_name=None):
    # ResNet Model Setup for Embedding Extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(pretrained=False).to(device)
    weights_path = "./../resnet50-0676ba61.pth"
    resnet.load_state_dict(torch.load(weights_path, map_location=device))
    resnet.eval()   
    
    #return resnet

def TransformInitiation():
    # Preprocessing Transformations
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform

# Function to Create ResNet Embeddings from Image
def extract_resnet_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(image)

    # Remove the classification head (last layer) and use the features from earlier layers
    embedding = embedding.squeeze().cpu().numpy()  # Convert tensor to NumPy array

    # Ensure the embedding has a consistent size
    if embedding.size != EMBEDDING_SIZE:
        print(f"Warning: embedding size {embedding.size} is not {EMBEDDING_SIZE}. Padding or truncating.")
        # Pad or truncate the embedding to the fixed size
        embedding = np.pad(embedding, (0, EMBEDDING_SIZE - embedding.size), 'constant') if embedding.size < EMBEDDING_SIZE else embedding[:EMBEDDING_SIZE]

    return embedding

# Function to Aggregate Embeddings Using Max Pooling
def aggregate_embeddings_with_max_pooling(embeddings):
    print("inside aggregate_embeddings_with_max_pooling function", flush=True)
    # Apply max pooling across the embeddings (along the first axis)
    max_pooled_embedding = np.max(embeddings, axis=0)
    return max_pooled_embedding

# Function to Save Aggregated Embedding as String
def convert_embedding_to_string(embedding):
    embedding = np.nan_to_num(embedding)  # Replace NaNs with zero
    return " ".join(map(str, embedding))

# Function to Balance Bags
def balance_bags(bags):
    bags_0 = [bag for bag in bags if bag["bag_label"] == 0]
    bags_1 = [bag for bag in bags if bag["bag_label"] == 1]
    min_count = min(len(bags_0), len(bags_1))
    balanced_bags_0 = np.random.choice(bags_0, min_count, replace=False).tolist()
    balanced_bags_1 = np.random.choice(bags_1, min_count, replace=False).tolist()
    balanced_bags = balanced_bags_0 + balanced_bags_1
    np.random.shuffle(balanced_bags)
    return balanced_bags

#####
# Function to Create Bags from Split Data
def create_bags_from_split(split_data, split):
    print("inside create bags from split function", flush=True)
    bags = []
    images = []

    # Collect all images for the current split
    for _, row in split_data.iterrows():
        image_path = row["image_path"]
        embedding = extract_resnet_embedding(image_path)
        images.append((embedding, row["er_status_by_ihc"], row["sample"]))

    # Shuffle images to ensure randomness
    np.random.shuffle(images)
    num_images = len(images)
    i = 0

    while i < num_images:
        # Determine random bag size between 3 and 7
        bag_size = np.random.randint(3, 8)

        # Check if remaining images are fewer than the bag size
        if i + bag_size > num_images:
            # Use all remaining images to form the final bag
            bag_images = images[i:]
            i = num_images  # End the loop
        else:
            # Create a bag with the selected size
            bag_images = images[i:i + bag_size]
            i += bag_size

        bag_image_embeddings = [x[0] for x in bag_images]
        bag_labels = [x[1] for x in bag_images]
        bag_samples = [x[2] for x in bag_images]

        # Aggregate images into a single embedding using max pooling
        aggregated_embedding = aggregate_embeddings_with_max_pooling(np.array(bag_image_embeddings))

        # Convert the aggregated embedding to a string for the DataFrame
        embedding_string = convert_embedding_to_string(aggregated_embedding)

        # Bag-level label (if any image has label 1, the bag label is 1)
        bag_label = int(any(np.array(bag_labels) == 1))

        # Add the bag information to the records
        bags.append({
            "embedding": embedding_string,
            "bag_label": bag_label,
            "split": split,
        })

    return bags


# Create Bags for All Splits
for dataset in ["all_data", "no_white"]:
    data = LoadDataset(dataset)
    unique_samples, num_samples = UniqueSamples(data)
    data, train_samples, val_samples, test_samples = SplitData (unique_samples, num_samples, data)
    resnet_model = ModelExtractor()
    transform = TransformInitiation()

    all_bags = []
    for split, split_samples in [(0, train_samples), (1, val_samples), (2, test_samples)]:
        print(f"create bags for: {split}", flush=True)
        split_data = data[data["sample"].isin(split_samples)]
        split_bags = create_bags_from_split(split_data, split)
        balanced_split_bags = balance_bags(split_bags)
        all_bags.extend(balanced_split_bags)

    # Save Metadata to CSV
    metadata_df = pd.DataFrame(all_bags)
    metadata_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved balanced metadata to {OUTPUT_CSV}")
