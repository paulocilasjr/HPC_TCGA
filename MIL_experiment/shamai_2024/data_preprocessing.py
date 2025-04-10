import os
import numpy as np
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image

# Constants
IMG_SIZE = (224, 224)  # Image size for model input
EMBEDDING_SIZE = 1000  # Fixed embedding size

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the ResNet model
def initialize_resnet(weights_path=None):
    resnet = models.resnet50(pretrained=False).to(device)
    if weights_path:
        resnet.load_state_dict(torch.load(weights_path, map_location=device))
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove classification head
    resnet.eval()
    return resnet

# Preprocessing Transformations
def initialize_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Extract ResNet Embedding
def extract_resnet_embedding(image_path, transform, resnet):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(image).squeeze().cpu().numpy()

    # Ensure the embedding has a consistent size
    if embedding.size != EMBEDDING_SIZE:
        embedding = np.pad(embedding, (0, EMBEDDING_SIZE - embedding.size), 'constant') \
            if embedding.size < EMBEDDING_SIZE else embedding[:EMBEDDING_SIZE]
    return embedding

# Main Function
def main(image_paths_file, output_csv, weights_path):
    # Load image paths
    image_paths = pd.read_csv(image_paths_file, header=None).iloc[:, 0].tolist()

    # Initialize model and transform
    resnet = initialize_resnet(weights_path)
    transform = initialize_transform()

    # Extract embeddings
    embeddings = []
    for image_path in image_paths:
        embedding = extract_resnet_embedding(image_path, transform, resnet)
        embeddings.append({"image_path": image_path, "embedding": " ".join(map(str, embedding))})

    # Save to CSV
    pd.DataFrame(embeddings).to_csv(output_csv, index=False)
    print(f"Embeddings saved to {output_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract image embeddings using ResNet.")
    parser.add_argument("image_paths_file", help="Path to a text file containing image paths (one per line).")
    parser.add_argument("output_csv", help="Path to save the extracted embeddings CSV.")
    parser.add_argument("--weights_path", help="Path to ResNet weights file.", required=True)

    args = parser.parse_args()
    main(args.image_paths_file, args.output_csv, args.weights_path)
