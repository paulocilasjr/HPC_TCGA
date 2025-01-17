import pandas as pd
import numpy as np
from time import sleep


def create_bags_from_split(split_data, split):
    print("inside create bags from split function", flush=True)
    bags = []
    images = []

    # Collect all images for the current split
    for _, row in split_data.iterrows():
        image_path = row["image_path"]
        embedding = extract_resnet_embedding(image_path)
        images.append((embedding, row["er_status_by_ihc"], row["sample"]))

    images_0 = [image for image in images if image[1] == 0]
    images_1 = [image for image in images if image[1] == 1]

    # Ensure randomness by shuffling both groups
    np.random.shuffle(images_0)
    np.random.shuffle(images_1)

    make_bag_1 = True

    # Continue until all images are used
    while len(images_0) + len(images_1) > 0:
        print (f"length of remanining images 1 = {len(images_1)}")
        print (f"length of remaining images 0 = {len(images_0)}")
        
        sleep(5)

        # Determine random bag size between 3 and 7
        bag_size = np.random.randint(3, 8)
        print(f"bag size={bag_size}")
        sleep(5)
        if make_bag_1 and len(images_1) > 0:
            print("TURN = 1")    
            if len(images_0) > 0:
                num_1_tiles = np.random.randint(1, bag_size)
            else:
                num_1_tiles = bag_size
                
            try:
                selected_images_1 = images_1[:num_1_tiles]
                images_1 = images_1[num_1_tiles:]  # Remove selected images
                print(f"Num of tiles 1 taken: {len(selected_images_1)}")
            except:
                print(f"#Except - Num of tiles 1 taken: {len(selected_images_1)}")
                selected_images_1 = images_1
                images_1 = []
                
            
            # Fill the rest of the bag with images_0
            num_0_tiles = min(bag_size - num_1_tiles, len(images_0))
            try:    
                selected_images_0 = images_0[:num_0_tiles]
                print(f"Num of tiles 0 taken: {len(selected_images_0)}")
                images_0 = images_0[num_0_tiles:]
            except:
                print(f"#Except - num of tiles 0 taken {len(selected_images_0)}")
                selected_images_0 = images_0
                images_0 = []
                
            
            make_bag_1 = False
            # Combine selected images to form the bag
            bag_images = selected_images_1 + selected_images_0
            if len(bag_images) != bag_size:
                print("can still fit images")
                num_extra_tiles = bag_size - len(bag_images)
                print(f"fit {num_extra_tiles}")
                try:    
                    selected_images_extra = images_1[:num_extra_tiles]
                    print(f"Num of tiles 1 taken: {len(selected_images_extra)}")
                    images_1 = images_1[num_extra_tiles:]
                except:
                    print(f"#Except - num of tiles 1 taken {len(selected_images_1)}")
                    selected_images_1 = images_1
                    images_1 = []
                    
                # Combine selected images to form the bag
                bag_images += selected_images_extra

        elif not make_bag_1 and len(images_0) > 0:
            print("TURN = 0")
            try:
                num_0_tiles = bag_size
                selected_images_0 = images_0[:num_0_tiles]
                print(f"Num of tiles 0 taken: {num_0_tiles}")
                images_0 = images_0[num_0_tiles:]
            except:
                selected_images_0 = images_0
                images_0 = []
                print(f"#Except - num of tiles 0 taken {len(selected_images_0)}")
            selected_images_1 = []
            make_bag_1 = True
            bag_images = selected_images_0
        
        else:
            print("it is going to pass")
            make_bag_1 = not make_bag_1
            bag_images = [] 

        print (f"bag created length: {len(bag_images)}")
        sleep(5)

        
        if len(bag_images) > 0:
            # Extract data for bag-level representation
            bag_image_embeddings = [x[0] for x in bag_images]
            bag_labels = [x[1] for x in bag_images]
            bag_samples = [x[2] for x in bag_images]

            

            # Bag-level label (if any image has label 1, the bag label is 1)
            bag_label = int(any(np.array(bag_labels) == 1))
            
            # Add the bag information to the records
            bags.append({
                "embedding": bag_image_embeddings,
                "bag_label": bag_label,
                "split": split,
                "bag_size": len(bag_images),
            })
        
    return bags

# Mock function to simulate embedding extraction (for testing purposes)
def extract_resnet_embedding(image_path):
    return np.random.rand(512)  # Random embedding vector of size 512

# Create a mock input DataFrame with 5 elements
mock_split_data = pd.DataFrame({
    "image_path": [f"image_{i}.jpg" for i in range(20)],
    "er_status_by_ihc": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # Labels: 0 or 1
    "sample": [f"sample_{i}" for i in range(20)]    
})

test = create_bags_from_split(mock_split_data, 0)