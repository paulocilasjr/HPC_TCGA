import pandas as pd
import numpy as np
from time import sleep


def create_bags_from_split(split_data, split, repeats):
    print("inside create bags from split function", flush=True)
    bags = []
    images = []
    bag_set = set()

    # Collect all images for the current split
    for _, row in split_data.iterrows():
        image_path = row["image_path"]
        embedding = extract_resnet_embedding(image_path)
        images.append((embedding, row["er_status_by_ihc"], row["sample"]))
    
    for _ in range(repeats):
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
                    num_1_tiles = np.random.randint(1, bag_size + 1)
                    print(f"Random number is {num_1_tiles}")
                else:
                    num_1_tiles = bag_size
                
                selected_images_1 = images_1[:num_1_tiles]
                images_1 = images_1[num_1_tiles:]  # Remove selected images
                print(f"Num of tiles 1 taken: {len(selected_images_1)}")
                    
                # Fill the rest of the bag with images_0
                num_0_tiles = min(bag_size - num_1_tiles, len(images_0))
                selected_images_0 = images_0[:num_0_tiles]
                print(f"Num of tiles 0 taken: {len(selected_images_0)}")
                images_0 = images_0[num_0_tiles:]
                    
                make_bag_1 = False
                # Combine selected images to form the bag
                bag_images = selected_images_1 + selected_images_0
                if len(bag_images) != bag_size:
                    print("can still fit images")
                    num_extra_tiles = bag_size - len(bag_images)
                    print(f"fit {num_extra_tiles}")
                    selected_images_extra = images_1[:num_extra_tiles]
                    print(f"Num of tiles 1 taken: {len(selected_images_extra)}")
                    images_1 = images_1[num_extra_tiles:]
                        
                    # Combine selected images to form the bag
                    bag_images += selected_images_extra

            elif not make_bag_1 and len(images_0) > 0:
                print("TURN = 0")
                num_0_tiles = bag_size
                selected_images_0 = images_0[:num_0_tiles]
                print(f"Num of tiles 0 taken: {num_0_tiles}")
                images_0 = images_0[num_0_tiles:]
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

                bag_image_embeddings_tuple = tuple(map(tuple, bag_image_embeddings)) 
                bag_samples_tuple = tuple(bag_samples)
                bag_key = (bag_image_embeddings_tuple, len(bag_images), bag_samples_tuple)

                if bag_key not in bag_set:
                    bag_set.add(bag_key)

                    # Add the bag information to the records
                    bags.append({
                        "embedding": bag_image_embeddings,
                        "bag_label": bag_label,
                        "split": split,
                        "bag_size": len(bag_images),
                        "bag_samples": bag_samples
                    })
                else:
                    print("A bag was created twice", flush = True)
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

test = create_bags_from_split(mock_split_data, 0, 3)