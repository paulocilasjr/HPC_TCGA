import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_score

# Split dataset based on hash
def split_dataset_by_hash(data, split_ratios=[0.7, 0.1, 0.2], seed=42):
    # Define the hash function
    def hash_fn(value):
        return hash(value) % 1000  # This creates a consistent hash value

    # Create a new column in the dataframe to store hash values
    data['hash'] = data['image_path'].apply(hash_fn)

    # Create train, validation, test splits based on hash
    train_data = data[data['hash'] % 10 < split_ratios[0] * 10]
    val_data = data[data['hash'] % 10 < (split_ratios[0] + split_ratios[1]) * 10]
    test_data = data[~data.index.isin(train_data.index) & ~data.index.isin(val_data.index)]

    # Print dataset sizes
    print(f"Training set size: {len(train_data)} samples", flush=True)
    print(f"Validation set size: {len(val_data)} samples", flush=True)
    print(f"Test set size: {len(test_data)} samples", flush=True)

    return train_data, val_data, test_data

# Preprocessing function for images
IMG_SIZE = (224, 224)

def preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, tf.cast(label, tf.float32)

# Prepare tf.data.Dataset pipelines
BATCH_SIZE = 16

def create_dataset(dataframe, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((dataframe['image_path'].values, dataframe['er_status_by_ihc'].values))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# Build and compile the model
def build_model(input_shape):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = True  # Fine-tuning
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid")  # Binary classification
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

def evaluate_model(model, dataset):
    # Evaluate model on dataset
    loss, accuracy = model.evaluate(dataset, batch_size=16)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}", flush=True)

    # Predict labels
    predictions = model.predict(dataset)
    pred_labels = (predictions > 0.5).astype(int).flatten()

    # Extract true labels from the dataset
    true_labels = []
    for _, label_batch in dataset.unbatch():
        true_labels.extend(label_batch.numpy())
    true_labels = np.array(true_labels)

    # Calculate metrics
    acc = accuracy_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    roc_auc = roc_auc_score(true_labels, predictions)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    specificity = tn / (tn + fp)

    # Print metrics
    print(f"Accuracy: {acc:.4f}", flush=True)
    print(f"Recall: {recall:.4f}", flush=True)
    print(f"F1 Score: {f1:.4f}", flush=True)
    print(f"Precision: {precision:.4f}", flush=True)
    print(f"ROC-AUC: {roc_auc:.4f}", flush=True)
    print(f"Specificity: {specificity:.4f}", flush=True)

    # Confusion matrix printout
    print(f"""Confusion Matrix: 
          True Negative = {tn}, 
          False Negative = {fn}, 
          True Positive = {tp}, 
          False Positive = {fp}""", flush=True)

    # Return metrics as a dictionary
    return {
        "loss": loss,
        "accuracy": acc,
        "recall": recall,
        "f1": f1,
        "precision": precision,
        "roc_auc": roc_auc,
        "specificity": specificity
    }

# Save metrics to a file
def save_metrics(metrics, filename):
    with open(filename, "a") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n")

# Main script
if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("er_status_no_white.csv")

    # Split dataset based on hash
    train_data, val_data, test_data = split_dataset_by_hash(data)

    # Prepare datasets
    train_dataset = create_dataset(train_data, BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(val_data, BATCH_SIZE, shuffle=False)
    test_dataset = create_dataset(test_data, BATCH_SIZE, shuffle=False)

    # Build model
    input_shape = (224, 224, 3)
    model = build_model(input_shape)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    # Save the final trained model
    model.save('keras_h_2_model.h5')  # Save the final model in HDF5 format
    print("Final trained model saved as 'final_model.h5'", flush=True)
    
    # Save training and validation metrics
    train_metrics = {k: v[-1] for k, v in history.history.items()}
    save_metrics(train_metrics, "keras_h_2.txt")

    # Evaluate model on test dataset
    test_metrics = evaluate_model(model, test_dataset)

    # Save test metrics
    save_metrics(test_metrics, "keras_h_2.txt")
    print("Metrics saved to keras_h_2.txt", flush=True)
