import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Split dataset randomly
def split_dataset_randomly(data, split_ratios=[0.7, 0.2, 0.1], seed=42):
    train_size = split_ratios[0]
    temp_size = 1.0 - train_size

    train_data, temp_data = train_test_split(
        data, test_size=temp_size, random_state=seed
    )

    val_size = split_ratios[1] / temp_size

    val_data, test_data = train_test_split(
        temp_data, test_size=1.0 - val_size, random_state=seed
    )

    # Print dataset sizes
    print(f"Training set size: {len(train_data)} samples")
    print(f"Validation set size: {len(val_data)} samples")
    print(f"Test set size: {len(test_data)} samples")

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

# Evaluate and calculate metrics
def evaluate_model(model, dataset, true_labels):
    predictions = model.predict(dataset)
    pred_labels = (predictions > 0.5).astype(int).flatten()
    
    acc = accuracy_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    roc_auc = roc_auc_score(true_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    specificity = tn / (tn + fp)

    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Specificity: {specificity:.4f}")

    return {"accuracy": acc, "recall": recall, "f1": f1, "roc_auc": roc_auc, "specificity": specificity}

# Main script
if __name__ == "__main__":
    # Load dataset
    data = pd.read_csv("er_status_no_white.csv")
    
    # Split dataset randomly
    train_data, val_data, test_data = split_dataset_randomly(data)

    # Prepare datasets
    train_dataset = create_dataset(train_data, BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(val_data, BATCH_SIZE, shuffle=False)
    test_dataset = create_dataset(test_data, BATCH_SIZE, shuffle=False)

    # Build model
    input_shape = (224, 224, 3)
    model = build_model(input_shape)

    # Train the model
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    # Evaluate model on test dataset
    test_labels = test_data['er_status_by_ihc'].values
    metrics = evaluate_model(model, test_dataset, test_labels)

    # Print metrics
    print("Test set metrics:", metrics)
