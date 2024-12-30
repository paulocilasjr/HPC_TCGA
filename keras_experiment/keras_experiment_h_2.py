import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import hashlib

# Configuration parameters
IMG_SIZE = (224, 224)  # ResNet standard input size
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.0001
PATIENCE = 5
EARLY_STOP_PATIENCE = 10
SEED = 42

# Paths to dataset and columns
DATA_PATH = "."  
IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "er_status_by_ihc"

# Hash-based Dataset Splitting Function
def split_dataset_by_hash(data, split_ratios=[0.7, 0.1, 0.2], seed=42):
    np.random.seed(seed)

    def hash_sample(sample):
        return int(hashlib.md5(sample.encode()).hexdigest(), 16)

    # Generate a normalized hash key for splitting
    data["sample_hash"] = data[IMAGE_COLUMN].apply(hash_sample)
    data["split_key"] = data["sample_hash"] % 10000 / 10000.0

    train_data = data[data["split_key"] < split_ratios[0]]
    val_data = data[
        (data["split_key"] >= split_ratios[0]) & (data["split_key"] < sum(split_ratios[:2]))
    ]
    test_data = data[data["split_key"] >= sum(split_ratios[:2])]

# Print dataset sizes
    print(f"Training set size: {len(train_data)} samples")
    print(f"Validation set size: {len(val_data)} samples")
    print(f"Test set size: {len(test_data)} samples")

    return train_data, val_data, test_data

# Data Preprocessing Functions
def preprocess_image(file_path, label):
    # Read and decode image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize and normalize
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    # Data Augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=70, max_jpeg_quality=100)
    return image, tf.cast(label, tf.float32)

def create_dataset(df, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((df[IMAGE_COLUMN].values, df[LABEL_COLUMN].values))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load dataset and split
data = pd.read_csv(DATA_PATH)
train_data, val_data, test_data = split_dataset_by_hash(data)

train_dataset = create_dataset(train_data, BATCH_SIZE, shuffle=True)
val_dataset = create_dataset(val_data, BATCH_SIZE, shuffle=False)
test_dataset = create_dataset(test_data, BATCH_SIZE, shuffle=False)

# Build Model (ResNet50)
base_model = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = True  # Fine-tune the ResNet50 model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")  # Binary classification
])

# Compile Model
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc"), tf.keras.metrics.Recall(name="recall")]
)

# Callbacks
reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=PATIENCE,
    verbose=1
)
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=EARLY_STOP_PATIENCE,
    restore_best_weights=True
)

# Train Model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping]
)

# Evaluation Functions
def evaluate_model(dataset, dataset_name):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.flatten())

    y_pred_classes = [1 if p > 0.5 else 0 for p in y_pred]
    roc_auc = roc_auc_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    report = classification_report(y_true, y_pred_classes, output_dict=True)
    f1_score = report["1"]["f1-score"]
    recall = report["1"]["recall"]
    accuracy = report["accuracy"]

    print(f"--- {dataset_name} Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: N/A (calculated during training)")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print()

# Evaluate on train, validation, and test sets
evaluate_model(train_dataset, "Train")
evaluate_model(val_dataset, "Validation")
evaluate_model(test_dataset, "Test")
