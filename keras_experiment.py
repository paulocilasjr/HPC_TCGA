import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import hashlib

# Load and preprocess the dataset
data = pd.read_csv("er_status_no_white.csv")

# Hash-based dataset splitting function
def split_dataset_by_sample(data, split_ratios=[0.7, 0.2, 0.1], seed=42):
    np.random.seed(seed)
    
    def hash_sample(sample):
        """Hash the sample to an integer."""
        return int(hashlib.md5(sample.encode()).hexdigest(), 16)
    
    # Create a hash value for each sample
    data['sample_hash'] = data['sample'].apply(hash_sample)
    # Normalize the hash values to [0, 1]
    data['split_key'] = data['sample_hash'] % 10000 / 10000.0

    # Assign splits based on the normalized hash
    train_data = data[data['split_key'] < split_ratios[0]]
    val_data = data[(data['split_key'] >= split_ratios[0]) & (data['split_key'] < sum(split_ratios[:2]))]
    test_data = data[data['split_key'] >= sum(split_ratios[:2])]
    
    return train_data, val_data, test_data

train_data, val_data, test_data = split_dataset_by_sample(data)

# Image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Preprocessing function
def preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_jpeg_quality(image, min_jpeg_quality=70, max_jpeg_quality=100)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, tf.cast(label, tf.float32)

# Prepare tf.data.Dataset pipelines
def create_dataset(dataframe, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((dataframe['image_path'].values, dataframe['er_status_by_ihc'].values))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

train_dataset = create_dataset(train_data, BATCH_SIZE, shuffle=True)
val_dataset = create_dataset(val_data, BATCH_SIZE, shuffle=False)
test_dataset = create_dataset(test_data, BATCH_SIZE, shuffle=False)

# Build the ResNet model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Define optimizer and loss
optimizer = optimizers.Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Training loop
EPOCHS = 100
patience = 5
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    # Training step
    train_loss = tf.keras.metrics.Mean()
    for images, labels in train_dataset:
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)
    print(f"Train Loss: {train_loss.result().numpy():.4f}")

    # Validation step
    val_loss = tf.keras.metrics.Mean()
    for images, labels in val_dataset:
        predictions = model(images, training=False)
        loss = loss_fn(labels, predictions)
        val_loss.update_state(loss)
    print(f"Validation Loss: {val_loss.result().numpy():.4f}")

    # Early stopping
    if val_loss.result().numpy() < best_val_loss:
        best_val_loss = val_loss.result().numpy()
        patience_counter = 0
        model.save("best_model.keras")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# Load the best model
model = tf.keras.models.load_model("best_model.keras")

# Evaluate on test set
test_loss = tf.keras.metrics.Mean()
test_labels_list = []
test_preds_list = []

for images, labels in test_dataset:
    predictions = model(images, training=False)
    test_loss.update_state(loss_fn(labels, predictions))
    test_labels_list.extend(labels.numpy())
    test_preds_list.extend(predictions.numpy())

test_labels = np.array(test_labels_list)
test_preds = np.array(test_preds_list).flatten()
predicted_classes = (test_preds > 0.5).astype(int)

# Calculate metrics
roc_auc = roc_auc_score(test_labels, test_preds)
precision = precision_score(test_labels, predicted_classes)
recall = recall_score(test_labels, predicted_classes)
cm = confusion_matrix(test_labels, predicted_classes)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)

# Save results to a file
with open("keras_nh.txt", "w") as result_file:
    result_file.write(f"Test Loss: {test_loss.result().numpy():.4f}\n")
    result_file.write(f"ROC-AUC: {roc_auc:.4f}\n")
    result_file.write(f"Precision: {precision:.4f}\n")
    result_file.write(f"Recall: {recall:.4f}\n")
    result_file.write(f"Specificity: {specificity:.4f}\n")
    result_file.write(f"Confusion Matrix:\n{cm}\n")
    result_file.write(f"Classification Report:\n{classification_report(test_labels, predicted_classes)}\n")

print("Results saved to keras_nh.txt")

