
from pathlib import Path
import numpy as np
import tensorflow as tf

# Project paths
DATA_DIR = Path(r"C:\LeukemiaProject\data\sample_200")
MODEL_PATH = Path(r"C:\LeukemiaProject\baseline_cnn_best.keras")

# Image settings — keep these same as training
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
SEED = 42

# Load validation images
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    label_mode="categorical"
)

class_names = val_ds.class_names
print("\nClasses:", class_names)

# Load the best saved model
model = tf.keras.models.load_model(MODEL_PATH)

# Collect actual labels and predictions
actual_labels = []
predicted_labels = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)

    actual_labels.extend(
        np.argmax(labels.numpy(), axis=1)
    )

    predicted_labels.extend(
        np.argmax(predictions, axis=1)
    )

actual_labels = np.array(actual_labels)
predicted_labels = np.array(predicted_labels)

# Create confusion matrix
num_classes = len(class_names)
matrix = np.zeros(
    (num_classes, num_classes),
    dtype=int
)

for actual, predicted in zip(actual_labels, predicted_labels):
    matrix[actual, predicted] += 1

print("\nCONFUSION MATRIX")
print("Rows = Actual | Columns = Predicted")
print(matrix)

# Calculate per-class precision and recall
print("\nCLASS-WISE RESULTS")

for i, class_name in enumerate(class_names):
    true_positive = matrix[i, i]
    actual_count = matrix[i].sum()
    predicted_count = matrix[:, i].sum()

    recall = (
        true_positive / actual_count
        if actual_count > 0 else 0
    )

    precision = (
        true_positive / predicted_count
        if predicted_count > 0 else 0
    )

    print(f"\nClass: {class_name}")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")

# Overall accuracy
accuracy = np.mean(actual_labels == predicted_labels)

print("\nOVERALL VALIDATION ACCURACY")
print(f"{accuracy * 100:.2f}%")
