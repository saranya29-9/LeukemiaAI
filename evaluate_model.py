from pathlib import Path
import numpy as np
import tensorflow as tf

# Project paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data" / "sample"
MODEL_PATH = PROJECT_DIR / "leukemia_cnn_lstm.keras"

# Settings used during training
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
SEED = 42
VALIDATION_SPLIT = 0.2

# Load the validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    label_mode="categorical"
)

class_names = val_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
for name in class_names:
    print("-", name)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate images and labels together
actual_labels = []
predicted_labels = []

for images, labels in val_ds:
    predictions = model(images, training=False).numpy()

    actual_labels.extend(np.argmax(labels.numpy(), axis=1))
    predicted_labels.extend(np.argmax(predictions, axis=1))

actual_labels = np.array(actual_labels)
predicted_labels = np.array(predicted_labels)

# Overall accuracy
accuracy = np.mean(actual_labels == predicted_labels) * 100

print(f"\nValidation images: {len(actual_labels)}")
print(f"Overall accuracy: {accuracy:.2f}%")

# Confusion matrix
confusion = np.zeros((num_classes, num_classes), dtype=int)

for actual, predicted in zip(actual_labels, predicted_labels):
    confusion[actual, predicted] += 1

print("\nConfusion Matrix")
print("Rows = Actual class | Columns = Predicted class")
print("Class order:", class_names)
print(confusion)

# Per-class precision and recall
print("\nClass-wise Results")

for i, class_name in enumerate(class_names):
    true_positive = confusion[i, i]
    false_positive = confusion[:, i].sum() - true_positive
    false_negative = confusion[i, :].sum() - true_positive

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    print(f"\n{class_name}")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")