from pathlib import Path
import tensorflow as tf
import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "sample_200"

dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(128, 128),
    batch_size=8,
    shuffle=True,
    seed=42,
    label_mode="categorical"
)

print("\nClass names:")
print(dataset.class_names)

images, labels = next(iter(dataset))

print("\nImage batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("Minimum pixel value:", np.min(images.numpy()))
print("Maximum pixel value:", np.max(images.numpy()))
print("Mean pixel value:", np.mean(images.numpy()))

print("\nFirst batch labels:")
print(np.argmax(labels.numpy(), axis=1))