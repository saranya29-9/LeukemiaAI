
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

# Model location
MODEL_PATH = Path(__file__).parent / "leukemia_cnn_lstm.keras"

# Dataset location
DATA_DIR = Path(__file__).parent / "data" / "sample"

# Image dimensions
IMAGE_SIZE = (128, 128)

# Class names in the same alphabetical order
CLASS_NAMES = sorted([folder.name for folder in DATA_DIR.iterdir() if folder.is_dir()])

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Get image path from command line
if len(sys.argv) < 2:
    print("Usage: python predict_image.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

# Load and prepare image
img = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE
)

img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array, verbose=0)[0]

predicted_index = np.argmax(predictions)
predicted_class = CLASS_NAMES[predicted_index]
confidence = predictions[predicted_index] * 100

print("\nPrediction:", predicted_class)
print(f"Model score: {confidence:.2f}%")
print("\nScores for all classes:")

for class_name, score in zip(CLASS_NAMES, predictions):
    print(f"{class_name}: {score * 100:.2f}%")
