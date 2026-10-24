
from pathlib import Path
import tensorflow as tf

# ==========================================
# PROJECT SETTINGS
# ==========================================

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data" / "sample_200"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 15
SEED = 42

MODEL_PATH = PROJECT_DIR / "baseline_cnn_best.keras"

# ==========================================
# LOAD TRAINING DATASET
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    label_mode="categorical"
)

# ==========================================
# LOAD VALIDATION DATASET
# ==========================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")
for class_name in class_names:
    print(class_name)

print("\nNumber of classes:", num_classes)

# ==========================================
# IMPROVE DATA LOADING
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================================
# BUILD SIMPLE CNN MODEL
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),

    # Normalize pixel values
    tf.keras.layers.Rescaling(1.0 / 255),

    # First convolution block
    tf.keras.layers.Conv2D(
        16,
        kernel_size=3,
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # Second convolution block
    tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # Third convolution block
    tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    # Reduce feature dimensions
    tf.keras.layers.GlobalAveragePooling2D(),

    # Classification layers
    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    # Five output classes
    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model structure
model.summary()

# ==========================================
# CALLBACKS
# ==========================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
       filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n================================")
print("BASELINE CNN TRAINING COMPLETED")
print("================================")

print(
    "Final training accuracy:",
    history.history["accuracy"][-1]
)

print(
    "Final validation accuracy:",
    history.history["val_accuracy"][-1]
)

print("\nBest model location:")
print(MODEL_PATH)
