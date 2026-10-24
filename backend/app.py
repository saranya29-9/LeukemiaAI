
from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# Project paths
PROJECT_DIR = Path(r"C:\LeukemiaProject")
MODEL_PATH = PROJECT_DIR / "baseline_cnn_best.keras"

# Image settings
IMAGE_SIZE = (128, 128)

# Class names must follow the model's training order
CLASS_NAMES = [
    "Acute Lymphoblastic Leukemia",
    "Acute Myeloid Leukemia",
    "Chronic Lymphocytic Leukemia",
    "Chronic Myeloid Leukemia",
    "Normal & Reactive",
]

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Create FastAPI application
app = FastAPI(
    title="Leukemia Detection API",
    description="Experimental blood smear image classification API",
    version="1.0.0"
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Leukemia Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check uploaded file type
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    try:
        # Read uploaded image
        image_bytes = await file.read()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # Resize image
        image = image.resize(IMAGE_SIZE)

        # Convert image to NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Make prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index] * 100
        )

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "message": (
                "Experimental model output. "
                "Not a medical diagnosis."
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(error)}"
        )
