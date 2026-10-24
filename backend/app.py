
from pathlib import Path
from io import BytesIO
import os

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# Project paths
# Works on Windows and cloud hosting platforms
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Allows a different model path to be configured during deployment
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(PROJECT_DIR / "baseline_cnn_best.keras")
    )
)

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

# Model is loaded only when a prediction is requested
model = None


# Create FastAPI application
app = FastAPI(
    title="Leukemia Detection API",
    description="Experimental blood smear image classification API",
    version="1.0.0"
)


# Configure frontend access
# Set FRONTEND_URL during deployment if needed.
frontend_url = os.getenv("FRONTEND_URL", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model():
    """Load the model when needed and reuse it for later predictions."""
    global model

    if model is None:
        if not MODEL_PATH.is_file():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model file is unavailable. "
                    "Configure MODEL_PATH to point to the trained model."
                )
            )

        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=f"Could not load the trained model: {str(error)}"
            )

    return model


@app.get("/")
def home():
    return {
        "message": "Leukemia Detection API is running",
        "model_file": MODEL_PATH.name
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_available": MODEL_PATH.is_file()
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

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty."
            )

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

        # Load model and make prediction
        loaded_model = get_model()

        predictions = loaded_model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(predictions))

        predicted_class = CLASS_NAMES[predicted_index]

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

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image. Please use a valid JPG or PNG."
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

    finally:
        await file.close()
