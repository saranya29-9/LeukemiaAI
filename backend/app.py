
from pathlib import Path
from io import BytesIO
import os

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# PROJECT SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(PROJECT_DIR / "leukemia_cnn_lstm_200.keras")
    )
)

IMAGE_SIZE = (128, 128)

# Adjustable confidence threshold
CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "70")
)

# Maximum upload size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# Minimum image dimensions
MIN_IMAGE_SIZE = 32

# Maximum image dimensions
MAX_IMAGE_DIMENSION = 10000


# --------------------------------------------------
# MODEL CLASS NAMES
# Must match training class order
# --------------------------------------------------

CLASS_NAMES = [
    "Acute Lymphoblastic Leukemia",
    "Acute Myeloid Leukemia",
    "Chronic Lymphocytic Leukemia",
    "Chronic Myeloid Leukemia",
    "Normal & Reactive",
]


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = None


def get_model():
    """Load the trained model only when required."""

    global model

    if model is None:

        if not MODEL_PATH.is_file():
            raise HTTPException(
                status_code=503,
                detail="The trained model file is unavailable."
            )

        try:
            model = tf.keras.models.load_model(
                MODEL_PATH
            )

        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=f"Could not load the model: {str(error)}"
            )

    return model


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Leukemia Detection API",
    description=(
        "Experimental blood smear image classification API "
        "using a CNN-LSTM model."
    ),
    version="1.1.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

frontend_url = os.getenv("FRONTEND_URL", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# IMAGE VALIDATION
# --------------------------------------------------

def validate_image(image_bytes):
    """
    Validate image file format, dimensions, and readability.

    Note:
    This validates the image file itself.
    It does NOT prove that the image is a peripheral
    blood smear image.
    """

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty."
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Image size must be less than 10 MB."
        )

    try:

        # Verify image integrity
        with Image.open(
            BytesIO(image_bytes)
        ) as img:

            if img.format not in ["JPEG", "PNG"]:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid image format. "
                        "Please upload a JPG or PNG image."
                    )
                )

            width, height = img.size

            if (
                width < MIN_IMAGE_SIZE
                or height < MIN_IMAGE_SIZE
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Image resolution is too small. "
                        "Please upload a clearer image."
                    )
                )

            if (
                width > MAX_IMAGE_DIMENSION
                or height > MAX_IMAGE_DIMENSION
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions are too large."
                )

            img.verify()

        # Reopen image after verification
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        return image

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the uploaded image. "
                "Please upload a valid JPG or PNG image."
            )
        )


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):
    """
    Resize and convert image to model input format.

    IMPORTANT:
    This currently uses 0-255 pixel values, matching
    your original app.py preprocessing.

    If your model was trained using rescale=1./255,
    this preprocessing must also be changed.
    """

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Leukemia Detection API is running",
        "model_file": MODEL_PATH.name,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_available": MODEL_PATH.is_file()
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        # Check declared upload content type
        if file.content_type not in [
            "image/jpeg",
            "image/png",
            "image/jpg",
            "application/octet-stream"
        ]:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Please upload a JPG or PNG image."
                )
            )

        # Read uploaded image
        image_bytes = await file.read()

        # Validate actual image content
        image = validate_image(
            image_bytes
        )

        # Preprocess image
        image_array = preprocess_image(
            image
        )

        # Load trained model
        loaded_model = get_model()

        # Predict
        predictions = loaded_model.predict(
            image_array,
            verbose=0
        )[0]

        # Verify output dimensions
        if len(predictions) != len(CLASS_NAMES):

            raise HTTPException(
                status_code=500,
                detail=(
                    "Model output does not match "
                    "the configured class names."
                )
            )

        # Verify predictions are valid
        if not np.all(
            np.isfinite(predictions)
        ):

            raise HTTPException(
                status_code=500,
                detail="Model returned invalid prediction values."
            )

        # Find highest prediction
        predicted_index = int(
            np.argmax(predictions)
        )

        confidence = float(
            predictions[predicted_index] * 100
        )

        # Reject low-confidence prediction
        if confidence < CONFIDENCE_THRESHOLD:

            return {
                "predicted_class": "Unable to classify",
                "confidence": round(
                    confidence,
                    2
                ),
                "message": (
                    "The model confidence is too low. "
                    "Please upload a clear blood smear image."
                ),
                "medical_disclaimer": (
                    "Experimental model output. "
                    "Not a medical diagnosis."
                )
            }

        # Accepted prediction
        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        return {
            "predicted_class": predicted_class,
            "confidence": round(
                confidence,
                2
            ),
            "message": (
                "Experimental model output. "
                "Not a medical diagnosis."
            )
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

    finally:

        await file.close()
