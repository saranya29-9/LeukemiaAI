import os
from pathlib import Path
from io import BytesIO
from typing import Optional

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR.parent / "leukemia_cnn_lstm_200.keras")
    )
)

IMAGE_SIZE = 128
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB file size limit

# Adjusted threshold to 45.0% to prevent valid PBS images 
# (which might score in the 50-60% range) from being falsely rejected,
# while still filtering out completely unrelated documents/certificates.
CONFIDENCE_THRESHOLD = 45.0 

CLASS_NAMES = [
    "Acute Lymphoblastic Leukemia (ALL)",
    "Acute Myeloid Leukemia (AML)",
    "Chronic Lymphocytic Leukemia (CLL)",
    "Chronic Myeloid Leukemia (CML)",
    "Normal & Reactive",
]


# =========================================================
# FASTAPI APP SETUP
# =========================================================

app = FastAPI(
    title="LeukemiaAI API",
    description="Educational peripheral blood smear classification with OOD protection",
    version="2.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MODEL LOADING
# =========================================================

model = None
model_error: Optional[str] = None


def load_model():
    global model, model_error

    try:
        if not MODEL_PATH.exists():
            model_error = f"Model not found: {MODEL_PATH}"
            print(model_error)
            return

        loaded_model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False
        )

        output_shape = loaded_model.output_shape

        if isinstance(output_shape, tuple):
            output_count = output_shape[-1]
        else:
            output_count = None

        if output_count != len(CLASS_NAMES):
            model_error = (
                "Model output count does not match five classes. "
                f"Expected {len(CLASS_NAMES)}, got {output_count}."
            )
            print(model_error)
            return

        model = loaded_model
        model_error = None

        print("Leukemia classifier loaded successfully:", MODEL_PATH)

    except Exception as exc:
        model = None
        model_error = f"{type(exc).__name__}: {exc}"
        print("Model loading error:", model_error)


load_model()


# =========================================================
# REQUEST SCHEMAS
# =========================================================

class ChatRequest(BaseModel):
    message: str


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.asarray(image, dtype=np.float32)

    # Note: If your saved model already contains a Rescaling layer, 
    # raw values are handled inside the network.
    return np.expand_dims(image_array, axis=0)


# =========================================================
# OFFLINE LEUKEMIAAI CHATBOT
# =========================================================

def get_chatbot_answer(message: str) -> str:
    question = " ".join(message.lower().strip().split())

    if not question:
        return "Please enter a question."

    if question in ["hi", "hello", "hey", "hii", "good morning", "good evening"]:
        return """
Hello! I am the offline LeukemiaAI educational chatbot.

I can explain leukemia, blood smear images, the CNN-LSTM model,
prediction scores, and your project workflow.

I cannot diagnose a patient.
"""

    if any(term in question for term in ["pbs", "blood smear", "upload", "unrelated image", "other photo", "selfie", "reject", "certificate", "document"]):
        return f"""
LeukemiaAI is designed for peripheral blood smear (PBS) images.

To filter out completely unrelated documents or certificates, 
this version includes a Confidence Threshold ({CONFIDENCE_THRESHOLD}%). 

If an uploaded image does not match proper blood cell features and 
model certainty falls below {CONFIDENCE_THRESHOLD}%, it will be flagged as an 
'Uncertain / Non-PBS Image'.

Please upload a genuine peripheral blood smear image.
"""

    if any(term in question for term in ["cnn-lstm", "cnn lstm", "lstm", "cnn"]):
        return """
CNN stands for Convolutional Neural Network (learns visual patterns), and 
LSTM stands for Long Short-Term Memory (recurrent component). 
Together they handle sequential feature learning in blood cell images.
"""

    if any(term in question for term in ["symptom", "fever", "bruising", "tired"]):
        return """
Possible leukemia symptoms include persistent tiredness, fever, easy bruising, 
bleeding, swollen lymph nodes, and recurrent infections. Consult a healthcare 
professional for proper medical evaluation.
"""

    if any(term in question for term in ["type", "all", "aml", "cll", "cml"]):
        return """
The four primary leukemia types handled in this project are:
1. ALL (Acute Lymphoblastic Leukemia)
2. AML (Acute Myeloid Leukemia)
3. CLL (Chronic Lymphocytic Leukemia)
4. CML (Chronic Myeloid Leukemia)
Plus the 'Normal & Reactive' class.
"""

    return """
I am the offline LeukemiaAI educational chatbot. You can ask me about leukemia types, 
CNN-LSTM architectures, confidence scores, or image validation thresholds.
"""


# =========================================================
# HOME & HEALTH ENDPOINTS
# =========================================================

@app.get("/")
def home():
    return {
        "message": "LeukemiaAI API is running",
        "model_file": MODEL_PATH.name,
        "chat_mode": "offline",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_unavailable",
        "model_available": model is not None,
        "model_file": MODEL_PATH.name,
        "model_error": model_error,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "chat_available": True,
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    image_bytes = await file.read(MAX_FILE_BYTES + 1)

    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Image size must be 10 MB or less.")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")

    if model is None:
        raise HTTPException(status_code=503, detail=f"Leukemia model unavailable: {model_error}")

    try:
        processed_image = preprocess_image(image)
        scores = np.asarray(model.predict(processed_image, verbose=0)).reshape(-1)

        if scores.size != len(CLASS_NAMES):
            raise ValueError(f"Model returned {scores.size} scores; expected {len(CLASS_NAMES)}.")

        if not np.all(np.isfinite(scores)):
            raise ValueError("Model returned invalid numerical values.")

        # Convert raw scores to probabilities using softmax if needed
        if np.any(scores < 0) or not np.isclose(float(np.sum(scores)), 1.0, atol=0.05):
            shifted_scores = scores - np.max(scores)
            exp_values = np.exp(shifted_scores)
            scores = exp_values / np.sum(exp_values)
        else:
            scores = scores / max(float(np.sum(scores)), 1e-8)

        best_index = int(np.argmax(scores))
        confidence = round(float(scores[best_index]) * 100, 2)

        class_probabilities = {
            CLASS_NAMES[index]: round(float(scores[index]) * 100, 2)
            for index in range(len(CLASS_NAMES))
        }

        # --- OUT-OF-DISTRIBUTION (OOD) / REJECTION CHECK ---
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "success": True,
                "predicted_class": "Uncertain / Non-PBS Image",
                "result": "Uncertain / Non-PBS Image",
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "pbs_validation": {
                    "performed": True,
                    "note": f"Confidence ({confidence}%) is below the acceptance threshold ({CONFIDENCE_THRESHOLD}%). Image rejected as potential non-PBS/unrelated document."
                },
                "disclaimer": "Warning: Model confidence is too low or the uploaded image does not appear to be a valid Peripheral Blood Smear. Please upload a clear blood cell slide."
            }

        return {
            "success": True,
            "predicted_class": CLASS_NAMES[best_index],
            "result": CLASS_NAMES[best_index],
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "pbs_validation": {
                "performed": True,
                "note": "Passed confidence threshold check."
            },
            "disclaimer": "Educational prototype only. This is not a medical diagnosis. Consult a qualified healthcare professional."
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}")


# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    return {
        "success": True,
        "mode": "offline",
        "reply": get_chatbot_answer(message),
    }