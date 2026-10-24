
import os
import io
import re
import threading
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import tensorflow as tf
from openai import APITimeoutError, OpenAI, OpenAIError


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(PROJECT_DIR / "leukemia_cnn_lstm_200.keras"))
).expanduser()
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_DIR / MODEL_PATH

IMAGE_SIZE = (128, 128)
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

CLASS_NAMES = [
    "Acute Lymphoblastic Leukemia",
    "Acute Myeloid Leukemia",
    "Chronic Lymphocytic Leukemia",
    "Chronic Myeloid Leukemia",
    "Normal & Reactive",
]

DEFAULT_FRONTEND_ORIGIN = (
    "https://leukemiaai-frontend-git-main-saranyadunnapothula-5686.vercel.app"
)
FRONTEND_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FRONTEND_ORIGINS", DEFAULT_FRONTEND_ORIGIN
    ).split(",")
    if origin.strip()
]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
if not OPENAI_MODEL:
    OPENAI_MODEL = "gpt-4o-mini"

app = FastAPI(
    title="LeukemiaAI",
    description="Leukemia prediction and educational chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

model: tf.keras.Model | None = None
_model_lock = threading.Lock()


# ---------------- MODEL LOADING ----------------

def get_model() -> tf.keras.Model:
    global model

    if model is None:
        with _model_lock:
            if model is not None:
                return model

            if not MODEL_PATH.is_file():
                raise RuntimeError(
                    f"Model file is unavailable at {MODEL_PATH}. "
                    "Include the trained model in the deployment or set MODEL_PATH "
                    "to its deployed location."
                )

            print("Loading LeukemiaAI model...")

            try:
                loaded_model = tf.keras.models.load_model(
                    str(MODEL_PATH),
                    compile=False
                )
            except Exception as error:
                raise RuntimeError(
                    f"Could not load model at {MODEL_PATH}: {error}"
                ) from error

            if loaded_model.input_shape != (None, *IMAGE_SIZE, 3):
                raise RuntimeError(
                    f"Model input shape {loaded_model.input_shape} does not match "
                    f"the required (None, {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}, 3)."
                )
            if loaded_model.output_shape != (None, len(CLASS_NAMES)):
                raise RuntimeError(
                    f"Model output shape {loaded_model.output_shape} does not match "
                    f"the {len(CLASS_NAMES)} configured classes."
                )

            model = loaded_model

            print("Model loaded successfully.")

    return model


def get_model_type(loaded_model: tf.keras.Model) -> str:
    if any(isinstance(layer, tf.keras.layers.LSTM) for layer in loaded_model.layers):
        return "CNN-LSTM hybrid"
    return "CNN"


# ---------------- HEALTH CHECK ----------------

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "LeukemiaAI API is running",
        "model_file": MODEL_PATH.name,
    }


@app.get("/health")
def health():
    try:
        loaded_model = get_model()
    except Exception as error:
        print("Model health check failed:", repr(error))
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "model_available": False,
                "model_file": MODEL_PATH.name,
                "error": str(error),
            },
        )

    return {
        "status": "healthy",
        "model_available": True,
        "model_file": MODEL_PATH.name,
        "model_type": get_model_type(loaded_model),
        "input_shape": list(loaded_model.input_shape),
        "output_shape": list(loaded_model.output_shape),
    }


# ---------------- PREDICTION ----------------

@app.post("/predict")
def predict(file: UploadFile = File(...)):

    try:
        image_bytes = file.file.read(MAX_IMAGE_BYTES + 1)
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty."
            )
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=400,
                detail="Image must be smaller than 10 MB."
            )

        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise HTTPException(
                    status_code=400,
                    detail="Please upload a JPEG or PNG image."
                )
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions are too large to process."
                )
            image.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid JPEG or PNG image."
        )

    try:
        decoded_image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )
        resized_image = tf.image.resize(
            decoded_image,
            IMAGE_SIZE,
            method="bilinear"
        )
        image_array = np.expand_dims(
            np.asarray(resized_image, dtype=np.float32),
            axis=0
        )
    except (tf.errors.InvalidArgumentError, ValueError) as error:
        print("Image decoding error:", repr(error))
        raise HTTPException(
            status_code=400,
            detail="The uploaded image could not be decoded."
        )

    try:
        loaded_model = get_model()
        predictions = np.asarray(
            loaded_model.predict(image_array, verbose=0)[0],
            dtype=np.float32
        )
    except RuntimeError as error:
        print("Model unavailable:", repr(error))
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        print("Prediction error:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="The prediction model could not process this image. "
            "Check the server logs for details."
        )

    if (
        predictions.shape != (len(CLASS_NAMES),)
        or not np.all(np.isfinite(predictions))
        or np.any(predictions < 0)
        or np.any(predictions > 1)
        or not np.isclose(float(np.sum(predictions)), 1.0, atol=1e-3)
    ):
        raise HTTPException(
            status_code=500,
            detail="The model returned invalid class probabilities."
        )

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index] * 100)
    is_normal = predicted_class == "Normal & Reactive"
    result = "Normal & Reactive" if is_normal else "Leukemia Detected"

    class_probabilities = {
        CLASS_NAMES[index]: round(float(predictions[index] * 100), 2)
        for index in range(len(CLASS_NAMES))
    }

    return {
        "success": True,
        "predicted_class": predicted_class,
        "result": result,
        "is_normal": is_normal,
        "confidence": round(confidence, 2),
        "class_probabilities": class_probabilities,
        "message": (
            "The model predicted the Normal & Reactive class."
            if is_normal
            else "The model predicted a leukemia-related class."
        ),
        "disclaimer": (
            "Educational AI prediction only. "
            "This result is not a medical diagnosis. "
            "Only peripheral blood smear images are within scope; "
            "other images may be misclassified."
        )
    }


# ---------------- CHATBOT ----------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)


@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question."
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        fallback = get_educational_fallback(user_message, request.history)
        if fallback:
            return {
                "success": True,
                "mode": "educational_fallback",
                "answer": fallback,
            }
        raise HTTPException(
            status_code=503,
            detail=(
                "The OpenAI chatbot is not configured (OPENAI_API_KEY is missing), "
                "and the offline educational fallback does not cover this question."
            ),
        )

    try:
        client = OpenAI(api_key=api_key, timeout=45.0, max_retries=0)

        messages = [
            {
                "role": "system",
                "content": """
You are LeukemiaAI Assistant.

You are a conversational educational assistant for leukemia, blood-cell
health, and this project's image-classification workflow. Answer the
question actually asked in clear, simple English. You are not limited to
the listed examples or a fixed FAQ.

You can answer questions about:

1. Leukemia meaning and overview
2. ALL, AML, CLL and CML
3. Symptoms and warning signs
4. General risk factors and possible causes
5. Blood tests and diagnostic procedures
6. Treatment approaches
7. Chemotherapy, targeted therapy and immunotherapy
8. General side effects and patient support
9. Differences between normal and leukemia blood cells
10. General awareness and educational questions

Instructions:

- Answer the user's actual question.
- Understand misspellings, shorthand, incomplete questions, and common
  conversational wording. If the intent is reasonably clear, answer it;
  ask one brief clarifying question only when necessary.
- Answer general educational questions even when they are not listed above.
- Give detailed answers when requested.
- Use headings and bullet points where helpful.
- Understand follow-up questions using conversation history.
- Explain medical terms in simple language.
- Explain that a model's class probability is not a validated clinical
  confidence or proof of a diagnosis. Do not invent this project's
  validation accuracy or claim the model is reliable.
- Do not diagnose the user.
- Do not prescribe medicines or dosages.
- Do not claim that symptoms alone confirm leukemia.
- Encourage consultation with a qualified hematologist
  for personal medical concerns.
- For urgent or severe symptoms, advise seeking medical care.

This chatbot provides educational information only.
"""
            }
        ]

        # Keep only the latest 12 conversation messages.

        for item in request.history[-12:]:

            messages.append({
                "role": item.role,
                "content": item.content
            })

        messages.append({
            "role": "user",
            "content": user_message
        })

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The chatbot returned an empty response. Please try again."
            )

        return {
            "success": True,
            "answer": answer
        }

    except HTTPException:
        raise

    except APITimeoutError as error:
        print("Chatbot provider timeout:", repr(error))
        fallback = get_educational_fallback(user_message, request.history)
        if fallback:
            return {
                "success": True,
                "mode": "educational_fallback",
                "answer": fallback,
            }
        raise HTTPException(
            status_code=504,
            detail="The chatbot provider timed out. Please try again."
        )

    except OpenAIError as error:
        print("Chatbot provider error:", repr(error))
        fallback = get_educational_fallback(user_message, request.history)
        if fallback:
            return {
                "success": True,
                "mode": "educational_fallback",
                "answer": fallback,
            }
        raise HTTPException(
            status_code=502,
            detail="The chatbot service is temporarily unavailable. Please try again."
        )

    except Exception as error:
        print("Chatbot error:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="The chatbot could not answer this question. Please try again."
        )


def get_educational_fallback(
    message: str,
    history: list[ChatMessage],
) -> str | None:
    """Return a clearly labeled, limited offline answer for supported topics."""
    question = message.casefold()
    if any(term in question for term in ("symptom", "warning sign", "signs of")):
        return (
            "Leukemia can cause fatigue, fever or frequent infections, easy bruising "
            "or bleeding, and sometimes bone pain. These symptoms can have many "
            "causes and cannot confirm leukemia. A healthcare professional can "
            "evaluate personal concerns."
        )
    if re.search(r"\ball\b", question) and re.search(r"\baml\b", question):
        return (
            "ALL (acute lymphoblastic leukemia) affects immature lymphoid cells; "
            "AML (acute myeloid leukemia) affects immature myeloid cells. Both are "
            "acute leukemias, but they are distinct diseases diagnosed using "
            "professional laboratory testing."
        )
    if re.search(r"\b(all|aml|cll|cml)\b", question):
        type_information = {
            "all": "ALL means acute lymphoblastic leukemia, which affects immature lymphoid cells.",
            "aml": "AML means acute myeloid leukemia, which affects immature myeloid cells.",
            "cll": "CLL means chronic lymphocytic leukemia, which involves more mature lymphocytes.",
            "cml": "CML means chronic myeloid leukemia, which involves myeloid cells and is often associated with the BCR-ABL1 change.",
        }
        matched_types = [
            type_information[name]
            for name in ("all", "aml", "cll", "cml")
            if re.search(rf"\b{name}\b", question)
        ]
        return " ".join(matched_types) + (
            " These are educational descriptions, not a diagnosis."
        )
    if any(term in question for term in ("blood test", "diagnos", "detect", "cbc", "test")):
        return (
            "Evaluation may include a complete blood count (CBC), examination of a "
            "blood smear, and, when indicated, bone-marrow and specialized laboratory "
            "tests. A qualified clinician interprets these results; this chatbot and "
            "the image model cannot diagnose leukemia."
        )
    if any(term in question for term in ("treatment", "chemotherapy", "immunotherapy", "therapy")):
        return (
            "Leukemia treatment depends on its specific type and individual clinical "
            "factors. Care may involve chemotherapy, targeted therapy, immunotherapy, "
            "stem-cell transplant, or monitoring in some situations. Treatment "
            "decisions belong with a qualified oncology team."
        )
    if any(term in question for term in ("confidence", "probability score", "score mean")):
        return (
            "The displayed score is the model's probability-like output for a class. "
            "It is not a clinically validated confidence, does not measure the "
            "chance that a person has leukemia, and is not a diagnosis."
        )
    if any(term in question for term in ("project", "workflow", "blood smear image", "analyze an image")):
        return (
            "The project resizes an uploaded RGB image to 128 × 128, then sends it "
            "to the trained CNN-LSTM model. The model produces scores for its five "
            "trained classes. This experimental result is not a medical diagnosis."
        )
    if any(term in question for term in ("cnn-lstm", "cnn lstm", "lstm")):
        return (
            "A CNN learns visual features from an image. An LSTM processes those "
            "features as a sequence. In a CNN-LSTM hybrid, the CNN extracts image "
            "patterns and the LSTM uses their sequence to produce a classification."
        )
    if "cnn" in question or "convolutional neural network" in question:
        return (
            "A convolutional neural network (CNN) is a type of model that learns "
            "visual patterns such as edges and shapes from images."
        )
    if any(term in question for term in ("acute", "chronic", "leukemia types", "types of leukemia")):
        return (
            "The four major types represented in this project are ALL, AML, CLL, "
            "and CML. ALL and AML are acute forms; CLL and CML are chronic forms. "
            "They differ in the blood-cell lineage involved and require clinical "
            "testing to distinguish."
        )
    if "leukemia" in question or "blood cell" in question:
        return (
            "Leukemia is a group of cancers that begin in blood-forming tissues and "
            "affect the production or function of blood cells. Only qualified "
            "healthcare professionals can assess an individual's health."
        )
    latest_user_message = next(
        (item.content for item in reversed(history) if item.role == "user"),
        None,
    )
    if latest_user_message:
        return get_educational_fallback(latest_user_message, [])
    return None
