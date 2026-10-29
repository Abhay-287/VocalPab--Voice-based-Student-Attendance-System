
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from audio_utils import load_audio, extract_mfcc, extract_logmel

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


@lru_cache(maxsize=1)
def load_models():
    """Load the trained models once and reuse them."""
    svm_path = MODELS_DIR / "svm_model.joblib"
    cnn_path = MODELS_DIR / "cnn_model.keras"
    labels_path = MODELS_DIR / "label_encoder.json"

    for path in (svm_path, cnn_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(f"Required model file not found: {path.name}")

    svm_model = joblib.load(svm_path)
    cnn_model = tf.keras.models.load_model(cnn_path)

    with open(labels_path, "r", encoding="utf-8") as file:
        labels = json.load(file)

    return svm_model, cnn_model, labels


def predict_voice(audio_bytes, model_name="SVM"):
    """
    Identify a speaker from audio bytes.

    Returns the predicted student ID and the model's confidence score.
    """
    svm_model, cnn_model, labels = load_models()

    audio = load_audio(audio_bytes)

    if model_name.upper() == "SVM":
        features = extract_mfcc(audio).reshape(1, -1)

        probabilities = svm_model.predict_proba(features)[0]
        predicted_index = int(svm_model.predict(features)[0])
        confidence = float(probabilities.max())

    elif model_name.upper() == "CNN":
        features = extract_logmel(audio)
        features = np.expand_dims(features, axis=0)

        probabilities = cnn_model.predict(features, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

    else:
        raise ValueError("Model must be either 'SVM' or 'CNN'.")

    if predicted_index < 0 or predicted_index >= len(labels):
        raise ValueError("The model returned an unknown student class.")

    return {
        "student_id": labels[predicted_index],
        "confidence": confidence,
        "model": model_name.upper(),
    }
