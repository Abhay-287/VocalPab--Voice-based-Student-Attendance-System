
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from audio_utils import load_audio, extract_mfcc, extract_logmel


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_dataset():
    mfcc_features = []
    logmel_features = []
    labels = []

    student_folders = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    for folder in student_folders:
        wav_files = sorted(folder.glob("*.wav"))

        if not wav_files:
            print(f"Skipping {folder.name}: no WAV files found.")
            continue

        for wav_file in wav_files:
            try:
                audio = load_audio(str(wav_file))

                mfcc_features.append(extract_mfcc(audio))
                logmel_features.append(extract_logmel(audio))
                labels.append(folder.name)

                print(f"Loaded: {folder.name}/{wav_file.name}")

            except Exception as error:
                print(f"Could not read {wav_file.name}: {error}")

    if not labels:
        raise ValueError(
            "No valid WAV recordings found in the dataset folder."
        )

    X_mfcc = np.asarray(mfcc_features, dtype=np.float32)
    X_logmel = np.asarray(logmel_features, dtype=np.float32)
    y = np.asarray(labels)

    return X_mfcc, X_logmel, y


def main():
    print("\nLoading voice recordings...")
    X_mfcc, X_logmel, y = load_dataset()

    classes, counts = np.unique(y, return_counts=True)

    print("\nDataset summary:")
    for student, count in zip(classes, counts):
        print(f"{student}: {count} samples")

    if len(classes) < 2:
        raise ValueError(
            "At least two different students are needed for classification."
        )

    if np.min(counts) < 2:
        raise ValueError(
            "Each student needs at least two valid recordings "
            "for a stratified train/test split."
        )

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    indices = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    X_train = X_mfcc[train_idx]
    X_test = X_mfcc[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    # ---------------- SVM MODEL ----------------
    print("\nTraining SVM model...")

    svm_model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    )

    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    svm_accuracy = accuracy_score(y_test, svm_predictions)

    joblib.dump(svm_model, MODEL_DIR / "svm_model.joblib")

    print(f"SVM test accuracy: {svm_accuracy:.2%}")
    print(
        classification_report(
            y_test,
            svm_predictions,
            labels=np.arange(len(classes)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    # ---------------- CNN MODEL ----------------
    print("\nTraining CNN model...")

    X_cnn_train = X_logmel[train_idx]
    X_cnn_test = X_logmel[test_idx]

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    cnn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=X_logmel.shape[1:]),
        tf.keras.layers.Conv2D(8, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(classes), activation="softmax")
    ])

    cnn_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cnn_model.fit(
        X_cnn_train,
        y_train,
        epochs=10,
        batch_size=4,
        verbose=1
    )

    _, cnn_accuracy = cnn_model.evaluate(
        X_cnn_test,
        y_test,
        verbose=0
    )

    cnn_model.save(MODEL_DIR / "cnn_model.keras")

    # Save the label mapping used by both models.
    with open(MODEL_DIR / "label_encoder.json", "w", encoding="utf-8") as file:
        json.dump(encoder.classes_.tolist(), file, indent=4)

    # Save a basic comparison report.
    results = {
        "number_of_recordings": int(len(y)),
        "number_of_students": int(len(classes)),
        "svm_test_accuracy": float(svm_accuracy),
        "cnn_test_accuracy": float(cnn_accuracy),
        "note": (
            "Small-dataset results are preliminary and do not establish "
            "reliable real-world speaker recognition."
        )
    }

    with open(
        REPORT_DIR / "model_comparison.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    print("\nTraining finished.")
    print(f"SVM test accuracy: {svm_accuracy:.2%}")
    print(f"CNN test accuracy: {cnn_accuracy:.2%}")
    print(f"Models saved in: {MODEL_DIR}")
    print(f"Report saved in: {REPORT_DIR / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
