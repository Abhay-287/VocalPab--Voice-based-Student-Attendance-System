
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from audio_recorder_streamlit import audio_recorder

from audio_utils import load_audio
from predict_voice import predict_voice

# Project folders
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
STUDENTS_FILE = ROOT / "students.json"

DATASET_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="VocalPab",
    page_icon="🎙️",
    layout="centered"
)


def load_students():
    if not STUDENTS_FILE.exists():
        return []

    try:
        with open(STUDENTS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def save_students(students):
    with open(STUDENTS_FILE, "w", encoding="utf-8") as file:
        json.dump(students, file, indent=4)


def save_voice_sample(student_id, audio_bytes):
    # Validate that the recording can be processed
    audio = load_audio(audio_bytes)

    if len(audio) == 0:
        raise ValueError("The recording is empty.")

    student_folder = DATASET_DIR / student_id
    student_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = student_folder / f"sample_{timestamp}.wav"

    # Save the original WAV recording
    file_path.write_bytes(audio_bytes)

    return file_path


st.title("🎙️ VocalPab")
st.caption("Voice-Based Student Attendance System")
st.write(
    "Register students and collect voice samples for "
    "our Machine Learning and Deep Learning models."
)

students = load_students()

# Section 1: Student registration
st.header("1. Register a student")

with st.form("student_registration"):
    student_id = st.text_input(
        "Student ID",
        placeholder="e.g. student_001"
    ).strip()

    student_name = st.text_input(
        "Student name",
        placeholder="Enter full name"
    ).strip()

    register_clicked = st.form_submit_button("Register student")

if register_clicked:
    if not student_id or not student_name:
        st.error("Please enter both the student ID and name.")
    elif (
        not student_id.replace("_", "").replace("-", "").isalnum()
        or student_id in {".", ".."}
    ):
        st.error("Use only letters, numbers, underscores, or hyphens in the ID.")
    elif any(s["student_id"] == student_id for s in students):
        st.error("This student ID is already registered.")
    else:
        students.append({
            "student_id": student_id,
            "name": student_name
        })
        save_students(students)
        st.success(f"{student_name} registered successfully!")
        st.rerun()

# Refresh list after registration
students = load_students()

st.divider()

# Section 2: Collect voice samples
st.header("2. Collect voice samples")

if not students:
    st.info("Register a student above before collecting voice samples.")
else:
    student_options = {
        f'{s["name"]} ({s["student_id"]})': s["student_id"]
        for s in students
    }

    selected_label = st.selectbox(
        "Choose a student",
        list(student_options.keys())
    )
    selected_id = student_options[selected_label]

    st.write("Record a clear, natural voice sample of around 3 seconds.")
    st.caption(
        "For better training data, collect multiple samples using "
        "different sentences and recording sessions."
    )

    st.subheader("Option A — Record with microphone")

    recorded_audio = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#555555",
        icon_name="microphone",
        icon_size="2x"
    )

    if recorded_audio:
        st.audio(recorded_audio, format="audio/wav")

        if st.button("Save microphone recording", key="save_mic"):
            try:
                path = save_voice_sample(selected_id, recorded_audio)
                st.success(f"Recording saved: {path.name}")
            except Exception as error:
                st.error(f"Could not save recording: {error}")

    st.subheader("Option B — Upload a WAV file")

    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=["wav"],
        key="wav_upload"
    )

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        st.audio(uploaded_bytes, format="audio/wav")

        if st.button("Save uploaded recording", key="save_upload"):
            try:
                path = save_voice_sample(selected_id, uploaded_bytes)
                st.success(f"Recording saved: {path.name}")
            except Exception as error:
                st.error(f"Could not save recording: {error}")

st.divider()

# Section 3: Display collected data
st.header("3. Registered students")

students = load_students()

if students:
    for student in students:
        folder = DATASET_DIR / student["student_id"]
        sample_count = (
            len(list(folder.glob("*.wav"))) if folder.exists() else 0
        )

        st.write(
            f'**{student["name"]}** — {student["student_id"]} '
            f'— {sample_count} voice sample(s)'
        )
else:
    st.write("No students registered yet.")

st.caption(
    "Voice samples are stored locally. Collect recordings only with "
    "the speaker's informed consent."
)

st.divider()
st.header("4. Identify a speaker")
st.write(
    "Record a voice sample or upload a WAV file to test "
    "the trained speaker identification models."
)

model_choice = st.selectbox(
    "Choose the model",
    ["SVM", "CNN"],
    key="identification_model"
)

st.caption(
    "Use a clear recording of around 3 seconds. "
    "Model confidence is not a guarantee of correct identification."
)

st.subheader("Option A — Record a voice sample")

identification_audio = audio_recorder(
    text="Record voice for identification",
    recording_color="#e74c3c",
    neutral_color="#555555",
    icon_name="microphone",
    icon_size="2x",
    key="identify_mic"
)

if identification_audio:
    st.audio(identification_audio, format="audio/wav")

    if st.button("Identify recorded voice", key="identify_recorded"):
        try:
            result = predict_voice(identification_audio, model_choice)

            student_id = result["student_id"]
            confidence = result["confidence"]

            registered_students = load_students()
            student_names = {
                s["student_id"]: s["name"]
                for s in registered_students
            }

            st.success("Prediction completed")
            st.write("**Predicted student:**", student_names.get(
                student_id, "Name not found"
            ))
            st.write("**Student ID:**", student_id)
            st.write("**Model:**", result["model"])
            st.write(f"**Model confidence:** {confidence:.2%}")

        except Exception as error:
            st.error(f"Identification failed: {error}")


st.subheader("Option B — Upload a WAV file")

identification_file = st.file_uploader(
    "Choose a WAV file for identification",
    type=["wav"],
    key="identification_upload"
)

if identification_file is not None:
    identification_bytes = identification_file.getvalue()
    st.audio(identification_bytes, format="audio/wav")

    if st.button("Identify uploaded voice", key="identify_uploaded"):
        try:
            result = predict_voice(
                identification_bytes,
                model_choice
            )

            student_id = result["student_id"]
            confidence = result["confidence"]

            registered_students = load_students()
            student_names = {
                s["student_id"]: s["name"]
                for s in registered_students
            }

            st.success("Prediction completed")
            st.write("**Predicted student:**", student_names.get(
                student_id, "Name not found"
            ))
            st.write("**Student ID:**", student_id)
            st.write("**Model:**", result["model"])
            st.write(f"**Model confidence:** {confidence:.2%}")

        except Exception as error:
            st.error(f"Identification failed: {error}")
