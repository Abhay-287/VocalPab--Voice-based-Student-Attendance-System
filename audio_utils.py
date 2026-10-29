
import io

import librosa
import numpy as np

# Standard audio settings used throughout VocalPab
SAMPLE_RATE = 16000
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 20
N_MELS = 64


def load_audio(source):
    """
    Load a WAV/audio file or bytes from a microphone recording.
    Convert it to mono, resample to 16 kHz, and make it 3 seconds long.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    audio, _ = librosa.load(
        source,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The audio recording is empty.")

    # Pad short recordings or trim long ones to 3 seconds
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]

    return audio.astype(np.float32)


def extract_mfcc(audio):
    """
    Extract 40 features: mean and standard deviation
    of 20 MFCC coefficients. Used by the SVM model.
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=160
    )

    means = np.mean(mfcc, axis=1)
    stds = np.std(mfcc, axis=1)

    return np.concatenate([means, stds]).astype(np.float32)


def extract_logmel(audio):
    """
    Create a log-Mel spectrogram for the CNN model.
    This represents how audio energy changes across
    frequency bands over time.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=512,
        hop_length=160,
        n_mels=N_MELS,
        power=2.0
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = np.clip(logmel, -80, 0)
    logmel = (logmel + 80) / 80.0

    # Add a channel dimension for the CNN
    return logmel[..., np.newaxis].astype(np.float32)
