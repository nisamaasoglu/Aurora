"""
AURORA - Audio Analysis Engine
Extracts meaningful features from the live audio stream to feed the visuals.

Extracted features:
  - rms        : loudness (raw) + rms_norm (device-independent, auto-calibrated)
  - centroid   : spectral centroid (tone color / brightness)
  - rolloff    : spectral rolloff (harmonic vs. noise separation)
  - note       : dominant note (chroma), shown only when a pitch is present
  - pitch_hz   : real pitch via librosa.yin (optional)
  - mfcc_mean  : timbre characteristics (optional)
  - emotion    : rule-based (heuristic) emotion label
  - moment_id  : a one-time, non-repeatable signature for this instant

Design notes:
  * librosa is imported lazily so the pure decision logic (emotion thresholds,
    gain calibration, signatures) stays unit-testable with numpy alone.
  * Loudness is normalized against a slowly-decaying running peak, so the
    emotion mapping behaves the same across microphones with different gain.
  * The emotion assignment is NOT a trained model; it is a transparent set of
    hand-tuned thresholds over the features.
"""

import hashlib
import numpy as np

from config import (
    SAMPLE_RATE, CHUNK_SIZE, ENABLE_PITCH, ENABLE_MFCC,
    PITCH_FMIN, PITCH_FMAX,
)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------- #
# Pure logic (unit-testable without librosa)
# ---------------------------------------------------------------------- #
class GainCalibrator:
    """Tracks a slowly-decaying running peak of RMS so loudness can be
    expressed on a 0..1 scale regardless of microphone hardware gain."""

    def __init__(self, decay: float = 0.999, floor: float = 0.02):
        self.decay = decay
        self.floor = floor      # minimum peak; avoids division blow-up in silence
        self.peak = floor

    def normalize(self, rms: float) -> float:
        self.peak = max(rms, self.peak * self.decay, self.floor)
        return min(rms / self.peak, 1.0)


def determine_emotion(rms_norm: float, centroid: float, rolloff: float) -> str:
    """Map normalized loudness + spectral brightness to an emotion label.
    Transparent heuristic; thresholds are documented by the unit tests."""
    centroid_norm = min(centroid / 5000, 1.0)
    rolloff_norm = min(rolloff / 15000, 1.0)
    brightness = (centroid_norm + rolloff_norm) / 2

    if rms_norm < 0.15:
        if brightness < 0.3:
            return "calm"
        if brightness < 0.6:
            return "neutral"
        return "sad"
    if rms_norm < 0.5:
        if brightness < 0.3:
            return "neutral"
        if brightness < 0.6:
            return "happy"
        return "excited"
    if brightness < 0.4:
        return "angry"
    if brightness < 0.7:
        return "excited"
    return "happy"


def brightness_level(centroid: float) -> str:
    if centroid < 1500:
        return "very_dark"
    if centroid < 2500:
        return "dark"
    if centroid < 4000:
        return "neutral"
    if centroid < 6000:
        return "bright"
    return "very_bright"


def confidence(rms: float, centroid: float, rolloff: float) -> float:
    a = min(rms * 50, 1.0)
    b = min(centroid / 3000, 1.0)
    c = min(rolloff / 10000, 1.0)
    return (a + b + c) / 3


def moment_id(centroid: float, rolloff: float, pitch_hz: float, mfcc_mean: float) -> int:
    """A signature for this instant: the same feature combination effectively
    never recurs. Ambient noise blends into the signal too, so every moment
    is unique. Range: 0..99999."""
    raw = f"{centroid:.4f}|{rolloff:.4f}|{pitch_hz:.4f}|{mfcc_mean:.4f}"
    return int(hashlib.sha256(raw.encode()).hexdigest(), 16) % 100000


def note_from_chroma(chroma_profile: np.ndarray, pitch_hz: float) -> str:
    """Dominant note from a 12-bin chroma profile. Returns '-' when no pitch
    is present, so noise doesn't get labeled with a fake note."""
    if pitch_hz <= 0:
        return "-"
    return NOTE_NAMES[int(np.argmax(chroma_profile))]


# ---------------------------------------------------------------------- #
# Analyzer (librosa-backed feature extraction + temporal smoothing)
# ---------------------------------------------------------------------- #
class EmotionAnalyzer:
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.rms_history: list = []
        self.centroid_history: list = []
        self.rolloff_history: list = []
        self.gain = GainCalibrator()

    def extract_features(self, audio: np.ndarray) -> dict:
        import librosa  # lazy: keeps pure logic importable without librosa

        rms = float(librosa.feature.rms(y=audio, frame_length=CHUNK_SIZE)[0, 0])
        centroid = float(
            librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, n_fft=CHUNK_SIZE)[0, 0]
        )
        rolloff = float(
            librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE, n_fft=CHUNK_SIZE)[0, 0]
        )
        chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE, n_fft=CHUNK_SIZE)
        chroma_profile = np.mean(chroma, axis=1)

        pitch_hz = 0.0
        if ENABLE_PITCH:
            try:
                f0 = librosa.yin(
                    audio, fmin=PITCH_FMIN, fmax=PITCH_FMAX,
                    sr=SAMPLE_RATE, frame_length=CHUNK_SIZE,
                )
                f0 = f0[np.isfinite(f0)]
                if f0.size:
                    pitch_hz = float(np.median(f0))
            except Exception:
                pitch_hz = 0.0

        mfcc_mean = 0.0
        if ENABLE_MFCC:
            try:
                mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=CHUNK_SIZE)
                mfcc_mean = float(np.mean(mfcc))
            except Exception:
                mfcc_mean = 0.0

        return {
            "rms": rms, "centroid": centroid, "rolloff": rolloff,
            "chroma_profile": chroma_profile,
            "pitch_hz": pitch_hz, "mfcc_mean": mfcc_mean,
        }

    def analyze(self, audio: np.ndarray) -> dict:
        f = self.extract_features(audio)

        self._push(self.rms_history, f["rms"])
        self._push(self.centroid_history, f["centroid"])
        self._push(self.rolloff_history, f["rolloff"])

        rms_norm = self.gain.normalize(float(np.mean(self.rms_history)))
        avg_centroid = float(np.mean(self.centroid_history))
        avg_rolloff = float(np.mean(self.rolloff_history))

        emotion = determine_emotion(rms_norm, avg_centroid, avg_rolloff)
        note = note_from_chroma(f["chroma_profile"], f["pitch_hz"])

        return {
            "emotion": emotion,
            "confidence": round(confidence(f["rms"], f["centroid"], f["rolloff"]), 2),
            "rms": round(f["rms"], 4),
            "rms_norm": round(rms_norm, 3),
            "centroid": round(f["centroid"], 2),
            "rolloff": round(f["rolloff"], 2),
            "note": note,
            "pitch_hz": round(f["pitch_hz"], 1),
            "mfcc_mean": round(f["mfcc_mean"], 2),
            "brightness": brightness_level(avg_centroid),
            "moment_id": moment_id(f["centroid"], f["rolloff"], f["pitch_hz"], f["mfcc_mean"]),
        }

    def _push(self, buf: list, value: float):
        buf.append(value)
        if len(buf) > self.history_size:
            buf.pop(0)
