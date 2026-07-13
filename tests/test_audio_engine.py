"""
Unit tests for AURORA's decision logic.

The pure logic (emotion thresholds, gain calibration, note gating, signatures)
is tested with numpy alone. Feature-extraction tests that need librosa are
skipped automatically when librosa is not installed.
"""

import numpy as np
import pytest

from audio_engine import (
    GainCalibrator, determine_emotion, brightness_level,
    confidence, moment_id, note_from_chroma, NOTE_NAMES,
)


# ---------------------------------------------------------------------- #
# Emotion heuristic: documented, deterministic behavior
# ---------------------------------------------------------------------- #
class TestEmotionMapping:
    def test_quiet_dark_is_calm(self):
        assert determine_emotion(rms_norm=0.05, centroid=800, rolloff=2000) == "calm"

    def test_quiet_bright_is_sad(self):
        assert determine_emotion(rms_norm=0.05, centroid=4500, rolloff=14000) == "sad"

    def test_medium_mid_brightness_is_happy(self):
        assert determine_emotion(rms_norm=0.3, centroid=2500, rolloff=7000) == "happy"

    def test_loud_dark_is_angry(self):
        assert determine_emotion(rms_norm=0.9, centroid=1200, rolloff=3000) == "angry"

    def test_loud_mid_is_excited(self):
        assert determine_emotion(rms_norm=0.9, centroid=3000, rolloff=8000) == "excited"

    def test_always_returns_known_label(self):
        labels = {"calm", "neutral", "sad", "happy", "excited", "angry"}
        rng = np.random.default_rng(42)
        for _ in range(200):
            e = determine_emotion(
                rms_norm=float(rng.uniform(0, 1)),
                centroid=float(rng.uniform(0, 10000)),
                rolloff=float(rng.uniform(0, 20000)),
            )
            assert e in labels


# ---------------------------------------------------------------------- #
# Gain calibration: device-independent loudness
# ---------------------------------------------------------------------- #
class TestGainCalibrator:
    def test_output_bounded_0_1(self):
        g = GainCalibrator()
        for rms in [0.0, 0.001, 0.5, 3.0, 100.0]:
            assert 0.0 <= g.normalize(rms) <= 1.0

    def test_same_relative_dynamics_across_hardware_gain(self):
        """A quiet mic (x0.1) and a hot mic (x10) playing the same passage
        should converge to similar normalized loudness."""
        passage = [0.2, 0.5, 1.0, 0.5, 0.8, 1.0, 0.6]
        outs = []
        for gain in (0.1, 10.0):
            g = GainCalibrator()
            outs.append([g.normalize(v * gain) for v in passage])
        # after the peak is learned (index >= 2), curves should match closely
        for a, b in zip(outs[0][2:], outs[1][2:]):
            assert abs(a - b) < 0.05

    def test_silence_does_not_divide_by_zero(self):
        g = GainCalibrator()
        assert g.normalize(0.0) == 0.0

    def test_peak_decays_toward_floor(self):
        g = GainCalibrator(decay=0.9, floor=0.02)
        g.normalize(1.0)                      # learn a loud peak
        for _ in range(500):
            g.normalize(0.0)                  # long silence
        assert g.peak == pytest.approx(0.02)  # decayed to floor, not below


# ---------------------------------------------------------------------- #
# Note gating: no fake notes on noise
# ---------------------------------------------------------------------- #
class TestNoteGating:
    def test_no_pitch_returns_dash(self):
        profile = np.random.default_rng(1).random(12)
        assert note_from_chroma(profile, pitch_hz=0.0) == "-"

    def test_dominant_bin_maps_to_note_name(self):
        for i, name in enumerate(NOTE_NAMES):
            profile = np.zeros(12)
            profile[i] = 1.0
            assert note_from_chroma(profile, pitch_hz=440.0) == name


# ---------------------------------------------------------------------- #
# Signatures & misc
# ---------------------------------------------------------------------- #
class TestSignatures:
    def test_moment_id_in_range_and_deterministic(self):
        a = moment_id(1234.5, 6789.0, 440.0, -12.3)
        b = moment_id(1234.5, 6789.0, 440.0, -12.3)
        assert a == b and 0 <= a < 100000

    def test_different_moments_differ(self):
        a = moment_id(1234.5, 6789.0, 440.0, -12.3)
        b = moment_id(1234.6, 6789.0, 440.0, -12.3)
        assert a != b

    def test_confidence_bounded(self):
        assert 0.0 <= confidence(0.0, 0.0, 0.0) <= 1.0
        assert 0.0 <= confidence(10.0, 99999.0, 99999.0) <= 1.0

    def test_brightness_levels_ordered(self):
        assert brightness_level(500) == "very_dark"
        assert brightness_level(2000) == "dark"
        assert brightness_level(3000) == "neutral"
        assert brightness_level(5000) == "bright"
        assert brightness_level(9000) == "very_bright"


# ---------------------------------------------------------------------- #
# Feature extraction (needs librosa; skipped when unavailable)
# ---------------------------------------------------------------------- #
class TestFeatureExtraction:
    def test_a440_sine_detected(self):
        librosa = pytest.importorskip("librosa")  # noqa: F841
        from audio_engine import EmotionAnalyzer
        from config import SAMPLE_RATE, CHUNK_SIZE

        t = np.arange(CHUNK_SIZE) / SAMPLE_RATE
        audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        result = EmotionAnalyzer().analyze(audio)

        assert result["note"] == "A"
        assert abs(result["pitch_hz"] - 440.0) < 15.0       # yin within ~15 Hz
        assert result["emotion"] in {"calm", "neutral", "sad", "happy", "excited", "angry"}
        assert 0 <= result["moment_id"] < 100000
