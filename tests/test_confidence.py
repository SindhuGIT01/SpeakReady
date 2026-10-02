"""Tests for the voice confidence meter (Task 13).

No API key or mocking needed — ``analyze_voice_confidence`` works entirely
on local audio via librosa. Tests build small synthetic tones (with a
controlled amount of pitch/volume variation) so the sub-scores are
predictable, plus one smoke test against the real sample fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.confidence import VoiceConfidenceError, analyze_voice_confidence

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_ANSWER_MP3 = FIXTURES_DIR / "sample_answer.mp3"

_SR = 16000
_DURATION_S = 3.0


def _tone(
    duration_s: float = _DURATION_S,
    base_freq_hz: float = 150.0,
    amplitude: float = 0.3,
    vibrato_semitones: np.ndarray | None = None,
    sr: int = _SR,
) -> np.ndarray:
    """A synthetic sine tone, optionally with a time-varying pitch.

    ``vibrato_semitones`` (one value per sample) offsets the instantaneous
    frequency from ``base_freq_hz`` in semitones, so the resulting pitch
    standard deviation is directly controllable for test purposes.
    """
    n = int(duration_s * sr)
    if vibrato_semitones is None:
        freq = np.full(n, base_freq_hz)
    else:
        freq = base_freq_hz * (2.0 ** (vibrato_semitones / 12.0))
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return (amplitude * np.sin(phase)).astype(np.float32)


def _write_wav(tmp_path: Path, name: str, y: np.ndarray, sr: int = _SR) -> Path:
    path = tmp_path / name
    sf.write(path, y, sr)
    return path


# --- pitch variability -----------------------------------------------------------


def test_monotone_tone_scores_low_pitch_variability(tmp_path) -> None:
    """A perfectly flat pitch should score low (dead flat reads as monotone)."""
    path = _write_wav(tmp_path, "monotone.wav", _tone())

    result = analyze_voice_confidence(path)

    assert result.pitch_variability < 30.0


def test_natural_pitch_variation_scores_full_marks(tmp_path) -> None:
    """A moderate, natural amount of pitch wobble should score 100."""
    n = int(_DURATION_S * _SR)
    t = np.arange(n) / _SR
    vibrato = 3.0 * np.sin(2 * np.pi * 2.0 * t)  # ~3-semitone swings, slow enough to track
    path = _write_wav(tmp_path, "natural.wav", _tone(vibrato_semitones=vibrato))

    result = analyze_voice_confidence(path)

    assert result.pitch_variability == 100.0


def test_erratic_pitch_scores_low(tmp_path) -> None:
    """Wildly, abruptly swinging pitch should score low (erratic, not just flat)."""
    rng = np.random.default_rng(0)
    n = int(_DURATION_S * _SR)
    n_segments = 10
    vibrato = np.repeat(rng.uniform(-24, 24, n_segments), n // n_segments)
    vibrato = np.pad(vibrato, (0, n - vibrato.size), mode="edge")
    path = _write_wav(tmp_path, "erratic.wav", _tone(vibrato_semitones=vibrato))

    result = analyze_voice_confidence(path)

    assert result.pitch_variability < 50.0


# --- volume steadiness -------------------------------------------------------------


def test_steady_volume_scores_high_steadiness(tmp_path) -> None:
    """Constant amplitude should score clearly higher than a wavering one."""
    path = _write_wav(tmp_path, "steady_volume.wav", _tone(amplitude=0.3))

    result = analyze_voice_confidence(path)

    assert result.volume_steadiness > 80.0


def test_wavering_volume_scores_lower_steadiness(tmp_path) -> None:
    """A slow, strongly modulated amplitude envelope should score lower steadiness.

    The envelope swings slowly (0.5 Hz, a ~2s period) relative to the ~0.5s
    energy window :mod:`src.confidence` scores volume over — fast flutter
    within that window gets smoothed out on purpose (it's just phoneme
    texture), but a swell/fade over multiple seconds is genuine wavering.
    """
    n = int(_DURATION_S * _SR)
    t = np.arange(n) / _SR
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    y = (_tone(amplitude=0.3) * envelope).astype(np.float32)
    path = _write_wav(tmp_path, "wavering.wav", y)

    result = analyze_voice_confidence(path)

    assert result.volume_steadiness < 70.0


# --- speaking energy ---------------------------------------------------------------


def test_quiet_audio_scores_low_speaking_energy(tmp_path) -> None:
    """A very quiet recording should score low on speaking energy."""
    path = _write_wav(tmp_path, "quiet.wav", _tone(amplitude=0.002))

    result = analyze_voice_confidence(path)

    assert result.speaking_energy < 20.0


def test_loud_audio_scores_full_speaking_energy(tmp_path) -> None:
    """A comfortably loud recording should hit the full speaking energy score."""
    path = _write_wav(tmp_path, "loud.wav", _tone(amplitude=0.3))

    result = analyze_voice_confidence(path)

    assert result.speaking_energy == 100.0


# --- combined score / input handling ------------------------------------------------


def test_confidence_score_is_mean_of_subscores(tmp_path) -> None:
    """The combined score should be the rounded mean of the three sub-scores."""
    path = _write_wav(tmp_path, "mean_check.wav", _tone())

    result = analyze_voice_confidence(path)

    expected = round(
        (result.pitch_variability + result.volume_steadiness + result.speaking_energy) / 3, 1
    )
    assert result.confidence_score == expected


def test_accepts_raw_bytes() -> None:
    """Audio passed as raw bytes (not a path) should work the same way."""
    result = analyze_voice_confidence(SAMPLE_ANSWER_MP3.read_bytes())

    assert 0.0 <= result.confidence_score <= 100.0


def test_silence_raises_voice_confidence_error(tmp_path) -> None:
    """Pure silence has no detectable pitch and should raise, not silently score."""
    path = _write_wav(tmp_path, "silence.wav", np.zeros(int(2.0 * _SR), dtype=np.float32))

    with pytest.raises(VoiceConfidenceError, match="No voiced speech"):
        analyze_voice_confidence(path)


def test_unreadable_audio_raises_voice_confidence_error(tmp_path) -> None:
    """Garbage bytes that aren't real audio should raise a clear error, not crash."""
    path = tmp_path / "not_audio.wav"
    path.write_bytes(b"this is not an audio file")

    with pytest.raises(VoiceConfidenceError):
        analyze_voice_confidence(path)


def test_sample_answer_fixture_produces_a_result() -> None:
    """Smoke test against the real sample fixture (human speech, not a sine tone)."""
    result = analyze_voice_confidence(SAMPLE_ANSWER_MP3)

    assert 0.0 <= result.confidence_score <= 100.0
    assert result.summary
