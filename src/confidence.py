"""Voice confidence meter from raw audio (Task 13).

Scores how *confident* a spoken answer sounds from the raw waveform itself —
pitch variability, volume steadiness, and overall speaking energy, via
librosa — and combines them into a 0-100 ``confidence_score`` with a short,
plain-language explanation.

This is a signal-processing heuristic over acoustic properties, not a
trained model: unlike the ML fluency score (Task 6), which is a classifier
fit on labeled utterances, nothing here was fit to data. It is meant as a
rough, indicative coaching signal to practice against, same honesty as the
Task 11 body language heuristics.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src import config


class VoiceConfidenceError(RuntimeError):
    """Raised when audio can't be loaded or contains no detectable voice."""


@dataclass(frozen=True)
class VoiceConfidenceResult:
    """Heuristic voice-confidence signal for one spoken answer.

    Attributes:
        pitch_variability: 0-100 sub-score for pitch (f0) variation. Too
            flat (monotone) or too erratic both score lower than a natural
            amount of pitch movement — both commonly read as nervous.
        volume_steadiness: 0-100 sub-score for how consistent loudness
            (RMS energy) is across the answer; a shaky, wavering volume
            scores lower.
        speaking_energy: 0-100 sub-score for overall loudness, normalized
            against a reference level; a too-quiet recording scores lower.
        confidence_score: Combined 0-100 score (the mean of the three
            sub-scores above).
        summary: A short, human-readable explanation.
    """

    pitch_variability: float
    volume_steadiness: float
    speaking_energy: float
    confidence_score: float
    summary: str


def _load_waveform(audio_path_or_bytes: bytes | str | Path, sr: int) -> np.ndarray:
    """Decode audio to a mono waveform at the given sample rate.

    Args:
        audio_path_or_bytes: Raw audio bytes, or a path to an audio file.
        sr: Target sample rate to resample to.

    Returns:
        A 1-D array of mono audio samples.

    Raises:
        VoiceConfidenceError: If the audio can't be decoded.
    """
    source = (
        io.BytesIO(audio_path_or_bytes)
        if isinstance(audio_path_or_bytes, bytes)
        else audio_path_or_bytes
    )
    try:
        y, _ = librosa.load(source, sr=sr, mono=True)
    except Exception as exc:
        raise VoiceConfidenceError(f"Could not decode audio for confidence analysis: {exc}") from exc
    return y


def _pitch_variability_semitones(y: np.ndarray, sr: int) -> float | None:
    """Standard deviation of voiced f0 frames, in semitones.

    Measured relative to the answer's own median pitch (a log-frequency,
    i.e. perceptual, scale) so it isn't biased by a speaker's absolute
    vocal register — only how much it moves around.

    Args:
        y: Mono audio samples.
        sr: Sample rate of ``y``.

    Returns:
        The semitone std-dev of voiced frames, or ``None`` if no voiced
        pitch could be detected (e.g. silence or pure noise).
    """
    f0, voiced_flag, _voiced_probs = librosa.pyin(
        y,
        fmin=config.CONFIDENCE_PITCH_FMIN_HZ,
        fmax=config.CONFIDENCE_PITCH_FMAX_HZ,
        sr=sr,
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if voiced_f0.size < 2:
        return None

    median_f0 = float(np.median(voiced_f0))
    semitones = 12 * np.log2(voiced_f0 / median_f0)
    return float(np.std(semitones))


# Window/hop for the RMS energy envelope used for volume scoring, deliberately
# coarser than a typical analysis frame (~2048 samples/~128ms): a window this
# wide smooths over ordinary syllable-to-syllable amplitude texture (vowels
# are naturally louder than consonants) so "steadiness" reflects the voice's
# overall loudness arc, not that normal texture.
_ENERGY_WINDOW_SECONDS = 0.5
_ENERGY_HOP_SECONDS = 0.25
# Windows quieter than this fraction of the loudest window are treated as
# inter-word/sentence silence and excluded, so pauses between words don't
# get scored as "shaky" or "quiet" volume.
_SILENCE_WINDOW_RATIO = 0.1


def _speech_active_energy_windows(y: np.ndarray, sr: int) -> np.ndarray:
    """RMS energy over coarse windows, with near-silent windows dropped.

    Args:
        y: Mono audio samples.
        sr: Sample rate of ``y``.

    Returns:
        RMS energy per window, restricted to windows louder than
        ``_SILENCE_WINDOW_RATIO`` of the loudest window (or every window, if
        that leaves nothing — e.g. uniformly quiet audio).
    """
    frame_length = max(int(_ENERGY_WINDOW_SECONDS * sr), 1)
    hop_length = max(int(_ENERGY_HOP_SECONDS * sr), 1)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    threshold = _SILENCE_WINDOW_RATIO * float(np.max(rms))
    active = rms[rms > threshold]
    return active if active.size else rms


def _score_pitch_variability(semitone_std: float) -> float:
    """Score pitch variability against the configured ideal band.

    Args:
        semitone_std: Std-dev of voiced f0, in semitones; see
            :func:`_pitch_variability_semitones`.

    Returns:
        A 0-100 score: 100 within the ideal band; below it, a linear ramp
        from 0 (dead flat/monotone) up to 100 at the band's low edge; above
        it, a linear falloff back down to 0 (erratic) over
        ``config.PITCH_VARIABILITY_FALLOFF_SEMITONES``.
    """
    low = config.PITCH_VARIABILITY_IDEAL_MIN_SEMITONES
    high = config.PITCH_VARIABILITY_IDEAL_MAX_SEMITONES
    falloff = config.PITCH_VARIABILITY_FALLOFF_SEMITONES

    if low <= semitone_std <= high:
        return 100.0
    if semitone_std < low:
        return max(0.0, (semitone_std / low) * 100.0)
    return max(0.0, 100.0 - ((semitone_std - high) / falloff) * 100.0)


def _score_volume_steadiness(rms: np.ndarray) -> float:
    """Score how consistent RMS energy is across the answer's active speech.

    Args:
        rms: Per-window RMS energy values, e.g. from
            :func:`_speech_active_energy_windows`.

    Returns:
        A 0-100 score: 100 for perfectly steady volume, falling off
        linearly as the coefficient of variation (std/mean) rises, down to
        0 at ``config.VOLUME_STEADINESS_CV_HIGH``.
    """
    mean_rms = float(np.mean(rms))
    if mean_rms <= 0:
        return 0.0
    coefficient_of_variation = float(np.std(rms)) / mean_rms
    return max(0.0, 100.0 - (coefficient_of_variation / config.VOLUME_STEADINESS_CV_HIGH) * 100.0)


def _score_speaking_energy(rms: np.ndarray) -> float:
    """Score overall loudness, normalized against a reference level.

    Args:
        rms: Per-window RMS energy values, e.g. from
            :func:`_speech_active_energy_windows`.

    Returns:
        A 0-100 score: 0 for silence, 100 at/above
        ``config.SPEAKING_ENERGY_REFERENCE_RMS``, linear in between.
    """
    mean_rms = float(np.mean(rms))
    return min(100.0, (mean_rms / config.SPEAKING_ENERGY_REFERENCE_RMS) * 100.0)


def _build_summary(
    pitch_variability: float, volume_steadiness: float, speaking_energy: float
) -> str:
    """Turn the three sub-scores into a short, plain-language explanation.

    Args:
        pitch_variability: As in :class:`VoiceConfidenceResult`.
        volume_steadiness: As in :class:`VoiceConfidenceResult`.
        speaking_energy: As in :class:`VoiceConfidenceResult`.

    Returns:
        One or two short coaching sentences.
    """
    if pitch_variability >= 70:
        pitch_sentence = "Your pitch had a natural, confident variation."
    elif pitch_variability >= 40:
        pitch_sentence = "Your pitch was a little flat or uneven — try varying your tone more."
    else:
        pitch_sentence = "Your pitch sounded quite monotone or erratic — a sign of nerves."

    if volume_steadiness >= 70:
        volume_sentence = "Your voice was steady and clear"
    elif volume_steadiness >= 40:
        volume_sentence = "Your volume wavered a bit"
    else:
        volume_sentence = "Your volume was quite shaky"

    if speaking_energy >= 70:
        energy_clause = "."
    elif speaking_energy >= 40:
        energy_clause = ", and a bit on the quiet side — try projecting a little more."
    else:
        energy_clause = ", but quite quiet — try projecting more."

    return f"{pitch_sentence} {volume_sentence}{energy_clause}"


def analyze_voice_confidence(
    audio_path_or_bytes: bytes | str | Path, sr: int = config.CONFIDENCE_SAMPLE_RATE
) -> VoiceConfidenceResult:
    """Score how confident a spoken answer sounds from its raw audio.

    Args:
        audio_path_or_bytes: Raw audio bytes, or a path to an audio file,
            in any format librosa/soundfile can decode (wav, mp3, m4a, ...).
        sr: Sample rate to analyze at. Defaults to
            ``config.CONFIDENCE_SAMPLE_RATE``.

    Returns:
        The combined :class:`VoiceConfidenceResult`.

    Raises:
        VoiceConfidenceError: If the audio can't be decoded, or no voiced
            pitch could be detected (e.g. silence or pure noise).
    """
    y = _load_waveform(audio_path_or_bytes, sr)
    if y.size == 0:
        raise VoiceConfidenceError("Audio contains no samples to analyze.")

    semitone_std = _pitch_variability_semitones(y, sr)
    if semitone_std is None:
        raise VoiceConfidenceError("No voiced speech could be detected in this audio.")

    active_rms = _speech_active_energy_windows(y, sr)

    pitch_variability = round(_score_pitch_variability(semitone_std), 1)
    volume_steadiness = round(_score_volume_steadiness(active_rms), 1)
    speaking_energy = round(_score_speaking_energy(active_rms), 1)
    confidence_score = round((pitch_variability + volume_steadiness + speaking_energy) / 3, 1)

    summary = _build_summary(pitch_variability, volume_steadiness, speaking_energy)

    return VoiceConfidenceResult(
        pitch_variability=pitch_variability,
        volume_steadiness=volume_steadiness,
        speaking_energy=speaking_energy,
        confidence_score=confidence_score,
        summary=summary,
    )
