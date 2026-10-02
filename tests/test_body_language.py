"""Tests for webcam body language coaching.

Both MediaPipe landmarkers are mocked throughout (via ``face_landmarker``/
``pose_landmarker`` injection) so these tests need neither a real webcam nor
a network download of the MediaPipe model bundles.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src import config
from src.body_language import (
    BodyLanguageError,
    BodyLanguageResult,
    _head_pose_degrees,
    _posture_score_for_frame,
    analyze_frames,
    analyze_video,
)

_BLANK_FRAME = np.zeros((20, 20, 3), dtype=np.uint8)


class _Landmark:
    """Minimal stand-in for MediaPipe's NormalizedLandmark."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def _face_result(yaw_degrees: float = 0.0, pitch_degrees: float = 0.0) -> MagicMock:
    """A fake FaceLandmarkerResult with a rotation matrix for the given angles."""
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    rot_yaw = np.array(
        [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]]
    )
    rot_pitch = np.array(
        [[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rot_pitch @ rot_yaw
    result = MagicMock()
    result.facial_transformation_matrixes = [matrix]
    return result


def _no_face_result() -> MagicMock:
    result = MagicMock()
    result.facial_transformation_matrixes = []
    return result


def _pose_result(
    nose=(0.5, 0.3), left_shoulder=(0.35, 0.5), right_shoulder=(0.65, 0.5)
) -> MagicMock:
    """A fake PoseLandmarkerResult with nose/shoulder landmarks set."""
    landmarks = [None] * 13
    landmarks[0] = _Landmark(*nose)
    landmarks[11] = _Landmark(*left_shoulder)
    landmarks[12] = _Landmark(*right_shoulder)
    result = MagicMock()
    result.pose_landmarks = [landmarks]
    return result


def _no_pose_result() -> MagicMock:
    result = MagicMock()
    result.pose_landmarks = []
    return result


def _fake_landmarker(results: list) -> MagicMock:
    """A fake Face/Pose Landmarker whose .detect() returns results in order."""
    landmarker = MagicMock()
    landmarker.detect.side_effect = results
    return landmarker


# --- head pose geometry ----------------------------------------------------------


def test_head_pose_degrees_none_when_no_face() -> None:
    """No detected face should report no head pose."""
    assert _head_pose_degrees(_no_face_result()) is None


def test_head_pose_degrees_matches_known_rotation_angle() -> None:
    """A synthetic 30-degree yaw rotation should decode back to ~30 degrees."""
    yaw, pitch = _head_pose_degrees(_face_result(yaw_degrees=30.0))

    assert yaw == pytest.approx(30.0, abs=0.5)
    assert pitch == pytest.approx(0.0, abs=0.5)


def test_head_pose_degrees_identity_is_near_zero() -> None:
    """Looking straight at the camera should decode to ~0 yaw and pitch."""
    yaw, pitch = _head_pose_degrees(_face_result())

    assert yaw == pytest.approx(0.0, abs=0.01)
    assert pitch == pytest.approx(0.0, abs=0.01)


# --- posture geometry --------------------------------------------------------------


def test_posture_score_high_for_level_upright_shoulders() -> None:
    """Level shoulders with a normal neck ratio should score well."""
    score = _posture_score_for_frame(_pose_result())

    assert score > 80.0


def test_posture_score_low_for_tilted_shoulders() -> None:
    """Noticeably tilted shoulders should score lower than level ones."""
    level_score = _posture_score_for_frame(_pose_result())
    tilted_score = _posture_score_for_frame(
        _pose_result(left_shoulder=(0.35, 0.65), right_shoulder=(0.65, 0.35))
    )

    assert tilted_score < level_score


def test_posture_score_none_when_no_pose_detected() -> None:
    """No detected pose should report no posture score."""
    assert _posture_score_for_frame(_no_pose_result()) is None


# --- analyze_frames ----------------------------------------------------------------


def test_analyze_frames_rejects_empty_sequence() -> None:
    """An empty frame list should fail loudly rather than divide by zero."""
    with pytest.raises(BodyLanguageError, match="No frames"):
        analyze_frames([], face_landmarker=MagicMock(), pose_landmarker=MagicMock())


def test_analyze_frames_rejects_non_positive_fps() -> None:
    """A zero or negative fps makes "seconds away" meaningless."""
    with pytest.raises(ValueError, match="fps"):
        analyze_frames(
            [_BLANK_FRAME],
            fps=0.0,
            face_landmarker=MagicMock(),
            pose_landmarker=MagicMock(),
        )


def test_analyze_frames_full_eye_contact() -> None:
    """All frames forward-facing should give a perfect eye contact ratio."""
    frames = [_BLANK_FRAME] * 3
    face = _fake_landmarker([_face_result()] * 3)
    pose = _fake_landmarker([_pose_result()] * 3)

    result = analyze_frames(frames, fps=5.0, face_landmarker=face, pose_landmarker=pose)

    assert isinstance(result, BodyLanguageResult)
    assert result.eye_contact_ratio == pytest.approx(1.0)
    assert result.looking_away_count == 0
    assert "good eye contact" in result.summary.lower()


def test_analyze_frames_no_face_detected_gives_zero_ratio_and_says_so() -> None:
    """No face in any frame should report zero eye contact, not a crash."""
    frames = [_BLANK_FRAME] * 2
    face = _fake_landmarker([_no_face_result()] * 2)
    pose = _fake_landmarker([_no_pose_result()] * 2)

    result = analyze_frames(frames, fps=5.0, face_landmarker=face, pose_landmarker=pose)

    assert result.eye_contact_ratio == 0.0
    assert result.posture_score == 0.0
    assert "couldn't detect your face" in result.summary.lower()


def test_analyze_frames_counts_looking_away_once_per_streak() -> None:
    """A single long away-streak should count once, not once per frame."""
    # fps=2 -> 0.5s/frame; 4 consecutive "away" frames = 2.0s, one streak.
    frames = [_BLANK_FRAME] * 4
    face = _fake_landmarker([_face_result(yaw_degrees=90.0)] * 4)
    pose = _fake_landmarker([_pose_result()] * 4)

    result = analyze_frames(frames, fps=2.0, face_landmarker=face, pose_landmarker=pose)

    assert result.looking_away_count == 1
    assert result.eye_contact_ratio == 0.0


def test_analyze_frames_counts_two_separate_looking_away_streaks() -> None:
    """Two away-streaks separated by a forward-facing frame should count as two."""
    # fps=2 -> 0.5s/frame; each away streak below is 1.0s, crossing the 1.0s minimum.
    away = _face_result(yaw_degrees=90.0)
    forward = _face_result()
    face_sequence = [away, away, forward, away, away]
    face = _fake_landmarker(face_sequence)
    pose = _fake_landmarker([_pose_result()] * len(face_sequence))

    result = analyze_frames(
        [_BLANK_FRAME] * len(face_sequence), fps=2.0, face_landmarker=face, pose_landmarker=pose
    )

    assert result.looking_away_count == 2


def test_analyze_frames_brief_glance_away_not_counted() -> None:
    """A streak shorter than the configured minimum shouldn't be counted."""
    # fps=10 -> 0.1s/frame; a single away frame is only 0.1s, well under 1.0s.
    frames = [_BLANK_FRAME] * 3
    face_sequence = [_face_result(), _face_result(yaw_degrees=90.0), _face_result()]
    face = _fake_landmarker(face_sequence)
    pose = _fake_landmarker([_pose_result()] * 3)

    result = analyze_frames(frames, fps=10.0, face_landmarker=face, pose_landmarker=pose)

    assert result.looking_away_count == 0


def test_analyze_frames_default_landmarkers_are_lazily_loaded(monkeypatch) -> None:
    """Omitting face_landmarker/pose_landmarker should fall back to the cached loaders."""
    from src import body_language

    fake_face = _fake_landmarker([_face_result()])
    fake_pose = _fake_landmarker([_pose_result()])
    monkeypatch.setattr(body_language, "_load_face_landmarker", lambda: fake_face)
    monkeypatch.setattr(body_language, "_load_pose_landmarker", lambda: fake_pose)

    result = body_language.analyze_frames([_BLANK_FRAME], fps=5.0)

    assert result.eye_contact_ratio == pytest.approx(1.0)


# --- analyze_video -----------------------------------------------------------------


def test_analyze_video_accepts_a_raw_frame_sequence() -> None:
    """Passing frames directly (no file) should skip video decoding entirely."""
    frames = [_BLANK_FRAME] * 2
    face = _fake_landmarker([_face_result()] * 2)
    pose = _fake_landmarker([_pose_result()] * 2)

    result = analyze_video(frames, fps=5.0, face_landmarker=face, pose_landmarker=pose)

    assert result.eye_contact_ratio == pytest.approx(1.0)


def test_analyze_video_reads_a_short_synthetic_video_file(tmp_path) -> None:
    """A short synthetic video file on disk should be decoded frame-by-frame."""
    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"XVID"), 5.0, (20, 20)
    )
    try:
        for _ in range(5):
            writer.write(_BLANK_FRAME)
    finally:
        writer.release()

    face = _fake_landmarker([_face_result()] * 5)
    pose = _fake_landmarker([_pose_result()] * 5)

    result = analyze_video(video_path, face_landmarker=face, pose_landmarker=pose)

    assert isinstance(result, BodyLanguageResult)
    assert face.detect.call_count == 5
    assert result.eye_contact_ratio == pytest.approx(1.0)


def test_analyze_video_raises_for_missing_file(tmp_path) -> None:
    """A nonexistent video path should fail with a clear error, not a cryptic one."""
    with pytest.raises(BodyLanguageError, match="Could not open"):
        analyze_video(tmp_path / "does_not_exist.avi")


# --- config thresholds are respected -----------------------------------------------


def test_eye_contact_threshold_is_configurable(monkeypatch) -> None:
    """A wider yaw threshold should classify a moderate turn as forward-facing."""
    monkeypatch.setattr(config, "EYE_CONTACT_YAW_THRESHOLD_DEGREES", 45.0)
    face = _fake_landmarker([_face_result(yaw_degrees=30.0)])
    pose = _fake_landmarker([_pose_result()])

    result = analyze_frames([_BLANK_FRAME], fps=5.0, face_landmarker=face, pose_landmarker=pose)

    assert result.eye_contact_ratio == pytest.approx(1.0)
