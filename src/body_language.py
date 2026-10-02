"""Webcam-based body language coaching (Task 11).

Heuristic, frame-by-frame analysis of a short webcam clip (or a bare list of
frames, e.g. snapshots captured in the Streamlit app) using MediaPipe's Face
Landmarker and Pose Landmarker: how often the candidate looks toward the
camera (``eye_contact_ratio``), how upright and level their shoulders are
(``posture_score``), and how many distinct times they looked away for more
than about a second (``looking_away_count``).

This is landmark geometry, not a trained model — see the README for the
accuracy caveats. Everything here is optional: callers that don't have a
webcam clip simply never call :func:`analyze_video`.
"""

from __future__ import annotations

import math
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.error import URLError

import cv2
import mediapipe as mp
import numpy as np

from src import config

# BlazePose landmark indices (33-point topology) used for posture.
_POSE_NOSE = 0
_POSE_LEFT_SHOULDER = 11
_POSE_RIGHT_SHOULDER = 12


class BodyLanguageError(RuntimeError):
    """Raised when a webcam clip can't be read or the landmarker models are unavailable."""


@dataclass(frozen=True)
class BodyLanguageResult:
    """Heuristic webcam-based body language coaching for one answer.

    Attributes:
        eye_contact_ratio: Fraction (0-1) of face-detected frames that were
            reasonably forward-facing toward the camera.
        posture_score: 0-100 heuristic score for level, upright shoulders,
            averaged over frames with detected pose landmarks (0 if no pose
            was ever detected).
        looking_away_count: Number of distinct times the candidate looked
            away from the camera for longer than
            ``config.LOOKING_AWAY_MIN_SECONDS``.
        summary: A short, human-readable coaching summary.
    """

    eye_contact_ratio: float
    posture_score: float
    looking_away_count: int
    summary: str


def _ensure_model_file(path: Path, url: str) -> Path:
    """Download a MediaPipe model bundle to ``path`` if it isn't there yet.

    Args:
        path: Local cache location for the model bundle.
        url: URL to download it from.

    Returns:
        ``path``, once the file is confirmed to exist on disk.

    Raises:
        BodyLanguageError: If the file is missing and the download fails.
    """
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, path)  # noqa: S310 - trusted Google model host
    except (URLError, OSError) as exc:
        raise BodyLanguageError(
            f"Could not download the required model file from {url}: {exc}"
        ) from exc
    return path


@lru_cache(maxsize=1)
def _load_face_landmarker() -> Any:
    """Load and cache the MediaPipe Face Landmarker, downloading its model if needed."""
    model_path = _ensure_model_file(
        config.FACE_LANDMARKER_MODEL_PATH, config.FACE_LANDMARKER_MODEL_URL
    )
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


@lru_cache(maxsize=1)
def _load_pose_landmarker() -> Any:
    """Load and cache the MediaPipe Pose Landmarker, downloading its model if needed."""
    model_path = _ensure_model_file(
        config.POSE_LANDMARKER_MODEL_PATH, config.POSE_LANDMARKER_MODEL_URL
    )
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def _bgr_to_mp_image(frame: np.ndarray) -> mp.Image:
    """Wrap a BGR (OpenCV-convention) frame as a MediaPipe ``Image``.

    Args:
        frame: An HxWx3 BGR frame, e.g. from ``cv2.VideoCapture`` or
            ``cv2.imdecode``.

    Returns:
        The frame as an ``mp.Image`` in SRGB format.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _rotation_matrix_to_yaw_pitch(rotation: np.ndarray) -> tuple[float, float]:
    """Decompose a 3x3 rotation matrix into yaw/pitch Euler angles, in degrees.

    Uses the standard Tait-Bryan (Y-then-X) decomposition of a rotation
    matrix. Only the *magnitude* of each angle is used downstream (against
    ``config.EYE_CONTACT_*_THRESHOLD_DEGREES``), so the exact sign convention
    of MediaPipe's facial transformation matrix doesn't matter here.

    Args:
        rotation: The top-left 3x3 rotation block of a 4x4 transform matrix.

    Returns:
        A ``(yaw_degrees, pitch_degrees)`` tuple.
    """
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy < 1e-6:
        pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
    else:
        pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(-rotation[2, 0], sy)
    return math.degrees(yaw), math.degrees(pitch)


def _head_pose_degrees(face_result: Any) -> tuple[float, float] | None:
    """Extract ``(yaw, pitch)`` in degrees from a Face Landmarker result.

    Args:
        face_result: A ``FaceLandmarkerResult`` (or compatible mock) from a
            single detection call.

    Returns:
        The ``(yaw_degrees, pitch_degrees)`` of the first detected face, or
        ``None`` if no face was detected.
    """
    matrices = getattr(face_result, "facial_transformation_matrixes", None)
    if not matrices:
        return None
    rotation = np.asarray(matrices[0])[:3, :3]
    return _rotation_matrix_to_yaw_pitch(rotation)


def _is_forward_facing(yaw_degrees: float, pitch_degrees: float) -> bool:
    """Whether a head pose counts as "looking at the camera".

    Args:
        yaw_degrees: Left/right head rotation, in degrees.
        pitch_degrees: Up/down head rotation, in degrees.

    Returns:
        ``True`` if both angles are within the configured thresholds.
    """
    return (
        abs(yaw_degrees) <= config.EYE_CONTACT_YAW_THRESHOLD_DEGREES
        and abs(pitch_degrees) <= config.EYE_CONTACT_PITCH_THRESHOLD_DEGREES
    )


def _posture_score_for_frame(pose_result: Any) -> float | None:
    """Score one frame's shoulder level/upright posture, from 0-100.

    Combines two heuristics: how level the shoulders are (a tilted line
    suggests leaning or slouching to one side) and how far the nose sits
    above the shoulder midpoint relative to shoulder width (a short
    nose-to-shoulder distance suggests hunching toward the camera).

    Args:
        pose_result: A ``PoseLandmarkerResult`` (or compatible mock) from a
            single detection call.

    Returns:
        A 0-100 score, or ``None`` if no pose (or no usable shoulder
        landmarks) was detected.
    """
    poses = getattr(pose_result, "pose_landmarks", None)
    if not poses:
        return None

    landmarks = poses[0]
    try:
        nose = landmarks[_POSE_NOSE]
        left_shoulder = landmarks[_POSE_LEFT_SHOULDER]
        right_shoulder = landmarks[_POSE_RIGHT_SHOULDER]
    except (IndexError, TypeError):
        return None
    if nose is None or left_shoulder is None or right_shoulder is None:
        return None

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    if shoulder_width < 1e-6:
        return None

    tilt_degrees = math.degrees(
        math.atan2(abs(left_shoulder.y - right_shoulder.y), shoulder_width)
    )
    level_score = max(0.0, 100.0 - (tilt_degrees / config.POSTURE_TILT_SCALE_DEGREES) * 100.0)

    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
    neck_ratio = (shoulder_mid_y - nose.y) / shoulder_width
    upright_score = min(100.0, max(0.0, (neck_ratio / config.POSTURE_IDEAL_NECK_RATIO) * 100.0))

    return (level_score + upright_score) / 2


def _build_summary(
    eye_contact_ratio: float,
    posture_score: float,
    looking_away_count: int,
    frames_with_face: int,
    frames_with_pose: int,
) -> str:
    """Turn the aggregate metrics into a short, plain-language summary.

    Args:
        eye_contact_ratio: As in :class:`BodyLanguageResult`.
        posture_score: As in :class:`BodyLanguageResult`.
        looking_away_count: As in :class:`BodyLanguageResult`.
        frames_with_face: Number of frames a face was detected in.
        frames_with_pose: Number of frames usable pose landmarks were found in.

    Returns:
        One or two short coaching sentences.
    """
    if frames_with_face == 0:
        eye_sentence = "We couldn't detect your face clearly enough to judge eye contact."
    else:
        pct = round(eye_contact_ratio * 100)
        if eye_contact_ratio >= 0.75:
            eye_sentence = f"Good eye contact {pct}% of the time."
        elif eye_contact_ratio >= 0.5:
            eye_sentence = (
                f"Eye contact was okay ({pct}% of the time) — try facing the camera a bit more."
            )
        else:
            eye_sentence = (
                f"Eye contact was low ({pct}% of the time) — try to look at the camera more often."
            )

    if frames_with_pose == 0:
        posture_sentence = "We couldn't see your shoulders clearly enough to judge posture."
    elif posture_score >= 80:
        posture_sentence = "Your posture looked steady and upright."
    elif posture_score >= 60:
        posture_sentence = "Your posture was decent; try to keep your shoulders level."
    else:
        posture_sentence = "Try sitting up straighter with your shoulders level and facing the camera."

    sentences = [eye_sentence, posture_sentence]
    if looking_away_count > 0:
        plural = "s" if looking_away_count != 1 else ""
        sentences.append(
            f"You looked away from the camera {looking_away_count} time{plural} "
            "for more than a second."
        )
    return " ".join(sentences)


def analyze_frames(
    frames: Sequence[np.ndarray],
    fps: float = config.BODY_LANGUAGE_DEFAULT_FPS,
    face_landmarker: Any = None,
    pose_landmarker: Any = None,
) -> BodyLanguageResult:
    """Run body language analysis over an already-decoded sequence of frames.

    Processes frames one at a time (rather than requiring the whole clip up
    front as a single blob), so this also works for a live stream of frames
    in the future, not just a recorded clip.

    Args:
        frames: BGR frames (OpenCV convention), in chronological order.
        fps: Frames per second, used to convert "looking away" frame runs
            into seconds. Defaults to ``config.BODY_LANGUAGE_DEFAULT_FPS``,
            which matters mainly for a raw frame sequence that has no
            inherent timing of its own (e.g. webcam snapshots).
        face_landmarker: Optional pre-built MediaPipe Face Landmarker
            (mainly for testing). Defaults to the cached, lazily-downloaded
            model.
        pose_landmarker: Optional pre-built MediaPipe Pose Landmarker
            (mainly for testing). Defaults to the cached, lazily-downloaded
            model.

    Returns:
        The aggregated :class:`BodyLanguageResult` for the whole sequence.

    Raises:
        BodyLanguageError: If ``frames`` is empty.
        ValueError: If ``fps`` is not positive.
    """
    if not frames:
        raise BodyLanguageError("No frames were provided to analyze.")
    if fps <= 0:
        raise ValueError("fps must be positive.")

    face = face_landmarker if face_landmarker is not None else _load_face_landmarker()
    pose = pose_landmarker if pose_landmarker is not None else _load_pose_landmarker()

    frame_duration = 1.0 / fps
    frames_with_face = 0
    frames_forward = 0
    posture_scores: list[float] = []
    looking_away_count = 0
    away_run_frames = 0
    away_run_counted = False

    for frame in frames:
        mp_image = _bgr_to_mp_image(frame)

        pose_angles = _head_pose_degrees(face.detect(mp_image))
        forward = False
        if pose_angles is not None:
            frames_with_face += 1
            forward = _is_forward_facing(*pose_angles)

        if forward:
            frames_forward += 1
            away_run_frames = 0
            away_run_counted = False
        else:
            away_run_frames += 1
            if (
                not away_run_counted
                and away_run_frames * frame_duration >= config.LOOKING_AWAY_MIN_SECONDS
            ):
                looking_away_count += 1
                away_run_counted = True

        posture_score = _posture_score_for_frame(pose.detect(mp_image))
        if posture_score is not None:
            posture_scores.append(posture_score)

    eye_contact_ratio = frames_forward / frames_with_face if frames_with_face else 0.0
    posture_score_avg = mean(posture_scores) if posture_scores else 0.0

    summary = _build_summary(
        eye_contact_ratio=eye_contact_ratio,
        posture_score=posture_score_avg,
        looking_away_count=looking_away_count,
        frames_with_face=frames_with_face,
        frames_with_pose=len(posture_scores),
    )

    return BodyLanguageResult(
        eye_contact_ratio=round(eye_contact_ratio, 3),
        posture_score=round(posture_score_avg, 1),
        looking_away_count=looking_away_count,
        summary=summary,
    )


def _read_video_frames(path: Path) -> tuple[list[np.ndarray], float]:
    """Decode every frame of a video file, in order.

    Args:
        path: Path to a video file readable by OpenCV.

    Returns:
        A ``(frames, fps)`` tuple. ``fps`` is 0.0 if OpenCV couldn't report it.

    Raises:
        BodyLanguageError: If the file can't be opened or has no frames.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise BodyLanguageError(f"Could not open video file: {path}")

    try:
        detected_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise BodyLanguageError(f"No frames could be read from video file: {path}")
    return frames, detected_fps


def analyze_video(
    video_path_or_frames: str | Path | Sequence[np.ndarray],
    fps: float = config.BODY_LANGUAGE_DEFAULT_FPS,
    face_landmarker: Any = None,
    pose_landmarker: Any = None,
) -> BodyLanguageResult:
    """Analyze eye contact and posture from a short recorded clip.

    Args:
        video_path_or_frames: Either a path to a short video file, or an
            already-decoded sequence of BGR frames (e.g. snapshots captured
            one at a time in the Streamlit app, which has no video recorder
            widget).
        fps: Frames per second. Read from the video file automatically when
            a path is given and OpenCV can report it; otherwise (and always
            for a raw frame sequence) this value is used as-is.
        face_landmarker: Optional pre-built MediaPipe Face Landmarker
            (mainly for testing).
        pose_landmarker: Optional pre-built MediaPipe Pose Landmarker
            (mainly for testing).

    Returns:
        The aggregated :class:`BodyLanguageResult`.

    Raises:
        BodyLanguageError: If a video path can't be opened/read, or no
            frames are available to analyze.
        ValueError: If ``fps`` is not positive.
    """
    if isinstance(video_path_or_frames, (str, Path)):
        frames, detected_fps = _read_video_frames(Path(video_path_or_frames))
        fps = detected_fps if detected_fps > 0 else fps
    else:
        frames = list(video_path_or_frames)

    return analyze_frames(
        frames, fps=fps, face_landmarker=face_landmarker, pose_landmarker=pose_landmarker
    )
