"""Fluency scoring from extracted speech features.

Wraps the classifier trained in ``notebooks/fluency_model.ipynb`` (Task 6) and
saved to ``models/fluency_model.joblib``. Turns a feature dict from
:func:`src.features.extract_features` into a 3-class fluency label, the
model's confidence in that label, and a 0-100 score for display in the app.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from src import config
from src.features import FEATURE_NAMES

# Ordered worst-to-best; must match the class labels the model was trained on.
FLUENCY_LABELS: tuple[str, ...] = ("Beginner", "Intermediate", "Fluent")

# Representative point (on a 0-100 scale) for each class, used to turn class
# probabilities into a single continuous score: the midpoint of the raw
# 0-10 expert-score band each class was mapped from in the notebook
# (Beginner 0-5, Intermediate 6-7, Fluent 8-10), scaled by 10.
_LABEL_SCORE_ANCHOR: dict[str, float] = {
    "Beginner": 25.0,
    "Intermediate": 65.0,
    "Fluent": 90.0,
}


class ScorerError(RuntimeError):
    """Raised when the trained fluency model is missing or unusable."""


class LabelDecodingClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a classifier trained on integer-encoded labels to expose strings.

    XGBoost's multiclass objective requires integer class labels, unlike
    scikit-learn's own classifiers. This wrapper (used in
    ``notebooks/fluency_model.ipynb`` if XGBoost is the chosen model) makes
    an integer-label classifier behave like a normal scikit-learn one again:
    ``classes_``, ``predict``, and ``predict_proba`` all speak the original
    string labels, so :func:`predict_fluency` doesn't need to know which
    underlying model it's calling. Defined here (rather than inline in the
    notebook) so it has a stable, importable path and can be unpickled by
    ``joblib.load`` from any process, including this module's own.
    """

    def __init__(self, inner: Any, label_encoder: LabelEncoder) -> None:
        """Wrap an integer-label classifier with its label encoder.

        Args:
            inner: A fitted classifier that predicts integer-encoded labels.
            label_encoder: The encoder used to produce those integer labels,
                used here to translate them back to strings.
        """
        self.inner = inner
        self.label_encoder = label_encoder

    @property
    def classes_(self) -> np.ndarray:
        """The original string class labels, in the encoder's order."""
        return self.label_encoder.classes_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict string labels for ``X`` by decoding the inner model's output."""
        return self.label_encoder.inverse_transform(self.inner.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for ``X``, in ``classes_`` order."""
        return self.inner.predict_proba(X)


class FluencyPrediction(NamedTuple):
    """The result of scoring one spoken answer's fluency.

    Attributes:
        label: Predicted class, one of ``FLUENCY_LABELS``.
        confidence: The model's predicted probability for ``label`` (0-1).
        score: A continuous 0-100 fluency score, computed as the class
            probabilities weighted by each class's score anchor.
    """

    label: str
    confidence: float
    score: float


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Load and cache the trained fluency classifier from disk.

    Returns:
        The unpickled scikit-learn/XGBoost classifier (or pipeline).

    Raises:
        ScorerError: If no trained model file exists yet.
    """
    if not config.FLUENCY_MODEL_PATH.exists():
        raise ScorerError(
            f"No trained fluency model found at {config.FLUENCY_MODEL_PATH}. "
            "Run notebooks/fluency_model.ipynb to train and save one first."
        )
    return joblib.load(config.FLUENCY_MODEL_PATH)


def predict_fluency(features: dict[str, float], model: Any = None) -> FluencyPrediction:
    """Predict a fluency label, confidence, and 0-100 score from features.

    Args:
        features: A feature dict as returned by
            :func:`src.features.extract_features`. Must contain every key in
            ``src.features.FEATURE_NAMES``.
        model: Optional pre-loaded classifier (mainly for testing). Defaults
            to the cached model loaded from ``config.FLUENCY_MODEL_PATH``.

    Returns:
        A :class:`FluencyPrediction` with the predicted label, the model's
        confidence in it, and a continuous 0-100 score.

    Raises:
        ScorerError: If no ``model`` is given and no trained model file
            exists on disk.
        KeyError: If ``features`` is missing a required feature name.
    """
    clf = model if model is not None else _load_model()
    row = pd.DataFrame([{name: features[name] for name in FEATURE_NAMES}])

    proba = clf.predict_proba(row)[0]
    classes = list(clf.classes_)

    best_index = int(proba.argmax())
    label = classes[best_index]
    confidence = float(proba[best_index])
    score = sum(p * _LABEL_SCORE_ANCHOR[c] for p, c in zip(proba, classes, strict=True))

    return FluencyPrediction(label=label, confidence=confidence, score=round(score, 1))
