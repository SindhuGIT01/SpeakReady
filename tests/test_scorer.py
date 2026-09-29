"""Tests for fluency scoring.

The trained classifier is mocked throughout so these tests never depend on
``models/fluency_model.joblib`` existing. One live test exercises the real
saved model and is skipped when it isn't present.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src import config
from src.features import FEATURE_NAMES
from src.scorer import FLUENCY_LABELS, FluencyPrediction, ScorerError, predict_fluency

FIXTURE_FEATURES: dict[str, float] = {
    "words_per_minute": 140.0,
    "pause_count": 2,
    "long_pause_count": 0,
    "mean_pause_duration": 0.6,
    "total_pause_ratio": 0.08,
    "filler_count": 1,
    "filler_rate": 1.2,
    "repetition_count": 0,
    "type_token_ratio": 0.75,
    "mean_sentence_length": 12.0,
}


def _mock_model(classes: list[str], proba: list[float]) -> MagicMock:
    """A mock classifier whose predict_proba always returns ``proba``."""
    model = MagicMock()
    model.classes_ = classes
    model.predict_proba.return_value = np.array([proba])
    return model


# --- predict_fluency -------------------------------------------------------------


def test_predict_fluency_picks_highest_probability_class() -> None:
    """The label with the highest predicted probability should be returned."""
    model = _mock_model(list(FLUENCY_LABELS), [0.1, 0.2, 0.7])

    result = predict_fluency(FIXTURE_FEATURES, model=model)

    assert isinstance(result, FluencyPrediction)
    assert result.label == "Fluent"
    assert result.confidence == pytest.approx(0.7)


def test_predict_fluency_score_is_probability_weighted_average() -> None:
    """The 0-100 score should be the anchors weighted by class probability."""
    model = _mock_model(list(FLUENCY_LABELS), [0.1, 0.2, 0.7])

    result = predict_fluency(FIXTURE_FEATURES, model=model)

    expected = 0.1 * 25.0 + 0.2 * 65.0 + 0.7 * 90.0
    assert result.score == pytest.approx(expected, abs=0.05)


def test_predict_fluency_score_in_valid_range_for_each_class() -> None:
    """A confident prediction for any class should stay within 0-100."""
    for i, label in enumerate(FLUENCY_LABELS):
        proba = [0.0, 0.0, 0.0]
        proba[i] = 1.0
        model = _mock_model(list(FLUENCY_LABELS), proba)

        result = predict_fluency(FIXTURE_FEATURES, model=model)

        assert result.label == label
        assert result.confidence == pytest.approx(1.0)
        assert 0.0 <= result.score <= 100.0


def test_predict_fluency_passes_features_in_fixed_column_order() -> None:
    """Features must be assembled into a row using FEATURE_NAMES order."""
    model = _mock_model(list(FLUENCY_LABELS), [0.2, 0.3, 0.5])

    predict_fluency(FIXTURE_FEATURES, model=model)

    row = model.predict_proba.call_args[0][0]
    assert list(row.columns) == list(FEATURE_NAMES)
    assert row.iloc[0]["words_per_minute"] == FIXTURE_FEATURES["words_per_minute"]


def test_predict_fluency_missing_feature_raises_key_error() -> None:
    """A features dict missing a required key should fail loudly."""
    model = _mock_model(list(FLUENCY_LABELS), [0.2, 0.3, 0.5])
    incomplete = {k: v for k, v in FIXTURE_FEATURES.items() if k != "filler_rate"}

    with pytest.raises(KeyError):
        predict_fluency(incomplete, model=model)


# --- _load_model -------------------------------------------------------------------


def test_predict_fluency_raises_scorer_error_when_model_missing(monkeypatch) -> None:
    """No model file and no explicit model should raise a clear ScorerError."""
    from src import scorer

    scorer._load_model.cache_clear()
    monkeypatch.setattr(config, "FLUENCY_MODEL_PATH", config.MODELS_DIR / "does_not_exist.joblib")

    with pytest.raises(ScorerError, match="No trained fluency model"):
        predict_fluency(FIXTURE_FEATURES)

    scorer._load_model.cache_clear()


# --- live test using the real saved model -------------------------------------------

pytestmark_live = pytest.mark.skipif(
    not config.FLUENCY_MODEL_PATH.exists(),
    reason="models/fluency_model.joblib not found; run the training notebook first",
)


@pytestmark_live
def test_predict_fluency_live_model_returns_valid_prediction() -> None:
    """The real saved model should return a well-formed prediction."""
    result = predict_fluency(FIXTURE_FEATURES)

    assert result.label in FLUENCY_LABELS
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.score <= 100.0
