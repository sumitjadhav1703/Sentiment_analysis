from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from predict import load_model, predict_text


def test_load_model_raises_clear_error_when_artifact_missing(tmp_path):
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError, match="Run python train_model.py first"):
        load_model(missing_path)


def test_predict_text_returns_label_and_confidence(tmp_path):
    artifact_path = tmp_path / "emotion_model.joblib"
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    x_train = [
        "i feel delighted",
        "i feel happy",
        "i feel miserable",
        "i feel gloomy",
    ]
    y_train = ["joy", "joy", "sadness", "sadness"]
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, artifact_path)

    result = predict_text("i feel happy", artifact_path)

    assert result["label"] in {"joy", "sadness"}
    assert 0.0 <= result["confidence"] <= 1.0
