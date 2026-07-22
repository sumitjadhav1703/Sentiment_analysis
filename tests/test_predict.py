from pathlib import Path
import subprocess
import sys

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


def test_artifact_trained_via_script_loads_in_predict(tmp_path):
    dataset_path = tmp_path / "train.txt"
    artifact_path = tmp_path / "artifacts" / "emotion_model.joblib"
    dataset_path.write_text(
        "i feel calm today;joy\n"
        "i feel delighted now;joy\n"
        "i feel optimistic;joy\n"
        "i feel cheerful;joy\n"
        "i feel excited;joy\n"
        "i feel radiant;joy\n"
        "i feel abandoned;sadness\n"
        "i feel hopeless;sadness\n"
        "i feel lonely;sadness\n"
        "i feel miserable;sadness\n"
        "i feel heartbroken;sadness\n"
        "i feel grief;sadness\n"
        "i feel furious;anger\n"
        "i feel enraged;anger\n"
        "i feel irritated;anger\n"
        "i feel annoyed;anger\n"
        "i feel livid;anger\n"
        "i feel boiling;anger\n",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "/app/train_model.py"],
        check=True,
        cwd=tmp_path,
    )

    result = predict_text("i feel happy", artifact_path)

    assert result["label"] in {"joy", "sadness", "anger"}
    assert 0.0 <= result["confidence"] <= 1.0
