from pathlib import Path

import joblib
import pandas as pd
import pytest

import train_model
from train_model import build_and_save_model, load_dataset, preprocess_text


def test_load_dataset_reads_expected_columns(tmp_path):
    dataset_path = tmp_path / "sample.txt"
    dataset_path.write_text(
        "i feel calm;joy\n"
        "i feel ignored;sadness\n",
        encoding="utf-8",
    )

    frame = load_dataset(dataset_path)

    assert list(frame.columns) == ["text", "emotion"]
    assert frame.shape == (2, 2)


def test_build_and_save_model_creates_pipeline_artifact(monkeypatch, tmp_path):
    class StubStopwords:
        @staticmethod
        def words(language):
            return ["i", "feel", "now", "today"]

    dataset_path = tmp_path / "sample.txt"
    artifact_path = tmp_path / "emotion_model.joblib"
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
    monkeypatch.setattr(train_model, "_download_nltk_resources", lambda: None)
    monkeypatch.setattr(train_model, "stopwords", StubStopwords())

    metrics = build_and_save_model(dataset_path, artifact_path)

    assert artifact_path.exists()
    model = joblib.load(artifact_path)
    assert hasattr(model, "predict")
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_build_and_save_model_uses_fixed_notebook_split(monkeypatch, tmp_path):
    dataset_path = tmp_path / "sample.txt"
    artifact_path = tmp_path / "emotion_model.joblib"
    dataset_path.write_text(
        "i feel calm;joy\n"
        "i feel bright;joy\n"
        "i feel low;sadness\n"
        "i feel blue;sadness\n"
        "i feel mad;anger\n"
        "i feel upset;anger\n",
        encoding="utf-8",
    )

    captured = {}

    def fake_train_test_split(*args, **kwargs):
        captured["test_size"] = kwargs["test_size"]
        captured["random_state"] = kwargs["random_state"]
        captured["stratify"] = kwargs["stratify"]
        return (
            pd.Series(["i feel calm", "i feel low", "i feel mad"]),
            pd.Series(["i feel bright", "i feel blue", "i feel upset"]),
            pd.Series(["joy", "sadness", "anger"]),
            pd.Series(["joy", "sadness", "anger"]),
        )

    class DummyPipeline:
        def fit(self, x_train, y_train):
            return self

        def predict(self, x_test):
            return pd.Series(["joy", "sadness", "anger"])

    monkeypatch.setattr(train_model, "train_test_split", fake_train_test_split)
    monkeypatch.setattr(train_model, "build_pipeline", lambda: DummyPipeline())
    monkeypatch.setattr(train_model.joblib, "dump", lambda pipeline, path: path)

    metrics = build_and_save_model(dataset_path, artifact_path)

    assert captured["test_size"] == 0.2
    assert captured["random_state"] == 42
    assert captured["stratify"].tolist() == [
        "joy",
        "joy",
        "sadness",
        "sadness",
        "anger",
        "anger",
    ]
    assert metrics == {"accuracy": 1.0}


def test_preprocess_text_uses_nltk_stopwords_without_fallback(monkeypatch):
    class MissingStopwords:
        @staticmethod
        def words(language):
            raise LookupError("missing stopwords")

    monkeypatch.setattr(train_model, "_download_nltk_resources", lambda: None)
    monkeypatch.setattr(train_model, "stopwords", MissingStopwords())

    with pytest.raises(LookupError):
        preprocess_text("I feel calm")
