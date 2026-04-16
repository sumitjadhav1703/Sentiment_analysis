from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

import train_model
from train_model import build_and_save_model, load_dataset, load_stop_words, preprocess_text


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
    monkeypatch.setattr(train_model, "load_stop_words", lambda: {"i", "feel", "now", "today"})

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
    monkeypatch.setattr(train_model, "load_stop_words", lambda: (_ for _ in ()).throw(LookupError("missing stopwords")))

    with pytest.raises(LookupError):
        preprocess_text("I feel calm")


def test_build_pipeline_loads_stopwords_once(monkeypatch):
    call_count = 0

    def fake_load_stop_words():
        nonlocal call_count
        call_count += 1
        return {"i", "feel"}

    monkeypatch.setattr(train_model, "load_stop_words", fake_load_stop_words, raising=False)

    pipeline = train_model.build_pipeline()
    preprocessor = pipeline.named_steps["tfidf"].build_preprocessor()

    assert preprocessor("I feel calm") == "calm"
    assert preprocessor("I feel radiant") == "radiant"
    assert call_count == 1


def test_load_stop_words_skips_download_when_stopwords_are_present(monkeypatch):
    monkeypatch.setattr(train_model.nltk.data, "find", lambda resource: resource)
    monkeypatch.setattr(
        train_model.nltk,
        "download",
        lambda *args, **kwargs: pytest.fail("download should not be called"),
    )
    monkeypatch.setattr(
        train_model,
        "stopwords",
        SimpleNamespace(words=lambda language: ["calm", "steady"]),
    )

    assert load_stop_words() == {"calm", "steady"}
