from pathlib import Path

import joblib
import pandas as pd

from train_model import build_and_save_model, load_dataset


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


def test_build_and_save_model_creates_pipeline_artifact(tmp_path):
    dataset_path = tmp_path / "sample.txt"
    artifact_path = tmp_path / "emotion_model.joblib"
    dataset_path.write_text(
        "i feel calm today;joy\n"
        "i feel delighted now;joy\n"
        "i feel abandoned;sadness\n"
        "i feel hopeless;sadness\n"
        "i feel furious;anger\n"
        "i feel enraged;anger\n",
        encoding="utf-8",
    )

    metrics = build_and_save_model(dataset_path, artifact_path)

    assert artifact_path.exists()
    model = joblib.load(artifact_path)
    assert hasattr(model, "predict")
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
