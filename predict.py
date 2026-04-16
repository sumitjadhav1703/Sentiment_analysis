from __future__ import annotations

from pathlib import Path

import joblib


ARTIFACT_PATH = Path("artifacts/emotion_model.joblib")


def load_model(artifact_path: Path = ARTIFACT_PATH):
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path}. Run python train_model.py first."
        )

    return joblib.load(artifact_path)


def predict_text(text: str, artifact_path: Path = ARTIFACT_PATH) -> dict[str, float | str]:
    model = load_model(artifact_path)

    probabilities = model.predict_proba([text])[0]
    labels = list(model.classes_)
    best_index = int(probabilities.argmax())

    return {
        "label": labels[best_index],
        "confidence": float(probabilities[best_index]),
    }
