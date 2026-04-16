from __future__ import annotations

from functools import partial
import re
import string
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATASET_PATH = Path("train.txt")
ARTIFACT_PATH = Path("artifacts/emotion_model.joblib")


def _download_nltk_resources() -> None:
    for resource in ("corpora/stopwords", "corpora/stopwords.zip"):
        try:
            nltk.data.find(resource)
            return
        except LookupError:
            continue

    nltk.download("stopwords", quiet=True)


def load_stop_words() -> set[str]:
    _download_nltk_resources()
    return set(stopwords.words("english"))


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return pd.read_csv(
        dataset_path,
        sep=";",
        header=None,
        names=["text", "emotion"],
    )


def preprocess_text(text: str, stop_words: set[str] | None = None) -> str:
    if stop_words is None:
        stop_words = load_stop_words()

    lowered = text.lower()
    no_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    no_digits = re.sub(r"\d+", "", no_punctuation)
    ascii_only = "".join(char for char in no_digits if char.isascii())
    tokens = [token for token in ascii_only.split() if token and token not in stop_words]
    return " ".join(tokens).strip()


def build_pipeline() -> Pipeline:
    stop_words = load_stop_words()

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(preprocessor=partial(preprocess_text, stop_words=stop_words))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def build_and_save_model(dataset_path: Path, artifact_path: Path) -> dict[str, float]:
    frame = load_dataset(dataset_path)

    x_train, x_test, y_train, y_test = train_test_split(
        frame["text"],
        frame["emotion"],
        test_size=0.2,
        random_state=42,
        stratify=frame["emotion"],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)

    return {"accuracy": float(accuracy)}


def main() -> None:
    metrics = build_and_save_model(DATASET_PATH, ARTIFACT_PATH)
    print(f"Saved model to {ARTIFACT_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
