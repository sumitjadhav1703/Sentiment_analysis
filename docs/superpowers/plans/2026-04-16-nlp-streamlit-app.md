# NLP Streamlit App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible emotion-classification project from `NLP_NEW.ipynb` with a saved model artifact, a single-text Streamlit UI, and the repo files needed for GitHub and Streamlit Community Cloud deployment.

**Architecture:** Keep training, inference, and UI separate. `train_model.py` will convert the notebook workflow into a plain Python training script that reads `train.txt` and saves a scikit-learn pipeline artifact. `predict.py` will provide a small inference API that loads the artifact and returns `label` plus `confidence`, and `app.py` will stay Streamlit-only and call that API.

**Tech Stack:** Python 3.11+, pandas, scikit-learn, nltk, streamlit, joblib, pytest

---

## File Structure

Planned files and responsibilities:

- Create: `requirements.txt`
  - Runtime and local development dependencies
- Create: `.gitignore`
  - Ignore local-only files without excluding the trained model artifact
- Create: `train_model.py`
  - Reads `train.txt`, preprocesses text, trains the classifier, evaluates it, and saves `artifacts/emotion_model.joblib`
- Create: `predict.py`
  - Loads the saved artifact and exposes `predict_text(text: str) -> dict`
- Create: `app.py`
  - Streamlit UI for a single text input and prediction result display
- Create: `tests/test_train_model.py`
  - Verifies dataset loading and training artifact creation
- Create: `tests/test_predict.py`
  - Verifies prediction contract and missing-artifact behavior
- Create: `tests/test_app.py`
  - Verifies empty-input and artifact-missing handling through small pure helper functions
- Create: `artifacts/.gitkeep`
  - Keeps the artifacts directory in the repo before the first model is trained
- Modify: `docs/superpowers/specs/2026-04-16-nlp-streamlit-design.md`
  - No further spec changes expected during implementation

### Task 1: Project Scaffolding And Dependency Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `artifacts/.gitkeep`

- [ ] **Step 1: Write the failing structure test**

Create `tests/test_project_scaffold.py` with:

```python
from pathlib import Path


def test_required_project_files_exist():
    required_paths = [
        Path("requirements.txt"),
        Path(".gitignore"),
        Path("artifacts/.gitkeep"),
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    assert missing == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_project_scaffold.py -v
```

Expected:
- `FAIL`
- Failure mentions missing `requirements.txt`, `.gitignore`, or `artifacts/.gitkeep`

- [ ] **Step 3: Write minimal scaffolding files**

Create `requirements.txt` with:

```txt
streamlit
pandas
scikit-learn
nltk
joblib
pytest
```

Create `.gitignore` with:

```gitignore
.DS_Store
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
```

Create the artifact placeholder:

```bash
mkdir -p artifacts tests
touch artifacts/.gitkeep
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_project_scaffold.py -v
```

Expected:
- `PASS`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore artifacts/.gitkeep tests/test_project_scaffold.py
git commit -m "chore: add project scaffolding"
```

### Task 2: Training Script For Notebook Logic

**Files:**
- Create: `train_model.py`
- Create: `tests/test_train_model.py`

- [ ] **Step 1: Write the failing training tests**

Create `tests/test_train_model.py` with:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_train_model.py -v
```

Expected:
- `FAIL`
- Failure mentions `ModuleNotFoundError: No module named 'train_model'` or missing functions

- [ ] **Step 3: Write minimal training implementation**

Create `train_model.py` with:

```python
from __future__ import annotations

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
    nltk.download("stopwords", quiet=True)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return pd.read_csv(
        dataset_path,
        sep=";",
        header=None,
        names=["text", "emotion"],
    )


def preprocess_text(text: str) -> str:
    _download_nltk_resources()
    stop_words = set(stopwords.words("english"))

    lowered = text.lower()
    no_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    no_digits = re.sub(r"\d+", "", no_punctuation)
    ascii_only = no_digits.encode("ascii", errors="ignore").decode()
    tokens = [token for token in ascii_only.split() if token not in stop_words]
    return " ".join(tokens).strip()


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(preprocessor=preprocess_text)),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_train_model.py -v
```

Expected:
- `PASS`

- [ ] **Step 5: Refactor for notebook parity**

Update `preprocess_text` in `train_model.py` so it mirrors the notebook’s cleaning more closely while preserving the same public function names:

```python
def preprocess_text(text: str) -> str:
    _download_nltk_resources()
    stop_words = set(stopwords.words("english"))

    lowered = text.lower()
    no_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    no_digits = re.sub(r"\d+", "", no_punctuation)
    ascii_only = "".join(char for char in no_digits if char.isascii())
    tokens = [token for token in ascii_only.split() if token and token not in stop_words]
    return " ".join(tokens).strip()
```

Run:

```bash
pytest tests/test_train_model.py -v
```

Expected:
- `PASS`

- [ ] **Step 6: Commit**

```bash
git add train_model.py tests/test_train_model.py
git commit -m "feat: add reproducible model training script"
```

### Task 3: Inference Module For Saved Artifact

**Files:**
- Create: `predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write the failing inference tests**

Create `tests/test_predict.py` with:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_predict.py -v
```

Expected:
- `FAIL`
- Failure mentions `ModuleNotFoundError: No module named 'predict'` or missing functions

- [ ] **Step 3: Write minimal inference implementation**

Create `predict.py` with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_predict.py -v
```

Expected:
- `PASS`

- [ ] **Step 5: Commit**

```bash
git add predict.py tests/test_predict.py
git commit -m "feat: add saved-model inference module"
```

### Task 4: Streamlit App With Single-Text Prediction

**Files:**
- Create: `app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write the failing app-helper tests**

Create `tests/test_app.py` with:

```python
import pytest

from app import build_result_message, validate_text_input


def test_validate_text_input_rejects_blank_text():
    with pytest.raises(ValueError, match="Please enter some text before predicting."):
        validate_text_input("   ")


def test_build_result_message_formats_label_and_confidence():
    message = build_result_message({"label": "joy", "confidence": 0.8765})

    assert message["label"] == "joy"
    assert message["confidence_text"] == "87.65%"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_app.py -v
```

Expected:
- `FAIL`
- Failure mentions `ModuleNotFoundError: No module named 'app'` or missing functions

- [ ] **Step 3: Write minimal Streamlit app implementation**

Create `app.py` with:

```python
from __future__ import annotations

import streamlit as st

from predict import predict_text


def validate_text_input(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Please enter some text before predicting.")
    return cleaned


def build_result_message(result: dict[str, float | str]) -> dict[str, str]:
    confidence = float(result["confidence"]) * 100
    return {
        "label": str(result["label"]),
        "confidence_text": f"{confidence:.2f}%",
    }


def main() -> None:
    st.set_page_config(page_title="Emotion Classifier", page_icon="🧠")
    st.title("Emotion Classifier")
    st.write("Enter one sentence to predict its emotion.")

    text_input = st.text_area("Text", height=180, placeholder="Type a sentence here...")

    if st.button("Predict"):
        try:
            cleaned_text = validate_text_input(text_input)
            result = predict_text(cleaned_text)
            message = build_result_message(result)

            st.success(f"Predicted emotion: {message['label']}")
            st.metric("Confidence", message["confidence_text"])
        except FileNotFoundError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.warning(str(exc))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_app.py -v
```

Expected:
- `PASS`

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add single-text Streamlit UI"
```

### Task 5: End-To-End Verification And Trained Artifact

**Files:**
- Modify: `artifacts/` by adding `artifacts/emotion_model.joblib`
- Verify: `train.txt`
- Verify: `app.py`
- Verify: `predict.py`
- Verify: `train_model.py`

- [ ] **Step 1: Train the real model artifact**

Run:

```bash
python3 train_model.py
```

Expected:
- Output contains `Saved model to artifacts/emotion_model.joblib`
- Output contains an `Accuracy:` line
- File `artifacts/emotion_model.joblib` exists

- [ ] **Step 2: Run the full test suite**

Run:

```bash
pytest -v
```

Expected:
- All tests `PASS`

- [ ] **Step 3: Smoke-test local Streamlit startup**

Run:

```bash
streamlit run app.py
```

Expected:
- Terminal shows a local URL, usually `http://localhost:8501`
- Entering blank text shows the validation warning
- Entering a sentence shows a label and a confidence score

- [ ] **Step 4: Commit the application and artifact**

```bash
git add train.txt artifacts/emotion_model.joblib train_model.py predict.py app.py tests requirements.txt .gitignore
git commit -m "feat: ship deployable NLP Streamlit app"
```

### Task 6: GitHub Push And Streamlit Cloud Deployment Checklist

**Files:**
- No code changes required
- Uses repo state from previous tasks

- [ ] **Step 1: Verify remote status**

Run:

```bash
git remote -v
```

Expected:
- Either no output yet, or an existing `origin`

- [ ] **Step 2: Create or update the GitHub remote**

If there is no remote:

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/nlp-streamlit-app.git
```

If a remote already exists and needs replacement:

```bash
git remote set-url origin https://github.com/YOUR_GITHUB_USERNAME/nlp-streamlit-app.git
```

- [ ] **Step 3: Push the branch**

Run:

```bash
git branch -M main
git push -u origin main
```

Expected:
- GitHub receives the full repo including `artifacts/emotion_model.joblib`

- [ ] **Step 4: Deploy on Streamlit Community Cloud**

Use this exact UI flow:

1. Open `https://share.streamlit.io`
2. Sign in with GitHub
3. Click `Create app`
4. Select repository `YOUR_GITHUB_USERNAME/nlp-streamlit-app`
5. Select branch `main`
6. Set main file path to `app.py`
7. Open `Advanced settings`
8. Set Python version to `3.11` if available
9. Click `Deploy`

- [ ] **Step 5: Commit only if docs or deployment notes were added**

If no files changed, do not create an empty commit.

If you add a README during deployment, commit it with:

```bash
git add README.md
git commit -m "docs: add local run and deployment instructions"
```

## Self-Review

Spec coverage check:
- Saved-model architecture is covered by Tasks 2 through 5
- Single-text Streamlit UI is covered by Task 4
- GitHub push and Streamlit Community Cloud deployment are covered by Task 6
- Minimal repo structure and dependencies are covered by Task 1

Placeholder scan:
- No `TODO`, `TBD`, or deferred implementation markers remain in task instructions
- All command steps include explicit commands and expected outcomes

Type consistency:
- `predict_text()` returns `dict[str, float | str]` consistently across Tasks 3 and 4
- `validate_text_input()` and `build_result_message()` are defined in Task 4 and tested with matching names
- `build_and_save_model()` and `load_dataset()` are defined in Task 2 and used with matching names in tests
