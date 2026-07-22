# Sentiment Analysis

A deployable NLP project built from `NLP_NEW.ipynb`, with a trained text-classification pipeline, a Streamlit front-end, and free-hosting deployment support through Streamlit Community Cloud.

## Overview

This project turns a notebook-based emotion classification workflow into a small application that can be trained locally and served on the web.

It includes:
- A reproducible training script
- A saved model artifact for inference
- A Streamlit UI for single-text prediction
- Tests for training, inference, and UI helpers

## Features

- Train an emotion classifier from `train.txt`
- Preprocess text using the notebook’s cleaning flow
- Save and reload the trained model with `joblib`
- Predict from a single sentence in Streamlit
- Show both predicted label and confidence score

## Architecture

### Training flow

`train.txt` -> `train_model.py` -> TF-IDF + Logistic Regression pipeline -> `artifacts/emotion_model.joblib`

### Inference flow

User input -> `app.py` -> `predict.py` -> saved model artifact -> predicted label + confidence

### App flow

1. User enters text in the Streamlit app
2. Input is validated
3. The saved model is loaded
4. The model predicts the most likely emotion
5. The app displays the label and confidence

## Project Structure

```text
python/
├── app.py
├── predict.py
├── train_model.py
├── train.txt
├── NLP_NEW.ipynb
├── requirements.txt
├── README.md
├── artifacts/
│   ├── .gitkeep
│   └── emotion_model.joblib
├── tests/
│   ├── conftest.py
│   ├── test_app.py
│   ├── test_predict.py
│   ├── test_project_scaffold.py
│   └── test_train_model.py
└── docs/
    └── superpowers/
        ├── plans/
        │   └── 2026-04-16-nlp-streamlit-app.md
        └── specs/
            └── 2026-04-16-nlp-streamlit-design.md
```

## Local Setup

Install dependencies:

```bash
python3 -m pip install --user -r requirements.txt
```

Run the full test suite:

```bash
python3 -m pytest -v
```

Train the model artifact:

```bash
python3 train_model.py
```

Start the Streamlit app:

```bash
python3 -m streamlit run app.py
```

## Model And App Commands

Train:

```bash
python3 train_model.py
```

Expected output:

```text
Saved model to artifacts/emotion_model.joblib
Accuracy: 0.8638
```

Run app:

```bash
python3 -m streamlit run app.py
```

## Deployment

### GitHub

Push `main` to GitHub:

```bash
git push -u origin main
```

### Streamlit Community Cloud

1. Open `https://share.streamlit.io`
2. Sign in with GitHub
3. Click `Create app`
4. Select repository `sumitjadhav1703/Sentiment_analysis`
5. Select branch `main`
6. Set main file path to `app.py`
7. Choose a Python version supported by Streamlit Cloud
8. Click `Deploy`

## Notes

- The notebook `NLP_NEW.ipynb` is included as the original exploration and experimentation source.
- The training and deployment plan is included in `docs/superpowers/plans/2026-04-16-nlp-streamlit-app.md`.
- The design spec is included in `docs/superpowers/specs/2026-04-16-nlp-streamlit-design.md`.
