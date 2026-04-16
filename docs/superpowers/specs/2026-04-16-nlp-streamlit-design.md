# NLP Streamlit App Design

**Date:** 2026-04-16

**Goal:** Turn `NLP_NEW.ipynb` into a small, deployable Streamlit app that performs single-text emotion prediction using a saved trained model, then prepare the project for GitHub and Streamlit Community Cloud deployment.

## Scope

This design covers:
- A reproducible training script that saves a model artifact
- A small inference module that loads the artifact and returns label plus confidence
- A Streamlit UI for single-text prediction only
- Minimal project files needed for local execution, GitHub push, and free deployment

This design does not cover:
- Batch CSV upload
- In-app retraining
- User authentication
- Database storage
- Multi-page Streamlit navigation

## Current Project Context

The current workspace contains:
- `NLP_NEW.ipynb`
- No existing app structure
- No committed git history yet

The notebook appears to be an emotion-classification project using:
- `TfidfVectorizer`
- `CountVectorizer`
- `MultinomialNB`
- `LogisticRegression`

The deployable app should avoid retraining on startup and instead load a saved artifact for inference.

## Recommended Architecture

Use a small three-file application structure with clear separation between training, inference, and UI:

- `train_model.py`
  - Loads the dataset used by the notebook
  - Trains the selected text classification pipeline
  - Evaluates the model on a holdout split
  - Saves one artifact file to `artifacts/emotion_model.joblib`

- `predict.py`
  - Loads the saved artifact
  - Exposes a small inference API for the UI
  - Returns the predicted label and confidence score

- `app.py`
  - Streamlit front-end only
  - Accepts one text input
  - Validates empty input
  - Calls the inference API
  - Displays predicted emotion label and confidence score

Supporting files:
- `requirements.txt`
- `.gitignore`
- `artifacts/` for saved model files
- Optional `README.md` if needed later

## File Structure

The target structure is:

```text
python/
├── NLP_NEW.ipynb
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── .gitignore
├── artifacts/
│   └── emotion_model.joblib
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-16-nlp-streamlit-design.md
```

## Model Strategy

The app will use a saved model artifact and will not train on startup.

Rationale:
- Faster Streamlit startup
- More reliable free hosting deployment
- Clear separation of training and inference concerns
- Easier local debugging and redeployment

The training script should prefer the notebook's TF-IDF-based approach unless inspection during implementation shows a stronger reason to use a different model already present in the notebook. The saved artifact should contain everything needed for prediction, ideally as a scikit-learn pipeline so the app loads one file and predicts directly.

## UI Design

The Streamlit app should remain intentionally simple:

- Title and short project description
- Single large text area for user input
- Predict button
- Clear result display showing:
  - Predicted emotion label
  - Confidence score

The UI should not expose model training controls, dataset upload, or batch prediction in this first version.

## Data Flow

The expected runtime flow is:

1. Developer runs `python train_model.py`
2. Training script saves `artifacts/emotion_model.joblib`
3. Developer runs `streamlit run app.py`
4. User enters text into the UI
5. `app.py` sends the text to `predict.py`
6. `predict.py` loads the artifact and computes prediction
7. `app.py` renders the label and confidence score

## Error Handling

The first version should explicitly handle these cases:

- Missing artifact file:
  - Show a clear message telling the user to run `python train_model.py`

- Empty text submission:
  - Prevent inference
  - Show a validation warning in Streamlit

- Training script issues:
  - Fail with clear errors if the source dataset cannot be located
  - Print where the script expected the dataset to be

## Testing Strategy

Implementation should follow TDD for the new Python modules.

Minimum automated coverage for the first pass:
- Prediction helper returns both `label` and `confidence`
- Confidence is numeric and bounded between `0.0` and `1.0`
- Missing artifact produces a clear failure path
- Empty UI input does not attempt inference

The tests should focus on inference and artifact loading first. Streamlit UI can remain lightly tested or manually verified in this phase.

## Deployment Requirements

The repository should be structured so it can be deployed directly to Streamlit Community Cloud with:
- GitHub repository connected to Streamlit
- `app.py` as the main file
- `requirements.txt` at repo root

The deployed app should not require notebook execution in production.

## Tradeoffs Considered

### Option A: Separate training, inference, and UI files

Pros:
- Clean boundaries
- Reproducible training path
- Easier deployment and maintenance

Cons:
- Slightly more setup than a notebook-only approach

### Option B: One module for training and inference

Pros:
- Fewer files

Cons:
- Mixed responsibilities
- Harder to reason about later

### Option C: Notebook-only workflow

Pros:
- Fastest short-term path

Cons:
- Weak reproducibility
- Poor deployment ergonomics
- Harder to maintain

Option A is the selected design.

## Success Criteria

The work is complete when:
- `train_model.py` can create a saved model artifact from the project dataset
- `app.py` runs locally with Streamlit
- The UI predicts one text input at a time
- The result includes emotion label and confidence score
- The repo includes the files needed for GitHub push and Streamlit Cloud deployment
- The structure is simple enough for future extension
