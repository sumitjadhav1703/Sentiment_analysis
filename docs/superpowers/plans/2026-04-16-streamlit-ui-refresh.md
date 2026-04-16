# Streamlit UI Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the existing Streamlit front end into a clean professional dashboard while preserving the current single-text prediction workflow and model behavior.

**Architecture:** Keep all existing prediction behavior intact and focus the work on presentation inside `app.py`. The refresh will introduce a more intentional page structure, CSS styling, and supporting information panels, while reusing the current helper functions and inference path. Tests should continue to protect the helper-layer behavior, and the final verification should confirm the updated UI still serves and predicts correctly.

**Tech Stack:** Python 3.14, Streamlit, pytest

---

## File Structure

Planned files and responsibilities:

- Modify: `app.py`
  - Replace the default Streamlit look with a structured dashboard layout, custom CSS, and supporting panels
- Modify: `tests/test_app.py`
  - Keep helper tests aligned with any small presentation helper additions
- Verify: `predict.py`
  - No code changes expected; used to confirm UI still delegates inference cleanly
- Verify: `artifacts/emotion_model.joblib`
  - No code changes expected; used for end-to-end smoke verification

### Task 1: Add Dashboard Layout Helpers

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write the failing test for new presentation helpers**

Update `tests/test_app.py` to add a new test for a status-summary helper that the refreshed UI will use:

```python
import pytest

from app import build_result_message, build_status_summary, validate_text_input


def test_validate_text_input_rejects_blank_text():
    with pytest.raises(ValueError, match="Please enter some text before predicting."):
        validate_text_input("   ")


def test_build_result_message_formats_label_and_confidence():
    message = build_result_message({"label": "joy", "confidence": 0.8765})

    assert message["label"] == "joy"
    assert message["confidence_text"] == "87.65%"


def test_build_status_summary_formats_dashboard_copy():
    summary = build_status_summary({"label": "joy", "confidence": 0.8765})

    assert summary["badge"] == "Prediction ready"
    assert summary["headline"] == "Joy"
    assert summary["subtext"] == "Confidence: 87.65%"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_app.py -v
```

Expected:
- `FAIL`
- Failure mentions missing `build_status_summary`

- [ ] **Step 3: Add the minimal helper implementation**

Update `app.py` to include:

```python
def build_status_summary(result: dict[str, float | str]) -> dict[str, str]:
    message = build_result_message(result)
    return {
        "badge": "Prediction ready",
        "headline": message["label"].title(),
        "subtext": f"Confidence: {message['confidence_text']}",
    }
```

Leave `validate_text_input()` and `build_result_message()` behavior unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python3 -m pytest tests/test_app.py -v
```

Expected:
- `PASS`

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add UI refresh helpers"
```

### Task 2: Replace Default Streamlit Layout With A Styled Dashboard

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Write the failing style/layout smoke assertion**

Extend `tests/test_app.py` with a lightweight markup smoke test that checks the new CSS block and dashboard section labels are present in `app.py`:

```python
from pathlib import Path


def test_app_contains_dashboard_sections():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "About the model" in source
    assert "How to use" in source
    assert ".app-shell" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_app.py -v
```

Expected:
- `FAIL`
- Failure mentions missing dashboard section text or `.app-shell`

- [ ] **Step 3: Implement the dashboard structure and styling**

Replace the current minimal `main()` layout in `app.py` with:
- A custom CSS block injected through `st.markdown(..., unsafe_allow_html=True)`
- A top header section with:
  - small project label
  - main title
  - concise subtitle
- A centered main prediction card
- A result card directly below the input card
- Two supporting panels below:
  - `About the model`
  - `How to use`

Use CSS variables and restrained visual styling. Include a shell wrapper class named `.app-shell`.

Use this structure inside `main()`:

```python
st.set_page_config(page_title="Emotion Classifier", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #f4f7fb;
        --panel: #ffffff;
        --panel-alt: #f8fafc;
        --text: #142033;
        --muted: #5b6678;
        --line: #d9e1ec;
        --accent: #1f4aa8;
        --accent-soft: #e8f0ff;
        --success-soft: #e9f8ef;
        --shadow: 0 18px 40px rgba(20, 32, 51, 0.08);
        --radius: 20px;
    }

    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
        color: var(--text);
    }

    .app-shell {
        max-width: 920px;
        margin: 0 auto;
        padding: 2rem 0 3rem 0;
    }

    .hero-label {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 1rem 0 0.4rem 0;
        font-size: 2.5rem;
        line-height: 1.05;
        font-weight: 800;
        color: var(--text);
    }

    .hero-subtitle {
        margin: 0 0 1.6rem 0;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .panel-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 1.4rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .panel-title {
        margin: 0 0 0.35rem 0;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
    }

    .panel-copy {
        margin: 0;
        color: var(--muted);
        line-height: 1.7;
    }

    .result-badge {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        background: var(--success-soft);
        color: #18794e;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .result-headline {
        margin: 0.7rem 0 0.2rem 0;
        font-size: 2rem;
        font-weight: 800;
        color: var(--text);
    }

    .result-subtext {
        margin: 0;
        color: var(--muted);
    }

    @media (max-width: 768px) {
        .app-shell {
            padding-top: 1rem;
        }

        .hero-title {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
```

Then structure the page with:
- one top hero section
- one prediction card around the textarea/button
- conditional result card rendered only after a successful prediction
- two lower informational cards

Continue using `predict_text()` exactly as before.

- [ ] **Step 4: Run tests to verify the updated layout still passes helper checks**

Run:

```bash
python3 -m pytest tests/test_app.py -v
```

Expected:
- `PASS`

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: refresh Streamlit dashboard layout"
```

### Task 3: Improve Result Presentation Without Changing Prediction Logic

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Write the failing test for result summary wiring**

Extend `tests/test_app.py` with:

```python
from pathlib import Path


def test_app_uses_status_summary_in_result_section():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "build_status_summary(result)" in source
    assert "result-badge" in source
    assert "result-headline" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_app.py -v
```

Expected:
- `FAIL`
- Failure mentions the missing summary usage or result classes

- [ ] **Step 3: Implement the polished result block**

Update the prediction branch in `main()` so that after:

```python
result = predict_text(cleaned_text)
```

it also builds:

```python
summary = build_status_summary(result)
message = build_result_message(result)
```

Then render the result card using `st.markdown(..., unsafe_allow_html=True)` with this structure:

```python
st.markdown(
    f"""
    <div class="panel-card">
        <span class="result-badge">{summary['badge']}</span>
        <div class="result-headline">{summary['headline']}</div>
        <p class="result-subtext">{summary['subtext']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
```

Do not remove the existing user-facing error handling.

- [ ] **Step 4: Run tests to verify it passes**

Run:

```bash
python3 -m pytest tests/test_app.py -v
```

Expected:
- `PASS`

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: improve prediction result presentation"
```

### Task 4: Verify The Refreshed UI End-To-End

**Files:**
- Verify: `app.py`
- Verify: `predict.py`
- Verify: `artifacts/emotion_model.joblib`
- Verify: `tests/test_app.py`

- [ ] **Step 1: Run the full test suite**

Run:

```bash
python3 -m pytest -v
```

Expected:
- All tests `PASS`

- [ ] **Step 2: Exercise the app with Streamlit’s testing API**

Run:

```bash
python3 - <<'PY'
from streamlit.testing.v1 import AppTest

app = AppTest.from_file("/Users/sumitjadhav/python/app.py")
app.run()

app.button[0].click().run()
assert [warning.value for warning in app.warning] == ["Please enter some text before predicting."]

app = AppTest.from_file("/Users/sumitjadhav/python/app.py")
app.run()
app.text_area[0].set_value("i feel happy today").run()
app.button[0].click().run()

assert [success.value for success in app.success] == []  # result now renders in custom card
assert any(metric.value for metric in app.metric) is False
assert "Predicted emotion" not in str(app)
PY
```

Expected:
- Script exits successfully
- Blank input warning is still present
- No regression crashes during prediction flow

- [ ] **Step 3: Start Streamlit locally for a visual smoke check**

Run:

```bash
python3 -m streamlit run app.py --server.headless true --server.port 8505
```

Expected:
- Output includes a local URL such as `http://localhost:8505`
- Page boots successfully

If sandbox port binding fails, rerun the same command with the required elevated permissions and capture the boot output.

- [ ] **Step 4: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: finalize Streamlit UI refresh"
```

## Self-Review

Spec coverage check:
- Clean professional dashboard direction is covered by Tasks 2 and 3
- Single-column prediction emphasis is covered by the hero + prediction card layout in Task 2
- Supporting `About the model` and `How to use` panels are explicitly added in Task 2
- No behavior changes are preserved through helper tests and end-to-end verification in Tasks 1, 3, and 4

Placeholder scan:
- No `TODO`, `TBD`, or deferred implementation notes remain
- Every code-changing step includes exact code or precise structural instructions
- Every verification step includes exact commands and expected results

Type consistency:
- `build_status_summary()` consistently returns `badge`, `headline`, and `subtext`
- `build_result_message()` remains the source of confidence string formatting
- `validate_text_input()` behavior remains unchanged throughout the plan
