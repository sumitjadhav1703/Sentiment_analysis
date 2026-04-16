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


def build_status_summary(result: dict[str, float | str]) -> dict[str, str]:
    message = build_result_message(result)
    return {
        "badge": "Prediction ready",
        "headline": message["label"].title(),
        "subtext": f"Confidence: {message['confidence_text']}",
    }


def main() -> None:
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

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .app-shell {
            max-width: 920px;
            margin: 0 auto;
            padding: 0 0 3rem 0;
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
            max-width: 48rem;
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

        div[data-testid="stTextArea"] textarea {
            min-height: 180px;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: var(--panel-alt);
            color: var(--text);
            caret-color: var(--text);
        }

        div[data-testid="stTextArea"] textarea::placeholder {
            color: var(--muted);
            opacity: 1;
        }

        div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #1f4aa8 0%, #2d64d6 100%);
            color: #ffffff;
            font-weight: 700;
            padding: 0.8rem 1rem;
        }

        div[data-testid="stMetric"] {
            background: var(--panel-alt);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.75rem 1rem;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
            }

            .app-shell {
                padding-top: 0;
            }

            .hero-title {
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    summary: dict[str, str] | None = None

    st.markdown('<div class="app-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel-card">
            <span class="hero-label">Project Dashboard</span>
            <h1 class="hero-title">Emotion Classifier</h1>
            <p class="hero-subtitle">
                Paste a sentence, run a prediction, and review the model output in a calmer,
                more structured dashboard without changing the classifier workflow.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 2.2, 1])
    with center_col:
        st.markdown(
            """
            <div class="panel-card">
                <h2 class="panel-title">Try a prediction</h2>
                <p class="panel-copy">
                    Enter one sentence and submit it to classify the dominant emotion.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        text_input = st.text_area(
            "Text",
            height=180,
            placeholder="Type a sentence here...",
            label_visibility="collapsed",
        )

        if st.button("Predict"):
            try:
                cleaned_text = validate_text_input(text_input)
                result = predict_text(cleaned_text)
                summary = build_status_summary(result)
            except FileNotFoundError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.warning(str(exc))

        if summary is not None:
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

    about_col, how_to_col = st.columns(2)
    with about_col:
        st.markdown(
            """
            <div class="panel-card">
                <h2 class="panel-title">About the model</h2>
                <p class="panel-copy">
                    This app uses the saved emotion classification artifact to infer a label from
                    a single sentence and report the associated confidence score.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with how_to_col:
        st.markdown(
            """
            <div class="panel-card">
                <h2 class="panel-title">How to use</h2>
                <p class="panel-copy">
                    Write one clear sentence, click Predict, and review the returned label and
                    confidence. Blank input still triggers the existing validation warning.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
