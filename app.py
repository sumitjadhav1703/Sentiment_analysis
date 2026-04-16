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
