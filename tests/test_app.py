from pathlib import Path

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


def test_app_contains_dashboard_sections():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "About the model" in source
    assert "How to use" in source
    assert ".app-shell" in source


def test_app_uses_status_summary_in_result_section():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "build_status_summary(result)" in source
    assert "result-badge" in source
    assert "result-headline" in source
