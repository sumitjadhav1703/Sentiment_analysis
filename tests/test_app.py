import pytest

from app import build_result_message, validate_text_input


def test_validate_text_input_rejects_blank_text():
    with pytest.raises(ValueError, match="Please enter some text before predicting."):
        validate_text_input("   ")


def test_build_result_message_formats_label_and_confidence():
    message = build_result_message({"label": "joy", "confidence": 0.8765})

    assert message["label"] == "joy"
    assert message["confidence_text"] == "87.65%"
