# tests/test_utils.py
from utils import normalize_text

def test_normalize_text():
    text = "  Hello   World!  "
    result = normalize_text(text)
    assert result == "hello world!"
