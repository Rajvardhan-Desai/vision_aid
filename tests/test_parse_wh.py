import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import parse_wh


def test_parse_wh_valid():
    assert parse_wh("640x480") == (640, 480)
    assert parse_wh(" 320X240 ") == (320, 240)


def test_parse_wh_empty():
    assert parse_wh("") == (0, 0)


def test_parse_wh_invalid():
    with pytest.raises(ValueError):
        parse_wh("640*480")
    with pytest.raises(ValueError):
        parse_wh("640")
    with pytest.raises(ValueError):
        parse_wh("foo")
