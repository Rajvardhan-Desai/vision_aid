import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import resize_with_ratio


def test_resize_with_ratio_no_padding():
    img = np.ones((50, 100, 3), dtype=np.uint8) * 255
    canvas, (fx, fy), pad, original_size = resize_with_ratio(img, (200, 100))
    assert canvas.shape[:2] == (100, 200)
    assert (fx, fy) == (2, 2)
    assert original_size == (100, 50)


def test_resize_with_ratio_with_padding():
    img = np.ones((50, 100, 3), dtype=np.uint8) * 255
    canvas, (fx, fy), pad, original_size = resize_with_ratio(img, (60, 40))
    assert canvas.shape[:2] == (40, 60)
    # resized area should remain white
    assert np.all(canvas[0:30, 0:60] == 255)
    # padded area should be black
    assert np.all(canvas[30:, :] == 0)
    assert (fx, fy) == (0.6, 0.6)
    assert original_size == (100, 50)

