import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import calculate_adaptive_inference_size


def test_adaptive_decrease_and_increase():
    size = (640, 640)
    size = calculate_adaptive_inference_size(8.0, size, target_fps=12.0)
    assert size[0] <= 640 and size[0] >= 256
    bigger = calculate_adaptive_inference_size(25.0, size, target_fps=12.0)
    assert bigger[0] >= size[0]
