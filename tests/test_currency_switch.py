import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from detection import detect_with_mode

class DummyDet:
    def __init__(self):
        self.calls = 0
    def detect_all(self, img, conf_thres=0.5):
        self.calls += 1
        return [{'class_name': 'dummy', 'conf': 1.0}]


def test_detect_switches_between_models():
    img = object()
    obj = DummyDet()
    cur = DummyDet()
    # Default: object detector used
    detect_with_mode(img, obj, cur, False, obj_thresh=0.5)
    assert obj.calls == 1 and cur.calls == 0
    # Currency active: currency detector used exclusively
    detect_with_mode(img, obj, cur, True, obj_thresh=0.5)
    assert obj.calls == 1 and cur.calls == 1
