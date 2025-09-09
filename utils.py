import time
import cv2
import logging
from typing import Tuple

log = logging.getLogger("vision_aid")

class Throttle:
    """Call-per-key at most once per `period` seconds."""
    def __init__(self, period: float = 2.0):
        self.period = period
        self._last = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        last = self._last.get(key, 0.0)
        if now - last >= self.period:
            self._last[key] = now
            return True
        return False


def parse_wh(wh: str) -> Tuple[int,int]:
    if not wh: return (0, 0)
    w, h = wh.lower().split('x')
    return int(w), int(h)

def resize_with_ratio(img, size_wh):
    """Resize keeping aspect (letterbox), returning resized image and scaling ratios."""
    if not size_wh or size_wh == (0,0):  # no-op
        h, w = img.shape[:2]
        return img, (1.0, 1.0), (0, 0), (w, h)

    Wt, Ht = size_wh
    h, w = img.shape[:2]
    r = min(Wt / w, Ht / h)
    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = img
    if new_w != Wt or new_h != Ht:
        canvas = (0 * img[0:Ht, 0:Wt]).copy()
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if len(img.shape)==2 else canvas
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR) if len(canvas.shape)==2 else canvas
        # place resized in top-left (no center pad to keep math simple)
        canvas[0:new_h, 0:new_w] = resized
    fx, fy = r, r
    return canvas, (fx, fy), (0, 0), (w, h)

def scale_bbox_to_original(xyxy, fx, fy, pad=(0,0)):
    x1,y1,x2,y2 = xyxy
    px, py = pad
    return [
        int((x1 - px) / fx),
        int((y1 - py) / fy),
        int((x2 - px) / fx),
        int((y2 - py) / fy),
    ]
