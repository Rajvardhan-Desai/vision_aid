import cv2
import logging
from typing import Optional, Tuple

log = logging.getLogger("vision_aid")

class Camera:
    def __init__(self, source: str, resolution: Optional[Tuple[int,int]] = None):
        # Accept "usb0" → 0, "usb1" → 1, etc.
        if source.startswith("usb"):
            idx = int(source.replace("usb", "") or "0")
            self.cap = cv2.VideoCapture(idx)
        else:
            # Allow RTSP/file paths too
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera source: {source}")

        if resolution:
            w, h = resolution
            if w > 0 and h > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass
