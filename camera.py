import logging
from typing import Optional, Tuple

import cv2

log = logging.getLogger("vision_aid")


class Camera:
    def __init__(self, source: str, resolution: Optional[Tuple[int, int]] = None):
        self.cap = None
        self.picam2 = None

        # Accept "usb0" → 0, "usb1" → 1, etc.
        if source.startswith("usb"):
            idx = int(source.replace("usb", "") or "0")
            self.cap = cv2.VideoCapture(idx)
        elif source.startswith("picamera"):
            try:
                from picamera2 import Picamera2
            except ImportError as e:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "PiCamera2 library not available; cannot use picamera source"
                ) from e

            idx = int(source.replace("picamera", "") or "0")
            self.picam2 = Picamera2(camera_num=idx)
            if resolution and all(resolution):
                config = self.picam2.create_video_configuration(main={"size": resolution})
            else:
                config = self.picam2.create_video_configuration()
            self.picam2.configure(config)
            self.picam2.start()
        else:
            # Allow RTSP/file paths too
            self.cap = cv2.VideoCapture(source)

        if self.cap is not None and not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera source: {source}")

        if self.cap is not None and resolution:
            w, h = resolution
            if w > 0 and h > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def read(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                return None
            return frame

        if self.picam2 is not None:
            frame = self.picam2.capture_array()
            # Picamera2 returns RGB; convert to BGR for OpenCV compatibility
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return None

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
            if self.picam2 is not None:
                self.picam2.close()
        except Exception:
            pass
