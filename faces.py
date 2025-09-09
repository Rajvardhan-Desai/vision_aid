import logging
from typing import List, Dict, Any
import numpy as np
from .storage import FaceStore

log = logging.getLogger("vision_aid")

try:
    import face_recognition
    _FACE_OK = True
except Exception as e:
    log.warning("face_recognition not available: %s", e)
    _FACE_OK = False


class FaceDB:
    def __init__(self, db_path: str = "data/faces.json"):
        self.store = FaceStore(db_path)
        self.names: List[str] = []
        self.encs: List[np.ndarray] = []
        self._load_from_store()

    def _load_from_store(self):
        data = self.store.load_all()
        self.names = [f["name"] for f in data.get("faces", [])]
        self.encs = [np.array(f["enc"], dtype=np.float32) for f in data.get("faces", [])]
        log.info("Loaded %d known faces", len(self.names))

    def list_names(self) -> List[str]:
        return list(self.names)

    def register_from_image(self, name: str, img_bgr):
        if not _FACE_OK:
            raise RuntimeError("face_recognition not installed")
        img_rgb = img_bgr[:, :, ::-1]
        locs = face_recognition.face_locations(img_rgb)
        if not locs:
            raise ValueError("No face found for registration")
        enc = face_recognition.face_encodings(img_rgb, known_face_locations=locs)[0]
        self.store.add(name, enc)
        # refresh memory copy
        self._load_from_store()

    def recognize(self, img_bgr, fx: float = 1.0, fy: float = 1.0):
        if not _FACE_OK or not self.encs:
            return []
        img_rgb = img_bgr[:, :, ::-1]
        locs = face_recognition.face_locations(img_rgb)
        encs = face_recognition.face_encodings(img_rgb, known_face_locations=locs)
        results = []
        if not len(encs): return results

        known_encs = self.encs
        for (top, right, bottom, left), enc in zip(locs, encs):
            # distances & name
            dists = face_recognition.face_distance(known_encs, enc)
            name = "Unknown"; conf_like = 0.0
            if len(dists):
                best = int(np.argmin(dists))
                # threshold ~0.6 typical; compute a soft confidence
                if dists[best] < 0.6:
                    name = self.names[best]
                    conf_like = float(max(0.0, 1.0 - dists[best]))
            # dynamic scale back
            x1, y1, x2, y2 = int(left / fx), int(top / fy), int(right / fx), int(bottom / fy)
            results.append({"name": name, "conf_like": conf_like, "bbox_xyxy": [x1,y1,x2,y2]})
        return results
