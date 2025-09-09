import os, json, threading, logging
from typing import List, Dict, Any

log = logging.getLogger("vision_aid")

class FaceStore:
    """
    Thread-safe JSON store for face encodings.
    Schema:
    {
      "faces": [
        {"name": "Alice", "enc": [0.12, -0.07, ...]},
        ...
      ]
    }
    """
    def __init__(self, path: str = "data/faces.json"):
        self.path = path
        self._lock = threading.RLock()
        self._data = {"faces": []}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
        except Exception as e:
            log.error("Failed to load face DB: %s", e)
            self._data = {"faces": []}

    def _save(self):
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp, self.path)
        except Exception as e:
            log.error("Failed to save face DB: %s", e)

    def list_names(self) -> List[str]:
        with self._lock:
            return [f["name"] for f in self._data.get("faces", [])]

    def add(self, name: str, enc_vec: Any) -> None:
        # enc_vec is a numpy array -> store as list
        enc_list = [float(x) for x in enc_vec.tolist()]
        with self._lock:
            self._data.setdefault("faces", []).append({"name": name, "enc": enc_list})
            self._save()

    def load_all(self) -> Dict[str, Any]:
        with self._lock:
            return {"faces": list(self._data.get("faces", []))}
