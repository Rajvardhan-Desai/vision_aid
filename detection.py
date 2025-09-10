from typing import List, Dict, Any, Optional, Set, Tuple
import logging

log = logging.getLogger("vision_aid")


class Detector:
    def __init__(self, yolo_model):
        self.model = yolo_model
        # Fallback to model.names if available; else will be set per-result
        self.names = getattr(yolo_model, "names", None)

    def detect_all(self, img, conf_thres: float = 0.5) -> List[Dict[str, Any]]:
        """
        Run inference and return a flat list of detections:
        {
          "class_id": int,
          "class_name": str,
          "conf": float,
          "bbox_xyxy": [x1, y1, x2, y2]
        }
        """
        results = self.model(img, verbose=False)
        boxes = results[0].boxes
        # Prefer names attached to results; else fallback to model.names; else {}
        names = results[0].names if hasattr(results[0], "names") else (self.names or {})
        out: List[Dict[str, Any]] = []

        for i in range(len(boxes)):
            conf = float(boxes[i].conf.item())
            if conf < conf_thres:
                continue

            cls = int(boxes[i].cls.item())

            # Ensure bbox is returned as a flat list [x1, y1, x2, y2]
            xyxy = boxes[i].xyxy
            try:
                xyxy = xyxy.cpu().numpy().reshape(-1).tolist()
            except Exception as e:
                log.warning("Failed to extract bbox: %s", e)
                continue

            if len(xyxy) != 4:
                log.warning("Unexpected bbox shape: %s", xyxy)
                continue

            xyxy = [float(v) for v in xyxy]

            # names from ultralytics are typically dict[int] -> str
            cname = names.get(cls, f"class_{cls}") if isinstance(names, dict) else str(cls)

            out.append({
                "class_id": cls,
                "class_name": cname,
                "conf": conf,
                "bbox_xyxy": xyxy,
            })
        return out


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def nms_per_class(dets: List[Dict[str, Any]], iou_thres: float = 0.45) -> List[Dict[str, Any]]:
    """
    Apply greedy NMS separately for each class_name.
    """
    out: List[Dict[str, Any]] = []
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for d in dets:
        by_class.setdefault(d["class_name"], []).append(d)

    for cname, arr in by_class.items():
        arr = sorted(arr, key=lambda x: x["conf"], reverse=True)
        keep: List[Dict[str, Any]] = []
        while arr:
            best = arr.pop(0)
            keep.append(best)
            arr = [x for x in arr if _iou(best["bbox_xyxy"], x["bbox_xyxy"]) < iou_thres]
        out.extend(keep)
    return out


def process_currency_detections(
    dets: List[Dict[str, Any]],
    class_whitelist: Optional[Set[str]] = None,
    min_conf: float = 0.6,
    use_nms: bool = True,
    nms_iou: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    Filter and (optionally) NMS currency detections.
    """
    cand: List[Dict[str, Any]] = []
    for d in dets:
        if d["conf"] < min_conf:
            continue
        if class_whitelist and d["class_name"] not in class_whitelist:
            continue
        cand.append(d)
    if use_nms:
        cand = nms_per_class(cand, iou_thres=nms_iou)
    return cand


def detect_with_mode(
    img,
    obj_det: Detector,
    currency_det: Optional[Detector],
    currency_active: bool,
    obj_thresh: float,
    curr_thresh: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    Run detection using either the object detector or the currency detector.

    Only one model is invoked per call. When `currency_active` is True and
    a `currency_det` is provided, the currency detector is used with the
    higher `curr_thresh`. Otherwise the regular object detector runs with
    the supplied `obj_thresh`. Returns raw detections from the chosen detector.
    """
    if currency_active and currency_det is not None:
        return currency_det.detect_all(img, conf_thres=curr_thresh)
    return obj_det.detect_all(img, conf_thres=obj_thresh)


def update_object_history(
    history: Dict[Tuple[int, int, str], int],
    dets: List[Dict[str, Any]],
    grid_size: int = 50,
    required_frames: int = 3,
) -> List[Dict[str, Any]]:
    """
    Track objects across frames using a simple spatial grid.

    Each detection is placed into a grid cell based on its centre coordinates.
    When an object appears in the same cell for `required_frames` consecutive
    frames it is considered stable and included in the returned list. Entries
    that disappear are gradually aged out of `history`.
    """
    current: Set[Tuple[int, int, str]] = set()
    stable: List[Dict[str, Any]] = []

    for d in dets:
        x1, y1, x2, y2 = d.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        cell = (int(cx // grid_size), int(cy // grid_size), d.get("class_name", ""))
        current.add(cell)
        history[cell] = history.get(cell, 0) + 1
        if history[cell] >= required_frames:
            stable.append(d)

    # Age out cells not seen this frame
    for key in list(history.keys()):
        if key not in current:
            history[key] -= 1
            if history[key] <= 0:
                del history[key]

    return stable
