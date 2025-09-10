from typing import List, Dict, Any, Optional
import logging

log = logging.getLogger("vision_aid")

class Detector:
    def __init__(self, yolo_model):
        self.model = yolo_model
        self.names = getattr(yolo_model, 'names', None)

    def detect_all(self, img, conf_thres: float = 0.5) -> List[Dict[str, Any]]:
        results = self.model(img, verbose=False)
        boxes = results[0].boxes
        names = results[0].names if hasattr(results[0], "names") else self.names or {}
        out: List[Dict[str, Any]] = []
        for i in range(len(boxes)):
            conf = float(boxes[i].conf.item())
            if conf < conf_thres:
                continue
            cls = int(boxes[i].cls.item())
            # Ensure bbox is returned as a flat list [x1, y1, x2, y2]
            xyxy = boxes[i].xyxy.cpu().numpy().reshape(-1).tolist()
            if len(xyxy) != 4:
                log.warning("Unexpected bbox shape: %s", xyxy)
                continue
            xyxy = [float(v) for v in xyxy]
            out.append({
                "class_id": cls,
                "class_name": names.get(cls, f"class_{cls}"),
                "conf": conf,
                "bbox_xyxy": xyxy,
            })
        return out

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1,bx1), max(ay1,by1)
    inter_x2, inter_y2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
    inter = iw*ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter + 1e-6
    return inter/union

def nms_per_class(dets: List[Dict[str,Any]], iou_thres: float=0.45) -> List[Dict[str,Any]]:
    out = []
    by_class: dict = {}
    for d in dets:
        by_class.setdefault(d["class_name"], []).append(d)
    for cname, arr in by_class.items():
        arr = sorted(arr, key=lambda x: x["conf"], reverse=True)
        keep = []
        while arr:
            best = arr.pop(0)
            keep.append(best)
            arr = [x for x in arr if _iou(best["bbox_xyxy"], x["bbox_xyxy"]) < iou_thres]
        out.extend(keep)
    return out

def process_currency_detections(dets: List[Dict[str, Any]],
                               class_whitelist: Optional[set] = None,
                               min_conf: float = 0.6,
                               use_nms: bool = True,
                               nms_iou: float = 0.45) -> List[Dict[str, Any]]:
    cand = []
    for d in dets:
        if d["conf"] < min_conf: 
            continue
        if class_whitelist and d["class_name"] not in class_whitelist:
            continue
        cand.append(d)
    if use_nms:
        cand = nms_per_class(cand, iou_thres=nms_iou)
    return cand
