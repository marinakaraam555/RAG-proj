"""Extended Track: YOLO-based image detection, used to enrich the RAG prompt
with what's actually visible in an uploaded vehicle/plate photo.
"""
from pathlib import Path

_yolo_model = None


def get_model():
    """Lazy-loaded so the backend still starts even if ultralytics/weights aren't
    available in a given environment (image endpoint will just report unavailable).
    """
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        # Swap this path for a custom-trained plate-detector .pt for higher accuracy,
        # e.g. runs/detect/train/weights/best.pt
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def detect_image_context(image_path: str, conf: float = 0.25) -> str:
    model = get_model()
    results = model(image_path, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            detections.append(f"{cls_name} (confidence {confidence:.2f})")
    if not detections:
        return "No objects detected in the image above the confidence threshold."
    return "Detected in image: " + "; ".join(detections) + "."
