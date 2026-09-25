from pathlib import Path
from ultralytics import YOLO


class RoadDamageInferenceEngine:
    """Reusable production inference engine for road damage detection."""

    def __init__(self, model_path, config):
        self.model_path = Path(model_path)
        self.config = config

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Production model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

        if self.model.task != "detect":
            raise ValueError(
                f"Expected detection model, got: {self.model.task}"
            )

    def predict_image(self, image_path):
        """Run inference on a single image and return structured detections."""

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Input image not found: {image_path}"
            )

        results = self.model.predict(
            source=str(image_path),
            imgsz=self.config.image_size,
            conf=self.config.confidence,
            device=self.config.device,
            max_det=self.config.max_det,
            half=self.config.half,
            save=False,
            verbose=False,
        )

        result = results[0]
        detections = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": class_id,
                "class_name": self.model.names[class_id],
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                },
            })

        return {
            "image": image_path,
            "image_size": (
                result.orig_shape[1],
                result.orig_shape[0],
            ),
            "detection_count": len(detections),
            "detections": detections,
            "runtime": "FP16",
        }