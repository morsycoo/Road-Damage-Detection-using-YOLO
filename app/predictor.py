# Define the Production Prediction Service

from pathlib import Path

import cv2

from inference.config import PRODUCTION_CONFIG
from inference.engine import RoadDamageInferenceEngine


class RoadDamagePredictor:
    """Production prediction service for road damage detection."""

    def __init__(self, model_path: Path):
        self.engine = RoadDamageInferenceEngine(
            model_path=model_path,
            config=PRODUCTION_CONFIG,
        )

    def predict_image(self, image_path: Path):
        """Run road damage detection on an image."""
        return self.engine.predict_image(image_path)

    def predict_video(
        self,
        video_path: Path,
        output_path: Path,
    ):
        """Run frame-by-frame road damage detection on a video."""

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(
                "Unable to open the input video."
            )

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        fps = capture.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25.0

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError(
                "Unable to create the output video."
            )

        frames_processed = 0
        total_detections = 0

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                results = self.engine.model.predict(
                    source=frame,
                    imgsz=PRODUCTION_CONFIG.image_size,
                    conf=PRODUCTION_CONFIG.confidence,
                    device=PRODUCTION_CONFIG.device,
                    max_det=PRODUCTION_CONFIG.max_det,
                    half=PRODUCTION_CONFIG.half,
                    save=False,
                    verbose=False,
                )

                result = results[0]

                total_detections += len(result.boxes)

                annotated_frame = result.plot()

                writer.write(annotated_frame)

                frames_processed += 1

        finally:
            capture.release()
            writer.release()

        if frames_processed == 0:
            raise RuntimeError(
                "No video frames were processed."
            )

        return {
            "video_size": [width, height],
            "fps": round(fps, 2),
            "frames_processed": frames_processed,
            "total_detections": total_detections,
            "runtime": "FP16",
        }