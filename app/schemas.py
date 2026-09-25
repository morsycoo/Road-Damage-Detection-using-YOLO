# Define FastAPI Response Schemas

from pydantic import BaseModel


class DetectionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: dict[str, float]


class ImagePredictionResponse(BaseModel):
    input_type: str = "image"
    image: str
    image_size: list[int]
    detection_count: int
    detections: list[DetectionResponse]
    confidence_threshold: float
    runtime: str = "FP16"


class VideoPredictionResponse(BaseModel):
    input_type: str = "video"
    original_video: str
    video: str
    video_size: list[int]
    fps: float
    frames_processed: int
    total_detections: int
    runtime: str = "FP16"
    video_url: str
    watch_url: str