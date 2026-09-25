from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    image_size: int = 640
    confidence: float = 0.25
    max_det: int = 300
    device: int | str = 0
    half: bool = True


PRODUCTION_CONFIG = InferenceConfig()
