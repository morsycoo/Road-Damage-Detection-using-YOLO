# Define Application-Specific Exceptions


class RoadDamageException(Exception):
    """Base exception for the road damage detection application."""


class ModelLoadError(RoadDamageException):
    """Raised when the production model cannot be loaded."""


class InferenceError(RoadDamageException):
    """Raised when model inference fails."""


class UnsupportedFileTypeError(RoadDamageException):
    """Raised when an uploaded file type is not supported."""


class VideoProcessingError(RoadDamageException):
    """Raised when video processing fails."""