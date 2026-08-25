from enum import Enum


class ModelTasks(str, Enum):
    CLASSIFICATION = "cls"
    DETECTION = "det"
    SEGMENTATION = "seg"


class InspectionMode(str, Enum):
    TEST = "test"
    PRODUCTION = "production"
    REWORK = "rework"


class UploadStatus(str, Enum):
    COMPLETED = "completed"
    PROCESSING = "processing"
