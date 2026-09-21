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


class Granularity(str, Enum):
    """Period type of a statistics document. Boundaries are strictly UTC."""
    HOURLY = "hourly"      # 04:00 -> 05:00
    SHIFT = "shift"        # 00:00 -> 12:00, 12:00 -> 24:00
    DAILY = "daily"        # 00:00 -> next 00:00
    WEEKLY = "weekly"      # Monday 00:00 -> next Monday 00:00
