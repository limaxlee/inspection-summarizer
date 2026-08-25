from datetime import datetime
from typing import Literal, Any

from pydantic import NonNegativeInt, PositiveInt, NonNegativeFloat

from inspection_summarizer.common.constants import ModelTasks, InspectionMode
from inspection_summarizer.schemas.base import DataModel
from inspection_summarizer.schemas.inspections import HumanFeedback, BoundingBox


class ValueStats(DataModel):
    avg: NonNegativeFloat = 0.0
    min: NonNegativeFloat = 0.0
    max: NonNegativeFloat = 0.0
    sum: NonNegativeFloat = 0.0


class Statistics(DataModel):
    data_count: dict[str, NonNegativeInt] = {}
    confidence: dict[str, ValueStats] = {}
    elapsed_time: ValueStats = ValueStats()


class BaseDataSampleInfo(DataModel):
    inspection_id: str | None = None
    prediction_id: NonNegativeInt | None = None
    inference_started_at: datetime
    is_patch: bool = False
    patch_spec: dict[str, NonNegativeFloat] = {}
    data_path: str | None = None
    elapsed_time: float = 0.0
    additional_info: dict[str, Any] = {}
    tags: list[str] = []


class ClassificationSample(BaseDataSampleInfo):
    prediction: str
    decision: str | None = None
    confidence: list[NonNegativeFloat] | NonNegativeFloat
    feedbacks: list[HumanFeedback] = []
    threshold: NonNegativeFloat | None = None


class DetectionSample(BaseDataSampleInfo):
    detections: list[BoundingBox]


class SegmentationSample(BaseDataSampleInfo):
    mask: str
    mask_shape: dict[str, PositiveInt] = {}


# Resolved by pydantic's smart union: each member has a distinct required field
# (prediction+confidence / detections / mask), so there is no ambiguity.
DataSamples = ClassificationSample | DetectionSample | SegmentationSample


class BaseSummaryInfo(DataModel):
    model_name: str
    model_version: str
    gbm: str
    process: str
    mode: InspectionMode | None = None
    date: datetime


class InspectionSummaryDocumentV1(BaseSummaryInfo):
    schema_version: Literal["1.0"] = "1.0"
    inspection_ids: list[str] = []
    location: str
    equipment_id: str
    product_id: str | None = None
    local_timezone: str | None = None
    task: ModelTasks
    classes: list[str] = []
    threshold: NonNegativeFloat | None = None
    statistics: Statistics = Statistics()
    samples: dict[str, list[DataSamples]] = {}
