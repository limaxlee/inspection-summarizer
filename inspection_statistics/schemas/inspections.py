"""The inspection document as read from MongoDB.

This is the same tree the summarizer uses, with a few fields typed loosely on
purpose. The statistics count data-quality problems per prediction
(``quality.missingConfidenceCount`` / ``parseErrorCount``), which is only
possible if a bad confidence does NOT make the whole document fail validation:

  - ``confidence`` and ``threshold`` are ``Any`` and interpreted in the aggregator
  - ``mode`` is a plain string, so a mode the enum does not know is still counted
  - ``bbox``, ``mask`` and ``inferenceStartedAt`` are optional; they are not used
"""
from datetime import datetime
from typing import Literal, Any, Annotated

from pydantic import Field, NonNegativeInt, BeforeValidator

from inspection_statistics.common.constants import ModelTasks, UploadStatus
from inspection_statistics.schemas.base import DataModel


class InferenceResult(DataModel):
    prediction: str = Field(..., description="Predicted class name")
    confidence: Any = None                      # list[float] | float in a healthy document
    threshold: Any = None                       # float in a healthy document


class InferenceInfo(DataModel):
    prediction_id: NonNegativeInt | None = None
    file_index: NonNegativeInt | None = None
    inference_started_at: datetime | None = None
    elapsed_time: float = 0.0
    is_patch: bool = False
    data_path: str | None = None


class BoundingBox(InferenceResult):
    bbox_id: NonNegativeInt | None = None
    bbox: Any = None


class ObjectDetectionResult(InferenceInfo):
    detections: list[BoundingBox] = []


class ClassificationResult(InferenceResult, InferenceInfo):
    decision: str | None = None


class SegmentationResult(InferenceInfo):
    mask_shape: dict[str, Any] = {}
    mask: Any = None


class ModelInfo(DataModel):
    ai_model: str = Field(..., examples=["MetalCls/1.0"])
    task: ModelTasks
    backend: str | None = Field(None, description="Name of the backend used for inference")
    classes: list[str] = Field([], description="List of class names")


class ClassificationData(ModelInfo):
    task: Literal[ModelTasks.CLASSIFICATION] = ModelTasks.CLASSIFICATION
    predictions: list[ClassificationResult] = []


class ObjectDetectionData(ModelInfo):
    task: Literal[ModelTasks.DETECTION] = ModelTasks.DETECTION
    predictions: list[ObjectDetectionResult] = []


class SegmentationData(ModelInfo):
    task: Literal[ModelTasks.SEGMENTATION] = ModelTasks.SEGMENTATION
    predictions: list[SegmentationResult] = []


AIResultData = ClassificationData | ObjectDetectionData | SegmentationData
DiscriminatedAIResult = Annotated[AIResultData, Field(discriminator="task")]


class Metadata(DataModel):
    gbm: str = Field(..., description="Factory name", examples=["SEV", "SEVT", "SEHC"])
    process: str = Field(..., examples=["SMD", "PQC", "Main"])
    location: str = Field("", description="A specific location of the process", examples=["Line_01"])
    equipment_id: str = Field(..., examples=["Metal_Inspector_01"])
    product_id: str | None = None
    created_at: datetime | None = None
    local_timezone: str | None = None
    mode: str = Field(..., examples=["production", "rework", "test"])


class InspectionResult(DataModel):
    ai_results: list[DiscriminatedAIResult] = []
    conclusion: str | None = None


class InspectionDocumentV1(DataModel):
    inspection_id: Annotated[str, BeforeValidator(str)] | None = Field(None, alias="_id")
    schema_version: str = "1.0"
    metadata: Metadata
    # NOTE: stored as "inspectionResults" (plural) even though the field is singular.
    inspection_result: InspectionResult = Field(..., alias="inspectionResults")
    is_deleted: bool = False
    upload_status: UploadStatus | None = None
