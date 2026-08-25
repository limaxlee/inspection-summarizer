from datetime import datetime
from typing import Literal, Any, Annotated

from pydantic import Field, NonNegativeInt, PositiveInt, NonNegativeFloat, BeforeValidator

from inspection_summarizer.common.constants import ModelTasks, UploadStatus, InspectionMode
from inspection_summarizer.schemas.base import DataModel


class HumanFeedback(DataModel):
    feedback: str
    details: dict[str, Any] = Field({}, description="Any additional details")
    registered_by: str = ""
    registered_at: datetime


class InferenceResult(DataModel):
    prediction: str = Field(..., description="Predicted class name")
    confidence: list[NonNegativeFloat] | NonNegativeFloat
    threshold: NonNegativeFloat | None = Field(None, description="Minimum confidence to accept the prediction")
    feedbacks: list[HumanFeedback] = []


class InferenceInfo(DataModel):
    prediction_id: NonNegativeInt | None = None
    parent_id: NonNegativeInt | None = None
    child_ids: list[NonNegativeInt] = []
    file_index: NonNegativeInt | None = None
    inference_started_at: datetime
    elapsed_time: float = 0.0
    is_patch: bool = Field(False, description="Whether the image is a patch or not")
    patch_spec: dict[str, NonNegativeFloat] = Field({},
                                                    examples=[{"x1": 80, "y1": 50, "x2": 120, "y2": 90, "channels": 3}])
    data_path: str | None = None
    additional_info: dict[str, Any] = Field({}, description="Any additional information")
    tags: list[str] = Field([], description="List of tags to categorize the image")


class BoundingBox(InferenceResult):
    bbox_id: NonNegativeInt | None = None
    child_ids: list[NonNegativeInt] = []
    bbox: list[NonNegativeFloat] = Field(..., description="Bounding box coordinates in format [x1, y1, x2, y2]")


class ObjectDetectionResult(InferenceInfo):
    detections: list[BoundingBox]


class ClassificationResult(InferenceResult, InferenceInfo):
    decision: str | None = Field(None, description="Final output influenced by threshold")


class SegmentationResult(InferenceInfo):
    mask_shape: dict[str, PositiveInt] = Field({}, examples=[{"height": 512, "width": 512}])
    mask: str


class AIModel(DataModel):
    ai_model: str = Field(..., examples=["MetalCls/1.0"])
    task: ModelTasks


class ModelInfo(AIModel):
    backend: str = Field(..., description="Name of the backend used for inference")
    classes: list[str] = Field(..., description="List of class names")


class ClassificationData(ModelInfo):
    task: Literal[ModelTasks.CLASSIFICATION] = ModelTasks.CLASSIFICATION
    predictions: list[ClassificationResult]


class ObjectDetectionData(ModelInfo):
    task: Literal[ModelTasks.DETECTION] = ModelTasks.DETECTION
    predictions: list[ObjectDetectionResult]


class SegmentationData(ModelInfo):
    task: Literal[ModelTasks.SEGMENTATION] = ModelTasks.SEGMENTATION
    predictions: list[SegmentationResult]


AIResultData = ClassificationData | ObjectDetectionData | SegmentationData
DiscriminatedAIResult = Annotated[AIResultData, Field(discriminator="task")]


class Metadata(DataModel):
    gbm: str = Field(..., description="Factory name", examples=["SEV", "SEVT", "SEHC"])
    process: str = Field(..., examples=["SMD", "PQC", "Main"])
    location: str = Field(..., description="A specific location of the process", examples=["Line_01"])
    equipment_id: str = Field(..., examples=["Metal_Inspector_01"])
    product_id: str | None = None
    created_at: datetime | None = None
    local_timezone: str | None = None
    mode: InspectionMode
    extra_info: dict[str, Any] = Field({}, description="Any extra information about the data")


class InspectionResult(DataModel):
    ai_results: list[DiscriminatedAIResult]
    conclusion: str | None = Field(None, description="The final conclusion of the inspection")
    feedback_entries: list[NonNegativeInt] = []
    summary_file: str | None = Field(None, examples=["MaterialResult.json"])


class AdditionalDataStorage(DataModel):
    info: dict[str, Any] = Field({}, description="Any information about the data")
    path: str


class InspectionDocumentV1(DataModel):
    inspection_id: Annotated[str, BeforeValidator(str)] | None = Field(None, alias="_id")
    schema_version: Literal["1.0"] = "1.0"
    data_spec: list[dict[str, Any]] = []
    metadata: Metadata
    # NOTE: stored as "inspectionResults" (plural) even though the field is singular.
    inspection_result: InspectionResult = Field(..., alias="inspectionResults")
    file_list: list[str] = []
    expiry_date: datetime | None = None
    forwarding_date: datetime | None = None
    is_file_forwarded: bool = False
    is_deleted: bool = False
    upload_status: UploadStatus = UploadStatus.PROCESSING
    uploaded_at: datetime | None = None
    additional_data: list[AdditionalDataStorage] = []
