"""The identity of an inspectionStatistics document.

One document exists per:

    modelName + modelVersion + task + mode + gbm + process + granularity + startDate

``productId`` is NOT part of the key: it can be null and one period aggregates
every barcode inspected in it. The document carries the first productId seen,
purely as information. ``location`` lives inside each ``equipments[]`` entry.
"""
from datetime import datetime
from typing import NamedTuple

from inspection_statistics.common.constants import Granularity, ModelTasks
from inspection_statistics.schemas.inspections import InspectionDocumentV1, AIResultData

# Field names as stored in MongoDB (camelCase), used for the unique index and the upsert filter.
KEY_FIELDS = ["modelName", "modelVersion", "task", "mode", "gbm", "process", "granularity", "startDate"]


class StatisticsKey(NamedTuple):
    model_name: str
    model_version: str
    task: str
    mode: str
    gbm: str
    process: str
    granularity: Granularity
    start_date: datetime

    def to_query(self) -> dict:
        return {
            "modelName": self.model_name,
            "modelVersion": self.model_version,
            "task": self.task,
            "mode": self.mode,
            "gbm": self.gbm,
            "process": self.process,
            "granularity": self.granularity.value,
            "startDate": self.start_date,
        }

    def __str__(self) -> str:
        return (f"{self.model_name}/{self.model_version} [{self.task}] | {self.gbm}/{self.process} "
                f"| {self.mode} | {self.granularity.value} {self.start_date:%Y-%m-%d %H:%M}")


class ModelIdentity(NamedTuple):
    """What distinguishes one statistics document from another inside a single inspection."""
    model_name: str
    model_version: str
    task: str


def split_ai_model(ai_model: str) -> tuple[str, str]:
    """``"sidebottom/sidebottom_250429_260113072605"`` -> ``("sidebottom", "sidebottom_250429_260113072605")``.

    Split on the LAST slash, exactly like the summary collection, so that the two
    collections agree on modelName / modelVersion.
    """
    model_name, _, model_version = ai_model.rpartition("/")
    return model_name, model_version


def model_identity(ai_result: AIResultData) -> ModelIdentity:
    model_name, model_version = split_ai_model(ai_result.ai_model)
    task = ai_result.task.value if isinstance(ai_result.task, ModelTasks) else str(ai_result.task)
    return ModelIdentity(model_name, model_version, task)


def build_key(
        inspection: InspectionDocumentV1,
        identity: ModelIdentity,
        granularity: Granularity,
        start_date: datetime,
) -> StatisticsKey:
    return StatisticsKey(
        model_name=identity.model_name,
        model_version=identity.model_version,
        task=identity.task,
        mode=inspection.metadata.mode,
        gbm=inspection.metadata.gbm,
        process=inspection.metadata.process,
        granularity=granularity,
        start_date=start_date,
    )
