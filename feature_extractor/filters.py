"""User-controlled filters for selecting which inspection documents to process.

Every field is explicit -- nothing is inferred. Omitted fields simply add no
constraint. List-valued fields become ``$in`` clauses.
"""
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, model_validator

from feature_extractor.common.constants import ModelTasks, InspectionMode, UploadStatus


def _clause(values: list | None) -> Any:
    """A single value matches directly; several become an $in."""
    if not values:
        return None
    return values[0] if len(values) == 1 else {"$in": list(values)}


def _enum_values(values: list | None) -> list | None:
    return [value.value if hasattr(value, "value") else value for value in values] if values else None


class InspectionFilter(BaseModel):
    # Required time window, half-open: [start, end)
    start: datetime
    end: datetime

    # metadata.* filters
    gbm: list[str] | None = None
    process: list[str] | None = None
    location: list[str] | None = None
    equipment_id: list[str] | None = None
    product_id: list[str] | None = None
    mode: list[InspectionMode] | None = None
    local_timezone: list[str] | None = None

    # inspectionResults.* filters
    ai_model: list[str] | None = None
    model_name: list[str] | None = None
    task: list[ModelTasks] | None = None
    conclusion: list[str] | None = None

    # document-level filters
    upload_status: list[UploadStatus] | None = None
    include_deleted: bool = False

    @model_validator(mode="after")
    def _check_window(self) -> "InspectionFilter":
        if self.end <= self.start:
            raise ValueError(f"--end ({self.end}) must be after --start ({self.start})")
        return self

    def to_mongo_query(self, start: datetime | None = None, end: datetime | None = None) -> dict:
        """Build the query. ``start``/``end`` override the window for day-by-day walking.

        A ``$gte`` on a date already excludes documents whose createdAt is null or
        missing (null sorts below Date in BSON), so no extra null guard is needed.
        """
        query: dict[str, Any] = {
            "metadata.createdAt": {"$gte": start or self.start, "$lt": end or self.end},
        }

        metadata_filters = {
            "metadata.gbm": _clause(self.gbm),
            "metadata.process": _clause(self.process),
            "metadata.location": _clause(self.location),
            "metadata.equipmentId": _clause(self.equipment_id),
            "metadata.productId": _clause(self.product_id),
            "metadata.mode": _clause(_enum_values(self.mode)),
            "metadata.localTimezone": _clause(self.local_timezone),
            "inspectionResults.aiResults.aiModel": _clause(self.ai_model),
            "inspectionResults.aiResults.task": _clause(_enum_values(self.task)),
            "inspectionResults.conclusion": _clause(self.conclusion),
            "uploadStatus": _clause(_enum_values(self.upload_status)),
        }
        query.update({field: value for field, value in metadata_filters.items() if value is not None})

        # modelName is derived (aiModel is "name/version"), so match on the prefix.
        # Skipped when --ai-model is also set, since that is the stricter filter on
        # the same field; matches_ai_result() re-applies both per aiResult anyway.
        if self.model_name and not self.ai_model:
            alternation = "|".join(re.escape(name) for name in self.model_name)
            query["inspectionResults.aiResults.aiModel"] = {"$regex": f"^({alternation})/"}

        if not self.include_deleted:
            query["isDeleted"] = {"$ne": True}

        return query

    def matches_ai_result(self, ai_result) -> bool:
        """Re-apply the model/task filters per aiResult.

        Matching ``inspectionResults.aiResults.aiModel`` selects the DOCUMENT, so
        without this the other models inside that document would be processed too.
        """
        if self.ai_model and ai_result.ai_model not in self.ai_model:
            return False

        if self.model_name:
            derived_name, _, _ = ai_result.ai_model.rpartition("/")
            if derived_name not in self.model_name:
                return False

        if self.task and ai_result.task not in self.task:
            return False

        return True

    def describe(self) -> str:
        active = self.model_dump(exclude_none=True, exclude={"start", "end"}, mode="json")
        lines = [f"  {'window':<15}: [{self.start.isoformat()}, {self.end.isoformat()})"]
        for name, value in active.items():
            if name == "include_deleted" and value is False:
                value = "False (isDeleted != true)"
            lines.append(f"  {name:<15}: {value}")
        return "\n".join(lines)
