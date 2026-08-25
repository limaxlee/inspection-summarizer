"""The identity of an inspection summary document.

One summary document exists per:

    modelName + modelVersion + gbm + process + mode + date

where ``date`` is the UTC day start of the source inspection's
``metadata.createdAt``. Note that location / equipmentId / productId /
localTimezone live ON the summary document but are NOT part of this key -- the
summarizer copies them from whichever inspection creates the bucket first and
never updates them afterwards.
"""
from datetime import datetime
from typing import NamedTuple

from inspection_summarizer.schemas.summary import BaseSummaryInfo

# Field names as stored in MongoDB (camelCase), used to build the upsert filter.
KEY_FIELDS = ["modelName", "modelVersion", "gbm", "process", "mode", "date"]


class SummaryKey(NamedTuple):
    model_name: str
    model_version: str
    gbm: str
    process: str
    mode: str | None
    date: datetime

    def to_query(self) -> dict:
        return {
            "modelName": self.model_name,
            "modelVersion": self.model_version,
            "gbm": self.gbm,
            "process": self.process,
            "mode": self.mode,
            "date": self.date,
        }

    def __str__(self) -> str:
        return (f"{self.model_name}/{self.model_version} | {self.gbm}/{self.process} "
                f"| {self.mode} | {self.date:%Y-%m-%d}")


def build_key(summary_base: BaseSummaryInfo) -> SummaryKey:
    mode = summary_base.mode.value if summary_base.mode is not None else None
    return SummaryKey(
        model_name=summary_base.model_name,
        model_version=summary_base.model_version,
        gbm=summary_base.gbm,
        process=summary_base.process,
        mode=mode,
        date=summary_base.date,
    )
