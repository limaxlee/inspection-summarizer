"""The inspectionStatistics document.

Fields are declared snake_case and stored camelCase (``sum_sq`` -> ``sumSq``).
Every count, ``sum`` and ``sumSq`` is additive, so any set of hourly documents
can be combined exactly into a larger window. ``mean``, ``std``, ``min``,
``max`` and ``quantiles`` are stored for direct reading.
"""
from datetime import datetime
from typing import Literal

from pydantic import NonNegativeInt, NonNegativeFloat

from inspection_statistics.common.constants import Granularity
from inspection_statistics.schemas.base import DataModel


class Quantiles(DataModel):
    p05: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None


class ValueStats(DataModel):
    count: NonNegativeInt = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    quantiles: Quantiles = Quantiles()


class ConfidenceStats(ValueStats):
    histogram: list[NonNegativeInt] = []                  # len == len(bins.confidenceEdges) - 1
    below_threshold_count: NonNegativeInt | None = None   # None when no threshold applied
    near_threshold_count: NonNegativeInt | None = None
    threshold: float | None = None                        # threshold of this predicted class, both tasks, last seen


class ElapsedTimeStats(DataModel):
    count: NonNegativeInt = 0
    sum: float = 0.0
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    p50: float | None = None
    p90: float | None = None


class Quality(DataModel):
    missing_confidence_count: NonNegativeInt = 0
    parse_error_count: NonNegativeInt = 0


class BoxesPerImage(DataModel):
    sum: NonNegativeInt = 0
    sum_sq: NonNegativeInt = 0
    mean: float | None = None
    std: float | None = None
    max: NonNegativeInt | None = None
    histogram: list[NonNegativeInt] = []                  # bins 0..boxesPerImageMax, last is "max or more"


class ImageStats(DataModel):                               # det only
    no_box_count: NonNegativeInt = 0
    boxes_per_image: BoxesPerImage = BoxesPerImage()
    boxes_per_image_by_class: dict[str, float] = {}       # mean boxes of that class per image


class EquipmentStats(DataModel):
    inspection_count: NonNegativeInt = 0                  # inspection documents that contributed
    prediction_count: NonNegativeInt = 0                  # images
    box_count: NonNegativeInt | None = None               # det only
    quality: Quality = Quality()
    elapsed_time: ElapsedTimeStats = ElapsedTimeStats()
    class_counts: dict[str, NonNegativeInt] = {}
    confidence: ConfidenceStats = ConfidenceStats()
    per_class: dict[str, ConfidenceStats] = {}
    images: ImageStats | None = None                      # det only


class EquipmentIdentity(DataModel):
    equipment_id: str
    location: str
    backend: str | None = None                            # last value seen in the period
    # There is no equipment-level threshold: thresholds live in perClass[c].threshold for both tasks.


# Base order is deliberate: pydantic collects fields from the last base first, so the
# identity fields come before the statistics in the stored document.
class EquipmentEntry(EquipmentStats, EquipmentIdentity):
    pass


class Bins(DataModel):
    confidence_edges: list[NonNegativeFloat] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    near_threshold_margin: NonNegativeFloat = 0.05
    boxes_per_image_max: NonNegativeInt = 5


class InspectionStatisticsDocument(DataModel):
    # ---- identity (unique together) ----
    model_name: str
    model_version: str
    task: Literal["cls", "det"]
    mode: str
    gbm: str
    process: str
    granularity: Granularity
    start_date: datetime
    end_date: datetime

    # ---- model and site properties ----
    product_id: str | None = None                         # first value seen in the period; not part of the key
    classes: list[str] = []                               # ordered union of aiResults[].classes in the period
    local_timezone: str | None = None
    bins: Bins = Bins()

    # ---- bookkeeping ----
    computed_at: datetime
    equipment_count: NonNegativeInt = 0

    total: EquipmentStats = EquipmentStats()
    equipments: list[EquipmentEntry] = []
