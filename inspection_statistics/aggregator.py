"""Per-period accumulators that turn inspection documents into one statistics document.

A statistics document is a FULL recomputation of its period, not an incremental
fold: quantiles and histograms cannot be merged from partial results, so the raw
confidences of a period are kept in memory (as compact float arrays) until the
period is flushed. Everything else is a running count.

Units:
  cls  -> one prediction = one image
  det  -> confidence / classCounts / perClass are over bounding boxes;
          predictionCount and ``images`` are over images
"""
import logging
from array import array
from datetime import datetime
from typing import Iterable

from inspection_statistics.common.constants import ModelTasks
from inspection_statistics.keys import StatisticsKey
from inspection_statistics.schemas.inspections import (
    InspectionDocumentV1, AIResultData, ClassificationResult, ObjectDetectionResult,
)
from inspection_statistics.schemas.statistics import (
    InspectionStatisticsDocument, EquipmentStats, EquipmentEntry, ConfidenceStats, ElapsedTimeStats,
    Quality, ImageStats, BoxesPerImage, Bins,
)
from inspection_statistics.stats_math import (
    confidence_score, as_float, describe, histogram_bin, rounded, OK, MISSING,
)

logger = logging.getLogger(__name__)


class ValueAccumulator:
    """Raw values kept until the period is flushed, so that quantiles are exact."""

    def __init__(self):
        self.values = array("d")

    def add(self, value: float):
        self.values.append(value)

    def merge(self, other: "ValueAccumulator"):
        self.values.extend(other.values)

    @property
    def count(self) -> int:
        return len(self.values)

    def describe(self, digits: int) -> dict:
        return describe(self.values, digits)


class ConfidenceAccumulator(ValueAccumulator):
    """Confidence values plus the fixed-bin histogram and the threshold relation."""

    def __init__(self, bins: Bins):
        super().__init__()
        self._edges = bins.confidence_edges
        self._margin = bins.near_threshold_margin
        self.histogram = [0] * (len(self._edges) - 1)
        self.below = 0
        self.near = 0
        self.threshold_applied = False

    def add(self, value: float, threshold: float | None = None):
        super().add(value)
        self.histogram[histogram_bin(value, self._edges)] += 1

        if threshold is not None:
            self.threshold_applied = True
            if value < threshold:
                self.below += 1
            if abs(value - threshold) < self._margin:
                self.near += 1

    def merge(self, other: "ConfidenceAccumulator"):
        super().merge(other)
        self.histogram = [mine + theirs for mine, theirs in zip(self.histogram, other.histogram)]
        self.below += other.below
        self.near += other.near
        self.threshold_applied = self.threshold_applied or other.threshold_applied

    def build(self, digits: int, threshold: float | None = None) -> ConfidenceStats:
        return ConfidenceStats(
            **self.describe(digits),
            histogram=list(self.histogram),
            below_threshold_count=self.below if self.threshold_applied else None,
            near_threshold_count=self.near if self.threshold_applied else None,
            threshold=threshold,
        )


class EquipmentAccumulator:
    """Everything one equipment contributed to one statistics document. Also used for ``total``."""

    def __init__(self, task: str, bins: Bins, location: str = ""):
        self.task = task
        self.bins = bins
        self.location = location

        # runtime configuration, last value seen in the period
        self.backend: str | None = None
        self.class_thresholds: dict[str, float] = {}         # per predicted class (cls) / box class (det)
        self._conflicting_class_thresholds: set[str] = set()  # only after a merge

        self.inspection_count = 0
        self.prediction_count = 0                            # images
        self.box_count = 0                                   # det only
        self.missing_confidence_count = 0
        self.parse_error_count = 0

        self.elapsed = ValueAccumulator()
        self.class_counts: dict[str, int] = {}               # insertion order == first seen
        self.confidence = ConfidenceAccumulator(bins)
        self.per_class: dict[str, ConfidenceAccumulator] = {}

        # det only, image level
        self.no_box_count = 0
        self.boxes_per_image = ValueAccumulator()

    @property
    def is_detection(self) -> bool:
        return self.task == ModelTasks.DETECTION.value

    # ------------------------------------------------------------------ folding
    def add(self, inspection: InspectionDocumentV1, ai_results: Iterable[AIResultData]):
        """Fold every aiResult of ONE inspection that belongs to this document.

        Called once per inspection, so ``inspectionCount`` counts documents even
        when the same aiModel appears twice inside one of them (both entries'
        predictions still count).
        """
        self.inspection_count += 1

        for ai_result in ai_results:
            if ai_result.backend:
                self.backend = ai_result.backend

            for prediction in ai_result.predictions:
                self.prediction_count += 1
                self.elapsed.add(float(prediction.elapsed_time))

                if self.is_detection:
                    self._add_detection(prediction)
                else:
                    self._add_classification(prediction)

    def _add_classification(self, prediction: ClassificationResult):
        class_name = prediction.prediction
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

        threshold = as_float(prediction.threshold)
        if threshold is not None:
            self.class_thresholds[class_name] = threshold

        self._add_confidence(class_name, prediction.confidence, threshold)

    def _add_detection(self, prediction: ObjectDetectionResult):
        boxes = prediction.detections
        self.box_count += len(boxes)
        self.boxes_per_image.add(float(len(boxes)))
        if not boxes:
            self.no_box_count += 1

        for box in boxes:
            class_name = box.prediction
            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

            threshold = as_float(box.threshold)
            if threshold is not None:
                self.class_thresholds[class_name] = threshold

            self._add_confidence(class_name, box.confidence, threshold)

    def _add_confidence(self, class_name: str, raw_confidence, threshold: float | None):
        score, status = confidence_score(raw_confidence)
        if status != OK:
            if status == MISSING:
                self.missing_confidence_count += 1
            else:
                self.parse_error_count += 1
            return

        # Each confidence is compared against ITS OWN prediction's threshold.
        self.confidence.add(score, threshold)
        per_class = self.per_class.get(class_name)
        if per_class is None:
            per_class = self.per_class[class_name] = ConfidenceAccumulator(self.bins)
        per_class.add(score, threshold)

    # ------------------------------------------------------------------ total
    def merge(self, other: "EquipmentAccumulator"):
        """Pool another equipment into this one. Used to build ``total`` exactly, from raw values."""
        self.inspection_count += other.inspection_count
        self.prediction_count += other.prediction_count
        self.box_count += other.box_count
        self.missing_confidence_count += other.missing_confidence_count
        self.parse_error_count += other.parse_error_count
        self.no_box_count += other.no_box_count

        self.elapsed.merge(other.elapsed)
        self.boxes_per_image.merge(other.boxes_per_image)
        self.confidence.merge(other.confidence)

        for class_name, count in other.class_counts.items():
            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + count

        for class_name, accumulator in other.per_class.items():
            mine = self.per_class.get(class_name)
            if mine is None:
                mine = self.per_class[class_name] = ConfidenceAccumulator(self.bins)
            mine.merge(accumulator)

        # A per-class threshold is only meaningful for the total when every equipment agrees (both tasks).
        for class_name, threshold in other.class_thresholds.items():
            if class_name in self.class_thresholds and self.class_thresholds[class_name] != threshold:
                self._conflicting_class_thresholds.add(class_name)
            self.class_thresholds.setdefault(class_name, threshold)

    # ------------------------------------------------------------------ output
    def build(self, classes: list[str], digits: int) -> dict:
        """The ``EquipmentStats`` fields. ``classes`` is the document-level ordered class list."""
        class_counts = {name: self.class_counts.get(name, 0) for name in classes}

        per_class = {}
        for name in classes:
            threshold = None
            if name not in self._conflicting_class_thresholds:
                threshold = self.class_thresholds.get(name)

            accumulator = self.per_class.get(name)
            if accumulator is None:
                accumulator = ConfidenceAccumulator(self.bins)
            per_class[name] = accumulator.build(digits, threshold)

        elapsed = self.elapsed.describe(digits)
        elapsed_time = ElapsedTimeStats(
            count=elapsed["count"], sum=elapsed["sum"], mean=elapsed["mean"],
            min=elapsed["min"], max=elapsed["max"],
            p50=elapsed["quantiles"].p50, p90=elapsed["quantiles"].p90,
        )

        return {
            "inspection_count": self.inspection_count,
            "prediction_count": self.prediction_count,
            "box_count": self.box_count if self.is_detection else None,
            "quality": Quality(
                missing_confidence_count=self.missing_confidence_count,
                parse_error_count=self.parse_error_count,
            ),
            "elapsed_time": elapsed_time,
            "class_counts": class_counts,
            "confidence": self.confidence.build(digits),
            "per_class": per_class,
            "images": self._build_images(class_counts, digits) if self.is_detection else None,
        }

    def _build_images(self, class_counts: dict[str, int], digits: int) -> ImageStats:
        bin_max = self.bins.boxes_per_image_max
        histogram = [0] * (bin_max + 1)
        for value in self.boxes_per_image.values:
            histogram[min(int(value), bin_max)] += 1

        stats = self.boxes_per_image.describe(digits)
        per_image = BoxesPerImage(
            sum=self.box_count,
            sum_sq=int(round(sum(value * value for value in self.boxes_per_image.values))),
            mean=stats["mean"],
            std=stats["std"],
            max=None if stats["max"] is None else int(stats["max"]),
            histogram=histogram,
        )

        images = self.prediction_count
        by_class = {
            name: rounded(count / images, digits) if images else 0.0
            for name, count in class_counts.items()
        }
        return ImageStats(no_box_count=self.no_box_count, boxes_per_image=per_image, boxes_per_image_by_class=by_class)


class DocumentAccumulator:
    """One statistics document in the making: document-level properties plus one accumulator per equipment."""

    def __init__(self, key: StatisticsKey, bins: Bins, digits: int):
        self.key = key
        self.bins = bins
        self.digits = digits

        self.classes: list[str] = []                 # ordered union of aiResults[].classes
        self.local_timezone: str | None = None       # first seen
        self.product_id: str | None = None           # first seen, informational only
        self._initialized = False

        self.equipments: dict[str, EquipmentAccumulator] = {}
        self.conflicts: list[str] = []
        self._warned: set[str] = set()

    def add(self, inspection: InspectionDocumentV1, ai_results: list[AIResultData]):
        metadata = inspection.metadata

        if not self._initialized:
            self.local_timezone = metadata.local_timezone
            self.product_id = metadata.product_id
            self._initialized = True
        elif metadata.local_timezone != self.local_timezone:
            self._conflict("localTimezone", self.local_timezone, metadata.local_timezone)

        for ai_result in ai_results:
            for class_name in ai_result.classes:
                if class_name not in self.classes:
                    self.classes.append(class_name)

        equipment = self.equipments.get(metadata.equipment_id)
        if equipment is None:
            equipment = EquipmentAccumulator(self.key.task, self.bins, metadata.location)
            self.equipments[metadata.equipment_id] = equipment
        elif equipment.location != metadata.location:
            self._conflict(f"{metadata.equipment_id}.location", equipment.location, metadata.location)

        equipment.add(inspection, ai_results)

    def _conflict(self, field: str, stored, incoming):
        """First writer wins, like the summarizer. Reported once per field per document."""
        if field in self._warned:
            return
        self._warned.add(field)
        message = f"{self.key} -> {field}: keeping {stored!r}, ignoring {incoming!r}"
        self.conflicts.append(message)
        logger.warning(message)

    def build(self, computed_at: datetime, end_date: datetime) -> InspectionStatisticsDocument:
        # Predicted classes that the model never declared still get an entry, after the declared ones.
        classes = list(self.classes)
        for equipment in self.equipments.values():
            for class_name in equipment.class_counts:
                if class_name not in classes:
                    classes.append(class_name)

        total = EquipmentAccumulator(self.key.task, self.bins)
        entries = []
        for equipment_id, equipment in sorted(self.equipments.items()):
            total.merge(equipment)
            entries.append(EquipmentEntry(
                equipment_id=equipment_id,
                location=equipment.location,
                backend=equipment.backend,
                **equipment.build(classes, self.digits),
            ))

        return InspectionStatisticsDocument(
            model_name=self.key.model_name,
            model_version=self.key.model_version,
            task=self.key.task,
            mode=self.key.mode,
            gbm=self.key.gbm,
            process=self.key.process,
            granularity=self.key.granularity,
            start_date=self.key.start_date,
            end_date=end_date,
            product_id=self.product_id,
            classes=classes,
            local_timezone=self.local_timezone,
            bins=self.bins,
            computed_at=computed_at,
            equipment_count=len(entries),
            total=EquipmentStats(**total.build(classes, self.digits)),
            equipments=entries,
        )
