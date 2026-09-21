"""Synthetic-data checks for the statistics package. No MongoDB is contacted.

    python -m pytest inspection_statistics/tests -q
    python inspection_statistics/tests/test_statistics.py        # same checks without pytest

Covers: period boundaries and window snapping, confidence parsing, quantiles and
histogram bins, the classification and detection documents, the fan-out rules,
first-writer-wins conflicts, and a full pipeline run against in-memory fakes,
including the exact additivity of hourly documents into the daily one.
"""
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inspection_statistics.aggregator import DocumentAccumulator                         # noqa: E402
from inspection_statistics.common.constants import Granularity                            # noqa: E402
from inspection_statistics.common.datetime_util import (                                  # noqa: E402
    period_start, period_end, snap_window, iter_periods,
)
from inspection_statistics.filters import InspectionFilter                                # noqa: E402
from inspection_statistics.keys import build_key, model_identity, StatisticsKey           # noqa: E402
from inspection_statistics.pipeline import StatisticsPipeline                             # noqa: E402
from inspection_statistics.schemas.base import to_mongo                                   # noqa: E402
from inspection_statistics.schemas.inspections import InspectionDocumentV1                # noqa: E402
from inspection_statistics.schemas.statistics import Bins, InspectionStatisticsDocument   # noqa: E402
from inspection_statistics.stats_math import (                                            # noqa: E402
    confidence_score, quantile, histogram_bin, OK, MISSING, ERROR,
)

MONDAY = datetime(2026, 9, 21)          # 2026-09-21 is a Monday
T = MONDAY.replace(hour=5, minute=36)   # the example from the discussion
BINS = Bins()
DIGITS = 6


# ------------------------------------------------------------------ builders
def cls_pred(prediction: str, confidence, threshold=0.5, elapsed=0.2) -> dict:
    return {"prediction": prediction, "confidence": confidence, "threshold": threshold, "elapsedTime": elapsed}


def cls_result(predictions: list[dict], ai_model="side/side_v1", classes=("OK", "NG"), backend="ts") -> dict:
    return {"aiModel": ai_model, "task": "cls", "backend": backend, "classes": list(classes),
            "predictions": predictions}


def box(prediction: str, confidence, threshold=0.4) -> dict:
    return {"prediction": prediction, "confidence": confidence, "threshold": threshold, "bbox": [1, 2, 3, 4]}


def det_image(boxes: list[dict], elapsed=0.5) -> dict:
    return {"detections": boxes, "elapsedTime": elapsed}


def det_result(images: list[dict], ai_model="det/det_v1", classes=("scratch", "dent"), backend="trt") -> dict:
    return {"aiModel": ai_model, "task": "det", "backend": backend, "classes": list(classes),
            "predictions": images}


def seg_result(ai_model="seg/seg_v1") -> dict:
    return {"aiModel": ai_model, "task": "seg", "backend": "ts", "classes": ["mask"],
            "predictions": [{"mask": "...", "maskShape": {"height": 4, "width": 4}, "elapsedTime": 1.0}]}


def make_inspection(_id, created_at: datetime, equipment_id: str, ai_results: list[dict], *,
                    product_id="P1", mode="production", local_timezone="Asia/Bangkok",
                    location=None, is_deleted=False) -> dict:
    return {
        "_id": _id,
        "metadata": {
            "gbm": "SEHC", "process": "Side", "location": location or equipment_id[-4:],
            "equipmentId": equipment_id, "productId": product_id, "createdAt": created_at,
            "localTimezone": local_timezone, "mode": mode,
        },
        "inspectionResults": {"aiResults": ai_results},
        "isDeleted": is_deleted,
    }


def parse(raw: dict) -> InspectionDocumentV1:
    return InspectionDocumentV1.model_validate(raw)


def aggregate(raws: list[dict], granularity=Granularity.HOURLY) -> InspectionStatisticsDocument:
    """Fold raw inspections that all share one key into one document."""
    inspections = [parse(raw) for raw in raws]
    first = inspections[0]
    identity = model_identity(first.inspection_result.ai_results[0])
    start = period_start(first.metadata.created_at, granularity)
    key = build_key(first, identity, granularity, start)

    accumulator = DocumentAccumulator(key, BINS, DIGITS)
    for inspection in inspections:
        matching = [ai for ai in inspection.inspection_result.ai_results if model_identity(ai) == identity]
        accumulator.add(inspection, matching)

    document = accumulator.build(datetime(2026, 9, 21, 6, 1), period_end(start, granularity))
    # Whatever goes to MongoDB must validate back into the schema.
    InspectionStatisticsDocument.model_validate(to_mongo(document.model_dump(by_alias=True)))
    return document


# ------------------------------------------------------------------ fakes
class FakeInspections:
    def __init__(self, docs: list[dict]):
        self.docs = docs

    def _select(self, query: dict) -> list[dict]:
        window = query["metadata.createdAt"]
        exclude_deleted = query.get("isDeleted") == {"$ne": True}
        mode = query.get("metadata.mode")
        modes = None if mode is None else (mode["$in"] if isinstance(mode, dict) else [mode])
        selected = [
            doc for doc in self.docs
            if doc["metadata"]["createdAt"] is not None
            and window["$gte"] <= doc["metadata"]["createdAt"] < window["$lt"]
            and not (exclude_deleted and doc.get("isDeleted"))
            and (modes is None or doc["metadata"]["mode"] in modes)
        ]
        return sorted(selected, key=lambda doc: doc["metadata"]["createdAt"])

    def count(self, query: dict) -> int:
        return len(self._select(query))

    def iter_documents(self, query: dict, batch_size: int, limit=None):
        selected = self._select(query)
        yield from (selected[:limit] if limit else selected)


class FakeStatistics:
    def __init__(self):
        self.store: dict[StatisticsKey, dict] = {}

    def ensure_indexes(self, retention_seconds: int):
        pass

    def upsert(self, key: StatisticsKey, document: InspectionStatisticsDocument) -> str:
        payload = to_mongo(document.model_dump(by_alias=True))
        InspectionStatisticsDocument.model_validate(payload)
        outcome = "updated" if key in self.store else "created"
        self.store[key] = payload
        return outcome

    def by_granularity(self, granularity: Granularity) -> dict[StatisticsKey, dict]:
        return {key: doc for key, doc in self.store.items() if key.granularity == granularity}


def run_pipeline(docs: list[dict], start: datetime, end: datetime, granularities=None, **filter_kwargs):
    statistics = FakeStatistics()
    pipeline = StatisticsPipeline(FakeInspections(docs), statistics, BINS, DIGITS, batch_size=10)
    filters = InspectionFilter(start=start, end=end, **filter_kwargs)
    report = pipeline.run(filters, granularities or list(Granularity))
    return statistics, report


# ------------------------------------------------------------------ period arithmetic
def test_period_boundaries_for_the_example_timestamp():
    assert period_start(T, Granularity.HOURLY) == MONDAY.replace(hour=5)
    assert period_end(period_start(T, Granularity.HOURLY), Granularity.HOURLY) == MONDAY.replace(hour=6)
    assert period_start(T, Granularity.SHIFT) == MONDAY
    assert period_end(MONDAY, Granularity.SHIFT) == MONDAY.replace(hour=12)
    assert period_start(T.replace(hour=13), Granularity.SHIFT) == MONDAY.replace(hour=12)
    assert period_start(T, Granularity.DAILY) == MONDAY
    assert period_end(MONDAY, Granularity.DAILY) == MONDAY + timedelta(days=1)
    assert period_start(T, Granularity.WEEKLY) == MONDAY
    assert period_start(MONDAY + timedelta(days=6, hours=23), Granularity.WEEKLY) == MONDAY   # Sunday night
    assert period_end(MONDAY, Granularity.WEEKLY) == MONDAY + timedelta(days=7)


def test_window_is_snapped_outward_to_complete_periods():
    start, end = MONDAY.replace(hour=4, minute=30), MONDAY.replace(hour=6, minute=10)
    assert snap_window(start, end, Granularity.HOURLY) == (MONDAY.replace(hour=4), MONDAY.replace(hour=7))
    assert snap_window(start, end, Granularity.SHIFT) == (MONDAY, MONDAY.replace(hour=12))
    assert snap_window(start, end, Granularity.DAILY) == (MONDAY, MONDAY + timedelta(days=1))
    assert snap_window(start, end, Granularity.WEEKLY) == (MONDAY, MONDAY + timedelta(days=7))

    # an end that already sits on a boundary is not widened
    assert snap_window(MONDAY, MONDAY + timedelta(days=1), Granularity.HOURLY) == (MONDAY, MONDAY + timedelta(days=1))
    assert snap_window(MONDAY, MONDAY + timedelta(days=1), Granularity.WEEKLY) == (MONDAY, MONDAY + timedelta(days=7))

    hourly = list(iter_periods(start, end, Granularity.HOURLY))
    assert [h for h, _ in hourly] == [MONDAY.replace(hour=4), MONDAY.replace(hour=5), MONDAY.replace(hour=6)]
    assert list(iter_periods(MONDAY, MONDAY, Granularity.HOURLY)) == []


# ------------------------------------------------------------------ numerics
def test_confidence_score_rules():
    assert confidence_score([0.1, 0.97]) == (0.97, OK)
    assert confidence_score(0.52) == (0.52, OK)
    assert confidence_score(1) == (1.0, OK)
    assert confidence_score([]) == (None, MISSING)
    assert confidence_score(None) == (None, MISSING)
    assert confidence_score("0.9") == (None, ERROR)
    assert confidence_score([0.1, "x"]) == (None, ERROR)
    assert confidence_score(True) == (None, ERROR)
    assert confidence_score(float("nan")) == (None, ERROR)


def test_quantiles_and_histogram_bins():
    assert quantile([1, 2, 3, 4], 0.5) == 2.5
    assert quantile([1, 2, 3, 4], 0.25) == 1.75
    assert quantile([7], 0.9) == 7
    assert quantile([], 0.5) is None
    edges = BINS.confidence_edges
    assert histogram_bin(0.0, edges) == 0
    assert histogram_bin(0.09999, edges) == 0
    assert histogram_bin(0.1, edges) == 1
    assert histogram_bin(0.5, edges) == 5
    assert histogram_bin(0.95, edges) == 9
    assert histogram_bin(1.0, edges) == 9          # last bin closed
    assert histogram_bin(1.2, edges) == 9          # clamped


# ------------------------------------------------------------------ classification
def test_classification_document():
    vm07 = [
        make_inspection(1, T, "SEHC_Side_VM07", [cls_result([
            cls_pred("OK", [0.1, 0.97], threshold=0.5),
            cls_pred("NG", 0.52, threshold=0.5),          # near 0.5, not below
            cls_pred("OK", [], threshold=0.5),            # missing
            cls_pred("OK", "bad", threshold=0.5),         # parse error
            cls_pred("NG", 0.30, threshold=0.6),          # own threshold 0.6: below, not near; last seen
        ], classes=("OK", "NG", "SCRAP"))]),
    ]
    vm08 = [
        make_inspection(2, T + timedelta(minutes=5), "SEHC_Side_VM08", [cls_result([
            cls_pred("OK", 0.90, threshold=None, elapsed=1.0),
        ], backend="trt")]),
    ]
    document = aggregate(vm07 + vm08)

    assert document.task == "cls"
    assert document.start_date == MONDAY.replace(hour=5) and document.end_date == MONDAY.replace(hour=6)
    assert document.classes == ["OK", "NG", "SCRAP"]
    assert document.product_id == "P1" and document.local_timezone == "Asia/Bangkok"
    assert document.equipment_count == 2
    assert [entry.equipment_id for entry in document.equipments] == ["SEHC_Side_VM07", "SEHC_Side_VM08"]

    first, second = document.equipments
    assert (first.location, first.backend) == ("VM07", "ts")
    assert not hasattr(first, "threshold")                                  # no equipment-level threshold
    assert first.inspection_count == 1 and first.prediction_count == 5
    assert first.box_count is None and first.images is None
    assert first.quality.missing_confidence_count == 1 and first.quality.parse_error_count == 1
    assert first.class_counts == {"OK": 3, "NG": 2, "SCRAP": 0}          # counts every prediction, valid or not
    assert first.confidence.count == 3
    assert first.confidence.histogram == [0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    assert first.confidence.below_threshold_count == 1
    assert first.confidence.near_threshold_count == 1
    assert first.per_class["OK"].count == 1 and first.per_class["OK"].mean == 0.97
    assert first.per_class["NG"].histogram == [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    assert first.per_class["SCRAP"].count == 0 and first.per_class["SCRAP"].mean is None
    assert first.per_class["SCRAP"].histogram == [0] * 10
    assert first.per_class["SCRAP"].below_threshold_count is None
    assert first.per_class["OK"].threshold == 0.5                           # last threshold seen for OK
    assert first.per_class["NG"].threshold == 0.6                           # last threshold seen for NG
    assert first.per_class["SCRAP"].threshold is None

    assert second.backend == "trt"
    assert second.per_class["OK"].threshold is None                         # its only prediction had none
    assert second.confidence.below_threshold_count is None                  # no threshold applied at all
    assert document.total.per_class["OK"].threshold == 0.5                  # VM08 reported none, so no disagreement
    assert document.total.per_class["NG"].threshold == 0.6
    assert second.elapsed_time.p50 == 1.0

    total = document.total
    assert total.inspection_count == 2 and total.prediction_count == 6
    assert total.class_counts == {"OK": 4, "NG": 2, "SCRAP": 0}
    assert total.confidence.count == 4
    assert total.confidence.histogram == [0, 0, 0, 1, 0, 1, 0, 0, 0, 2]
    assert total.confidence.below_threshold_count == 1 and total.confidence.near_threshold_count == 1
    pooled = sorted([0.97, 0.52, 0.30, 0.90])
    assert total.confidence.quantiles.p50 == round(quantile(pooled, 0.5), DIGITS)
    assert total.confidence.min == 0.3 and total.confidence.max == 0.97
    assert math.isclose(total.confidence.sum, sum(pooled), abs_tol=1e-6)
    assert math.isclose(total.confidence.sum_sq, sum(v * v for v in pooled), abs_tol=1e-6)
    mean = sum(pooled) / 4
    assert math.isclose(total.confidence.std, math.sqrt(sum((v - mean) ** 2 for v in pooled) / 4), abs_tol=1e-6)
    assert total.elapsed_time.count == 6 and total.elapsed_time.max == 1.0


def test_predicted_class_not_declared_by_the_model_is_appended():
    document = aggregate([make_inspection(1, T, "E1", [cls_result([cls_pred("WEIRD", 0.7)], classes=("OK",))])])
    assert document.classes == ["OK", "WEIRD"]
    assert document.total.class_counts == {"OK": 0, "WEIRD": 1}


# ------------------------------------------------------------------ detection
def test_detection_document():
    raws = [
        make_inspection(1, T, "E1", [det_result([
            det_image([box("scratch", 0.8), box("scratch", 0.35), box("dent", 0.9, threshold=0.3)]),
            det_image([]),
            det_image([box("scratch", 0.5)] * 6),                       # 6 boxes -> "5+" bin
        ])]),
        make_inspection(2, T, "E2", [det_result([
            det_image([box("dent", [0.2, 0.31], threshold=0.3)]),        # near 0.3
        ])]),
    ]
    document = aggregate(raws)
    total = document.total
    first = document.equipments[0]

    assert document.task == "det"
    assert first.prediction_count == 3 and first.box_count == 9
    assert first.class_counts == {"scratch": 8, "dent": 1}
    assert first.per_class["scratch"].threshold == 0.4
    assert first.per_class["dent"].threshold == 0.3
    assert first.per_class["scratch"].below_threshold_count == 1
    assert first.images.no_box_count == 1
    assert first.images.boxes_per_image.histogram == [1, 0, 0, 1, 0, 1]
    assert first.images.boxes_per_image.sum == 9 and first.images.boxes_per_image.sum_sq == 45
    assert first.images.boxes_per_image.max == 6 and first.images.boxes_per_image.mean == 3.0
    assert first.images.boxes_per_image_by_class == {"scratch": round(8 / 3, DIGITS), "dent": round(1 / 3, DIGITS)}

    assert total.prediction_count == 4 and total.box_count == 10
    assert total.images.boxes_per_image.histogram == [1, 1, 0, 1, 0, 1]
    assert total.per_class["dent"].count == 2 and total.per_class["dent"].near_threshold_count == 1
    assert total.per_class["dent"].threshold == 0.3                        # both equipments agree
    assert total.images.boxes_per_image_by_class["dent"] == 0.5


def test_total_drops_per_class_threshold_when_equipments_disagree():
    raws = [
        make_inspection(1, T, "E1", [det_result([det_image([box("scratch", 0.8, threshold=0.4)])])]),
        make_inspection(2, T, "E2", [det_result([det_image([box("scratch", 0.8, threshold=0.6)])])]),
    ]
    document = aggregate(raws)
    assert [entry.per_class["scratch"].threshold for entry in document.equipments] == [0.4, 0.6]
    assert document.total.per_class["scratch"].threshold is None


# ------------------------------------------------------------------ fan-out rules
def test_same_model_twice_in_one_inspection_counts_both_but_one_inspection():
    twice = make_inspection(1, T, "E1", [cls_result([cls_pred("OK", 0.9)]), cls_result([cls_pred("NG", 0.4)])])
    document = aggregate([twice])
    assert document.total.inspection_count == 1
    assert document.total.prediction_count == 2
    assert document.total.class_counts == {"OK": 1, "NG": 1}


def test_first_writer_wins_conflicts_are_reported_once():
    raws = [
        make_inspection(1, T, "E1", [cls_result([cls_pred("OK", 0.9)])], location="L1", local_timezone="Asia/Bangkok"),
        make_inspection(2, T, "E1", [cls_result([cls_pred("OK", 0.9)])], location="L2", local_timezone="Asia/Seoul"),
        make_inspection(3, T, "E1", [cls_result([cls_pred("OK", 0.9)])], location="L3", local_timezone="Asia/Seoul"),
    ]
    inspections = [parse(raw) for raw in raws]
    identity = model_identity(inspections[0].inspection_result.ai_results[0])
    key = build_key(inspections[0], identity, Granularity.HOURLY, MONDAY.replace(hour=5))
    accumulator = DocumentAccumulator(key, BINS, DIGITS)
    for inspection in inspections:
        accumulator.add(inspection, inspection.inspection_result.ai_results)
    document = accumulator.build(datetime(2026, 9, 21, 6, 1), MONDAY.replace(hour=6))

    assert document.local_timezone == "Asia/Bangkok"
    assert document.equipments[0].location == "L1"
    assert len(accumulator.conflicts) == 2                                  # one per field, not per inspection


# ------------------------------------------------------------------ pipeline
def build_two_days() -> list[dict]:
    """Two days of hourly traffic, two equipments, plus a seg-only model, a deleted document and an odd mode."""
    rng = random.Random(0)
    docs = []
    next_id = 1
    for day in (MONDAY, MONDAY + timedelta(days=1)):
        for hour in (1, 2, 5, 13, 22):                       # gaps on purpose: those hours get no document
            for equipment in ("SEHC_Side_VM07", "SEHC_Side_VM08"):
                for _ in range(3):
                    created = day + timedelta(hours=hour, minutes=rng.randint(0, 59))
                    predictions = [
                        cls_pred("OK" if rng.random() < 0.9 else "NG", round(rng.uniform(0.3, 1.0), 4),
                                 threshold=0.5, elapsed=round(rng.uniform(0.1, 0.4), 3))
                        for _ in range(rng.randint(5, 12))
                    ]
                    images = [det_image([box("scratch", round(rng.uniform(0.2, 1.0), 4))
                                         for _ in range(rng.randint(0, 7))]) for _ in range(4)]
                    docs.append(make_inspection(next_id, created, equipment,
                                                [cls_result(predictions), det_result(images), seg_result()]))
                    next_id += 1
    # a deleted document, excluded by default
    docs.append(make_inspection(next_id, MONDAY + timedelta(hours=5, minutes=1), "SEHC_Side_VM07",
                                [cls_result([cls_pred("NG", 0.11)])], is_deleted=True))
    next_id += 1
    # a mode the enum does not know still gets its own document
    docs.append(make_inspection(next_id, MONDAY + timedelta(hours=5, minutes=2), "SEHC_Side_VM07",
                                [cls_result([cls_pred("OK", 0.99)])], mode="calibration"))
    next_id += 1
    # no createdAt: skipped
    docs.append(make_inspection(next_id, None, "SEHC_Side_VM07", [cls_result([cls_pred("OK", 0.99)])]))
    return docs


def test_pipeline_end_to_end():
    docs = build_two_days()
    start, end = MONDAY.replace(hour=4, minute=30), MONDAY + timedelta(days=1, hours=3)
    statistics, report = run_pipeline(docs, start, end)

    assert report.granularities_processed == 4
    assert report.inspections_no_created_at == 0                    # the null-createdAt doc never matches a window
    assert report.ai_results_seg_skipped > 0
    assert report.documents_failed == 0
    assert report.documents_created == report.documents_written == len(statistics.store)
    assert report.snapped_windows["hourly"] == "[2026-09-21T04:00:00, 2026-09-22T03:00:00)"
    assert report.snapped_windows["weekly"] == "[2026-09-21T00:00:00, 2026-09-28T00:00:00)"

    # documents exist only for periods that had an inspection
    hourly = statistics.by_granularity(Granularity.HOURLY)
    hours = sorted({key.start_date for key in hourly})
    assert hours == [MONDAY.replace(hour=5), MONDAY.replace(hour=13), MONDAY.replace(hour=22),
                     MONDAY + timedelta(days=1, hours=1), MONDAY + timedelta(days=1, hours=2)]

    # one document per (model, task, mode) per period: cls + det + the calibration-mode cls at 05:00
    at_five = [key for key in hourly if key.start_date == MONDAY.replace(hour=5)]
    assert sorted((key.task, key.mode) for key in at_five) == [("cls", "calibration"), ("cls", "production"),
                                                               ("det", "production")]
    assert not any(key.task == "seg" for key in statistics.store)

    # the deleted document did not contribute
    production_cls = statistics.store[next(k for k in at_five if k.mode == "production" and k.task == "cls")]
    assert 0.11 > production_cls["total"]["confidence"]["min"] or production_cls["total"]["confidence"]["min"] >= 0.3

    # weekly and daily periods each got their documents too
    assert len(statistics.by_granularity(Granularity.WEEKLY)) == 3      # cls + det + calibration cls, one week
    daily = statistics.by_granularity(Granularity.DAILY)
    assert sorted({key.start_date for key in daily}) == [MONDAY, MONDAY + timedelta(days=1)]

    # a second run over the same window replaces every document with an identical payload (minus computedAt)
    pipeline = StatisticsPipeline(FakeInspections(docs), statistics, BINS, DIGITS, batch_size=10)
    before = {key: {k: v for k, v in doc.items() if k != "computedAt"} for key, doc in statistics.store.items()}
    report2 = pipeline.run(InspectionFilter(start=start, end=end), list(Granularity))
    after = {key: {k: v for k, v in doc.items() if k != "computedAt"} for key, doc in statistics.store.items()}
    assert report2.documents_updated == report2.documents_written == len(before)
    assert before == after


def test_weekly_document_count_in_end_to_end_run():
    docs = build_two_days()
    statistics, _ = run_pipeline(docs, MONDAY, MONDAY + timedelta(days=2), granularities=[Granularity.WEEKLY])
    keys = sorted((key.task, key.mode) for key in statistics.store)
    assert keys == [("cls", "calibration"), ("cls", "production"), ("det", "production")]


def test_hourly_documents_add_up_exactly_to_the_daily_one():
    docs = build_two_days()
    statistics, _ = run_pipeline(docs, MONDAY, MONDAY + timedelta(days=1),
                                 granularities=[Granularity.HOURLY, Granularity.DAILY], mode=["production"])

    assert all(key.mode == "production" for key in statistics.store)     # the mode filter reached the fake

    for task in ("cls", "det"):
        daily_key = next(k for k in statistics.by_granularity(Granularity.DAILY) if k.task == task)
        daily = statistics.store[daily_key]["total"]
        hourly = [doc["total"] for k, doc in statistics.by_granularity(Granularity.HOURLY).items()
                  if k.task == task and k.start_date < MONDAY + timedelta(days=1)]
        assert len(hourly) == 5

        def total_of(path):
            return sum(_dig(doc, path) for doc in hourly)

        assert daily["inspectionCount"] == total_of("inspectionCount")
        assert daily["predictionCount"] == total_of("predictionCount")
        assert daily["confidence"]["count"] == total_of("confidence.count")
        assert math.isclose(daily["confidence"]["sum"], total_of("confidence.sum"), abs_tol=1e-5)
        assert math.isclose(daily["confidence"]["sumSq"], total_of("confidence.sumSq"), abs_tol=1e-5)
        assert daily["confidence"]["belowThresholdCount"] == total_of("confidence.belowThresholdCount")
        assert daily["confidence"]["histogram"] == [sum(col) for col in zip(*(d["confidence"]["histogram"] for d in hourly))]
        for name in daily["classCounts"]:
            assert daily["classCounts"][name] == sum(d["classCounts"][name] for d in hourly)
            assert daily["perClass"][name]["histogram"] == [
                sum(col) for col in zip(*(d["perClass"][name]["histogram"] for d in hourly))]
        assert daily["confidence"]["min"] == min(d["confidence"]["min"] for d in hourly)
        assert daily["confidence"]["max"] == max(d["confidence"]["max"] for d in hourly)
        # mean and std recomputed from the additive fields match the stored ones
        count, total, total_sq = daily["confidence"]["count"], daily["confidence"]["sum"], daily["confidence"]["sumSq"]
        assert math.isclose(daily["confidence"]["mean"], total / count, abs_tol=1e-5)
        assert math.isclose(daily["confidence"]["std"], math.sqrt(total_sq / count - (total / count) ** 2), abs_tol=1e-4)
        if task == "det":
            assert daily["boxCount"] == total_of("boxCount")
            assert daily["images"]["noBoxCount"] == total_of("images.noBoxCount")
            assert daily["images"]["boxesPerImage"]["histogram"] == [
                sum(col) for col in zip(*(d["images"]["boxesPerImage"]["histogram"] for d in hourly))]


def test_partial_filters_are_named():
    filters = InspectionFilter(start=MONDAY, end=MONDAY + timedelta(days=1), gbm=["SEHC"], equipment_id=["X"],
                               product_id=["P"])
    assert filters.partial_filters() == ["equipment_id", "product_id"]
    assert InspectionFilter(start=MONDAY, end=MONDAY + timedelta(days=1), model_name=["a"]).partial_filters() == []


def test_equipment_identity_fields_come_first_in_the_stored_entry():
    document = aggregate([make_inspection(1, T, "E1", [cls_result([cls_pred("OK", 0.9)])])])
    entry = to_mongo(document.model_dump(by_alias=True))["equipments"][0]
    assert list(entry)[:3] == ["equipmentId", "location", "backend"]
    assert "threshold" not in entry


def _dig(doc: dict, path: str):
    for part in path.split("."):
        doc = doc[part]
    return doc


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"ok    {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL  {name}: {error}")
    sys.exit(1 if failures else 0)
