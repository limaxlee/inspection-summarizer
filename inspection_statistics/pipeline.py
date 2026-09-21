"""Orchestration: for each granularity, walk the window period by period, aggregate, replace.

Unlike the summarizer, nothing is folded into an existing document. A period is
recomputed from scratch from every inspection it contains and the result
REPLACES whatever was stored under the same key. That makes a run idempotent
without any bookkeeping, at the price of always reading complete periods: the
requested window is snapped outward to period boundaries.

Memory is bounded by one period of one granularity: the raw confidences of a
week are held for the weekly pass, then released.
"""
import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from inspection_statistics.aggregator import DocumentAccumulator
from inspection_statistics.common.constants import Granularity, ModelTasks
from inspection_statistics.common.datetime_util import iter_periods, snap_window
from inspection_statistics.filters import InspectionFilter
from inspection_statistics.keys import StatisticsKey, ModelIdentity, model_identity, build_key
from inspection_statistics.report import RunReport
from inspection_statistics.repositories import InspectionRepository, StatisticsRepository
from inspection_statistics.schemas.inspections import InspectionDocumentV1, AIResultData
from inspection_statistics.schemas.statistics import Bins

logger = logging.getLogger(__name__)


class StatisticsPipeline:
    def __init__(
            self,
            inspections: InspectionRepository,
            statistics: StatisticsRepository,
            bins: Bins,
            round_digits: int,
            batch_size: int,
            dry_run: bool = False,
    ):
        self._inspections = inspections
        self._statistics = statistics
        self._bins = bins
        self._digits = round_digits
        self._batch_size = batch_size
        self._dry_run = dry_run

    def run(
            self,
            filters: InspectionFilter,
            granularities: list[Granularity],
            limit: int | None = None,
    ) -> RunReport:
        report = RunReport()

        for granularity in granularities:
            snapped_start, snapped_end = snap_window(filters.start, filters.end, granularity)
            report.snapped_windows[granularity.value] = f"[{snapped_start.isoformat()}, {snapped_end.isoformat()})"
            logger.info(f"=== {granularity.value}: reading [{snapped_start.isoformat()}, {snapped_end.isoformat()})")

            remaining = limit
            for period_start, period_end in iter_periods(filters.start, filters.end, granularity):
                if remaining is not None and remaining <= 0:
                    break

                processed = self._run_period(filters, granularity, period_start, period_end, report, remaining)
                report.periods_processed += 1
                if remaining is not None:
                    remaining -= processed

            report.granularities_processed += 1

        return report

    def _run_period(
            self,
            filters: InspectionFilter,
            granularity: Granularity,
            period_start: datetime,
            period_end: datetime,
            report: RunReport,
            limit: int | None,
    ) -> int:
        query = filters.to_mongo_query(period_start, period_end)
        accumulators: dict[StatisticsKey, DocumentAccumulator] = {}
        processed = 0

        logger.info(f"--- {granularity.value} [{period_start.isoformat()}, {period_end.isoformat()})")

        for raw in self._inspections.iter_documents(query, self._batch_size, limit):
            processed += 1
            report.inspections_read += 1

            inspection = self._parse(raw, report)
            if inspection is None:
                continue

            self._fold(inspection, filters, granularity, period_start, accumulators, report)

        if not accumulators:
            report.periods_empty += 1

        self._flush(accumulators, period_end, report)
        return processed

    def _parse(self, raw: dict, report: RunReport) -> InspectionDocumentV1 | None:
        document_id = raw.get("_id")
        try:
            inspection = InspectionDocumentV1.model_validate(raw)
        except ValidationError as error:
            report.inspections_invalid += 1
            logger.warning(f"Skipping unparsable inspection {document_id}: {error}")
            return None

        if inspection.metadata.created_at is None:
            report.inspections_no_created_at += 1
            logger.warning(f"Skipping inspection {document_id}: metadata.createdAt is null")
            return None

        return inspection

    def _fold(
            self,
            inspection: InspectionDocumentV1,
            filters: InspectionFilter,
            granularity: Granularity,
            period_start: datetime,
            accumulators: dict[StatisticsKey, DocumentAccumulator],
            report: RunReport,
    ):
        # One inspection contributes to one document per distinct (aiModel, task) in its aiResults.
        # The same aiModel appearing twice lands in the same group: both entries count.
        groups: dict[ModelIdentity, list[AIResultData]] = {}

        for ai_result in inspection.inspection_result.ai_results:
            report.ai_results_seen += 1

            if ai_result.task == ModelTasks.SEGMENTATION:
                report.ai_results_seg_skipped += 1
                continue

            if not filters.matches_ai_result(ai_result):
                report.ai_results_filtered_out += 1
                continue

            groups.setdefault(model_identity(ai_result), []).append(ai_result)
            self._count_predictions(ai_result, report)

        for identity, ai_results in groups.items():
            key = build_key(inspection, identity, granularity, period_start)
            accumulator = accumulators.get(key)
            if accumulator is None:
                accumulator = accumulators[key] = DocumentAccumulator(key, self._bins, self._digits)
            accumulator.add(inspection, ai_results)

    def _count_predictions(self, ai_result: AIResultData, report: RunReport):
        report.predictions_counted += len(ai_result.predictions)
        if ai_result.task == ModelTasks.DETECTION:
            report.boxes_counted += sum(len(prediction.detections) for prediction in ai_result.predictions)

    def _flush(self, accumulators: dict[StatisticsKey, DocumentAccumulator], period_end: datetime, report: RunReport):
        computed_at = datetime.now(timezone.utc).replace(tzinfo=None)

        for key, accumulator in accumulators.items():
            document = accumulator.build(computed_at, period_end)
            report.metadata_conflicts.extend(accumulator.conflicts)
            report.confidence_missing += document.total.quality.missing_confidence_count
            report.confidence_parse_errors += document.total.quality.parse_error_count

            action = "would write" if self._dry_run else "writing"
            logger.info(f"{action} {key} :: {self._describe(document)}")

            if self._dry_run:
                report.documents_written += 1
                continue

            outcome = self._statistics.upsert(key, document)
            if outcome is None:
                report.documents_failed += 1
                continue

            report.documents_written += 1
            if outcome == "created":
                report.documents_created += 1
            else:
                report.documents_updated += 1

    @staticmethod
    def _describe(document) -> str:
        total = document.total
        boxes = f" boxes={total.box_count}" if total.box_count is not None else ""
        return (f"equipments={document.equipment_count} inspections={total.inspection_count} "
                f"predictions={total.prediction_count}{boxes} classCounts={total.class_counts} "
                f"p50={total.confidence.quantiles.p50}")
