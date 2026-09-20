"""Orchestration: read inspections from MongoDB, run the extractor per aiResult.

The run walks the requested window one UTC day at a time, exactly like the
summarizer. Every aiResult that survives the filters is handed to
``FeatureExtractor.process``, which does the downloading, the API calls and the
Milvus writes. Nothing is buffered across inspections.
"""
import logging
from datetime import datetime

from pydantic import ValidationError

from feature_extractor.common.datetime_util import iter_day_ranges
from feature_extractor.extractor import FeatureExtractor
from feature_extractor.filters import InspectionFilter
from feature_extractor.report import RunReport
from feature_extractor.repositories import InspectionRepository
from feature_extractor.schemas.inspections import InspectionDocumentV1

logger = logging.getLogger(__name__)


class FeaturePipeline:
    def __init__(self, inspections: InspectionRepository, extractor: FeatureExtractor, batch_size: int):
        self._inspections = inspections
        self._extractor = extractor
        self._batch_size = batch_size

    def run(self, filters: InspectionFilter, limit: int | None = None) -> RunReport:
        report = RunReport()
        remaining = limit

        for day_start, day_end in iter_day_ranges(filters.start, filters.end):
            if remaining is not None and remaining <= 0:
                break

            processed = self._run_day(filters, day_start, day_end, report, remaining)
            report.days_processed += 1
            if remaining is not None:
                remaining -= processed

        return report

    def _run_day(
            self,
            filters: InspectionFilter,
            day_start: datetime,
            day_end: datetime,
            report: RunReport,
            limit: int | None,
    ) -> int:
        query = filters.to_mongo_query(day_start, day_end)
        processed = 0

        logger.info(f"--- {day_start:%Y-%m-%d} [{day_start.isoformat()}, {day_end.isoformat()})")

        for raw in self._inspections.iter_documents(query, self._batch_size, limit):
            processed += 1
            report.inspections_read += 1

            inspection = self._parse(raw, report)
            if inspection is None:
                continue

            self._fan_out(inspection, filters, report)

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

        if not inspection.inspection_id:
            report.inspections_no_id += 1
            logger.warning("Skipping inspection without an _id")
            return None

        return inspection

    def _fan_out(self, inspection: InspectionDocumentV1, filters: InspectionFilter, report: RunReport):
        """One inspection -> one extractor call per entry in inspectionResults.aiResults."""
        for ai_result in inspection.inspection_result.ai_results:
            report.ai_results_seen += 1

            if not filters.matches_ai_result(ai_result):
                report.ai_results_filtered_out += 1
                continue

            try:
                self._extractor.process(inspection, ai_result, report)
            except Exception as error:  # one broken aiResult must not abort the day
                report.ai_results_errors += 1
                logger.exception(f"aiResult {ai_result.ai_model} of inspection {inspection.inspection_id} "
                                 f"failed: {error}")
