"""Orchestration: read inspections from A, fold them into summaries, write to B.

The run walks the requested window one UTC day at a time. Because ``date`` is
part of the summary key, a day boundary partitions the key space cleanly: every
bucket touched inside a day is loaded once, folded over every matching
inspection, and written exactly once. That turns a read+write per inspection
into a read+write per bucket.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import ValidationError

from inspection_summarizer.common.datetime_util import iter_day_ranges
from inspection_summarizer.config import INSPECTION_IDS_WARN_THRESHOLD
from inspection_summarizer.filters import InspectionFilter
from inspection_summarizer.keys import SummaryKey, build_key
from inspection_summarizer.repositories import InspectionRepository, SummaryRepository
from inspection_summarizer.schemas.inspections import InspectionDocumentV1
from inspection_summarizer.schemas.summary import InspectionSummaryDocumentV1
from inspection_summarizer.summarizer import InspectionSummarizer

logger = logging.getLogger(__name__)

# Fields carried on the summary document but not part of its key: the summarizer
# copies them from whichever inspection creates the bucket and never updates them.
FIRST_WRITER_WINS_FIELDS = ["location", "equipment_id", "product_id", "local_timezone"]


@dataclass
class RunReport:
    days_processed: int = 0
    inspections_read: int = 0
    inspections_invalid: int = 0
    inspections_no_created_at: int = 0
    inspections_no_id: int = 0
    ai_results_seen: int = 0
    ai_results_filtered_out: int = 0
    merges_applied: int = 0
    merges_already_present: int = 0
    predictions_counted: int = 0
    predictions_not_sampleable: int = 0
    buckets_created: int = 0
    buckets_updated: int = 0
    buckets_written: int = 0
    metadata_conflicts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items()}

    def render(self, dry_run: bool) -> str:
        mode = "DRY RUN -- nothing was written" if dry_run else "COMPLETED"
        lines = [
            f"Run {mode}",
            f"  days processed            : {self.days_processed}",
            f"  inspections read          : {self.inspections_read}",
            f"    invalid (skipped)       : {self.inspections_invalid}",
            f"    no createdAt (skipped)  : {self.inspections_no_created_at}",
            f"    no _id (skipped)        : {self.inspections_no_id}",
            f"  aiResults seen            : {self.ai_results_seen}",
            f"    filtered out            : {self.ai_results_filtered_out}",
            f"  merges applied            : {self.merges_applied}",
            f"    already present         : {self.merges_already_present}",
            f"  predictions counted       : {self.predictions_counted}",
            f"    not sampleable          : {self.predictions_not_sampleable}",
            f"  buckets created           : {self.buckets_created}",
            f"  buckets updated           : {self.buckets_updated}",
            f"  buckets written           : {self.buckets_written}",
        ]
        if self.metadata_conflicts:
            lines.append(f"  metadata conflicts        : {len(self.metadata_conflicts)} (first-writer-wins)")
            lines.extend(f"      {conflict}" for conflict in self.metadata_conflicts[:10])
            if len(self.metadata_conflicts) > 10:
                lines.append(f"      ... and {len(self.metadata_conflicts) - 10} more")
        return "\n".join(lines)


@dataclass
class _Bucket:
    document: InspectionSummaryDocumentV1 | None
    existed: bool
    dirty: bool = False


class SummaryPipeline:
    def __init__(
            self,
            inspections: InspectionRepository,
            summaries: SummaryRepository,
            summarizer: InspectionSummarizer,
            batch_size: int,
            dry_run: bool = False,
    ):
        self._inspections = inspections
        self._summaries = summaries
        self._summarizer = summarizer
        self._batch_size = batch_size
        self._dry_run = dry_run
        self._warned_buckets: set[SummaryKey] = set()

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
        buckets: dict[SummaryKey, _Bucket] = {}
        processed = 0

        logger.info(f"--- {day_start:%Y-%m-%d} [{day_start.isoformat()}, {day_end.isoformat()})")

        for raw in self._inspections.iter_documents(query, self._batch_size, limit):
            processed += 1
            report.inspections_read += 1

            inspection = self._parse(raw, report)
            if inspection is None:
                continue

            self._fold(inspection, filters, buckets, report)

        self._flush(buckets, report)
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

    def _fold(
            self,
            inspection: InspectionDocumentV1,
            filters: InspectionFilter,
            buckets: dict[SummaryKey, _Bucket],
            report: RunReport,
    ):
        # One inspection fans out to one bucket per entry in inspectionResults.aiResults.
        for summary_base, ai_result in self._summarizer.get_summary_base(inspection):
            report.ai_results_seen += 1

            if not filters.matches_ai_result(ai_result):
                report.ai_results_filtered_out += 1
                continue

            self._count_sampleability(inspection, ai_result, report)

            key = build_key(summary_base)
            if key not in buckets:
                existing = self._summaries.find_by_key(key)
                buckets[key] = _Bucket(document=existing, existed=existing is not None)

            bucket = buckets[key]
            self._check_metadata_conflict(key, bucket.document, inspection, report)

            updated = self._summarizer.register(summary_base, bucket.document, inspection, ai_result)

            # register() returns None both for "already merged" and "no id" --
            # never assign that back over the accumulated document.
            if updated is None:
                report.merges_already_present += 1
                continue

            bucket.document = updated
            bucket.dirty = True
            report.merges_applied += 1

    def _count_sampleability(self, inspection: InspectionDocumentV1, ai_result, report: RunReport):
        """Predictions still contribute to statistics when they cannot be sampled."""
        file_count = len(inspection.file_list)
        for prediction in ai_result.predictions:
            report.predictions_counted += 1
            if prediction.file_index is None or prediction.file_index >= file_count:
                report.predictions_not_sampleable += 1

    def _check_metadata_conflict(
            self,
            key: SummaryKey,
            document: InspectionSummaryDocumentV1 | None,
            inspection: InspectionDocumentV1,
            report: RunReport,
    ):
        if document is None or key in self._warned_buckets:
            return

        for name in FIRST_WRITER_WINS_FIELDS:
            stored = getattr(document, name)
            incoming = getattr(inspection.metadata, name)
            if stored != incoming:
                message = f"{key} -> {name}: keeping {stored!r}, ignoring {incoming!r}"
                report.metadata_conflicts.append(message)
                logger.warning(message)
                self._warned_buckets.add(key)

    def _flush(self, buckets: dict[SummaryKey, _Bucket], report: RunReport):
        for key, bucket in buckets.items():
            if not bucket.dirty or bucket.document is None:
                continue

            if bucket.existed:
                report.buckets_updated += 1
            else:
                report.buckets_created += 1

            self._warn_on_size(key, bucket.document)

            action = "would write" if self._dry_run else "writing"
            logger.info(f"{action} {key} :: {self._describe(bucket.document)}")

            if self._dry_run:
                continue

            if self._summaries.upsert(key, bucket.document):
                report.buckets_written += 1

    def _warn_on_size(self, key: SummaryKey, document: InspectionSummaryDocumentV1):
        count = len(document.inspection_ids)
        if count >= INSPECTION_IDS_WARN_THRESHOLD:
            logger.warning(
                f"{key} holds {count} inspectionIds -- approaching the 16MB document limit. "
                f"Consider moving processed ids into a side collection."
            )

    def _describe(self, document: InspectionSummaryDocumentV1) -> str:
        counts = document.statistics.data_count
        return (f"inspections={len(document.inspection_ids)} "
                f"dataCount={counts} "
                f"classes={document.classes} "
                f"samples={ {name: len(items) for name, items in document.samples.items()} }")
