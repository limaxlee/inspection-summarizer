"""Command line entry point. Every filter and knob is explicit."""
import argparse
import json
import logging
import sys
from datetime import datetime

from inspection_statistics import config
from inspection_statistics.common.constants import ModelTasks, InspectionMode, UploadStatus, Granularity
from inspection_statistics.common.datetime_util import snap_window
from inspection_statistics.db import MongoConnection
from inspection_statistics.filters import InspectionFilter
from inspection_statistics.pipeline import StatisticsPipeline
from inspection_statistics.repositories import InspectionRepository, StatisticsRepository
from inspection_statistics.schemas.statistics import Bins

logger = logging.getLogger("inspection_statistics")


def parse_datetime(value: str) -> datetime:
    """Accept either a date (2026-06-20) or a full ISO timestamp."""
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO date/datetime: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspection-statistics",
        description="Aggregate inspection documents into per-period inspectionStatistics documents.",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="Connect, dump one raw inspection document, list indexes. Read-only.")

    run = subparsers.add_parser("run", help="Compute or recompute statistics documents.")

    window = run.add_argument_group("time window (required; snapped outward to full periods)")
    window.add_argument("--start", required=True, type=parse_datetime,
                        help="Inclusive UTC start of metadata.createdAt, e.g. 2026-06-20")
    window.add_argument("--end", required=True, type=parse_datetime,
                        help="Exclusive UTC end of metadata.createdAt, e.g. 2026-06-21")
    window.add_argument("--granularity", nargs="+", choices=[item.value for item in Granularity],
                        default=[item.value for item in Granularity],
                        help="Period types to compute. Default: all four.")

    metadata = run.add_argument_group("metadata filters (repeat a flag for OR)")
    metadata.add_argument("--gbm", nargs="+")
    metadata.add_argument("--process", nargs="+")
    metadata.add_argument("--location", nargs="+")
    metadata.add_argument("--equipment-id", nargs="+")
    metadata.add_argument("--product-id", nargs="+")
    metadata.add_argument("--mode", nargs="+",
                          help=f"metadata.mode, e.g. {', '.join(item.value for item in InspectionMode)}")
    metadata.add_argument("--local-timezone", nargs="+")

    result = run.add_argument_group("result filters")
    result.add_argument("--ai-model", nargs="+", help='Full aiModel string, e.g. "sidebottom/sidebottom_250429"')
    result.add_argument("--model-name", nargs="+", help="Model name only (the part before the last slash)")
    result.add_argument("--task", nargs="+", choices=[item.value for item in ModelTasks],
                        help="seg never produces a document; listing it only affects document selection.")
    result.add_argument("--conclusion", nargs="+")
    result.add_argument("--upload-status", nargs="+", choices=[item.value for item in UploadStatus])
    result.add_argument("--include-deleted", action="store_true",
                        help="By default documents with isDeleted=true are excluded.")

    knobs = run.add_argument_group("execution")
    knobs.add_argument("--batch-size", type=int, default=config.READ_BATCH_SIZE, help="Cursor batch size.")
    knobs.add_argument("--limit", type=int,
                       help="Stop after this many inspection documents PER GRANULARITY. Produces partial documents.")
    knobs.add_argument("--dry-run", action="store_true", help="Compute everything, write nothing.")
    knobs.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    knobs.add_argument("--report-json", help="Write the run report to this path.")

    return parser


def open_connections() -> tuple[MongoConnection, MongoConnection]:
    missing = config.validate()
    if missing:
        logger.error("Unfilled constants in inspection_statistics/config.py: " + ", ".join(missing))
        sys.exit(2)

    source = MongoConnection.connect(
        config.mongo_uri(), config.DB_NAME, config.INSPECTIONS_COLLECTION, label="inspections",
    )
    target = source.sibling(config.STATISTICS_COLLECTION, label="statistics")
    return source, target


def bins_from_config() -> Bins:
    return Bins(
        confidence_edges=config.CONFIDENCE_EDGES,
        near_threshold_margin=config.NEAR_THRESHOLD_MARGIN,
        boxes_per_image_max=config.BOXES_PER_IMAGE_MAX,
    )


def command_verify() -> int:
    source, target = open_connections()
    with source:
        if not source.ping():
            return 1

        inspections = InspectionRepository(source)
        statistics = StatisticsRepository(target)
        print(f"\nsource {source.target}: {inspections.count({})} documents")
        print(f"target {target.target}: {statistics.count()} documents")
        indexes = statistics.index_names()
        print(f"target indexes: {', '.join(indexes) if indexes else '(none yet -- created on the first real run)'}\n")

        raw = inspections.find_one()
        if raw is None:
            print("Source collection is empty -- nothing to inspect.")
            return 0

        print("Top-level field names as stored:")
        for name in raw:
            print(f"  {name}")

        container = "inspectionResults" if "inspectionResults" in raw else "inspectionResult"
        print(f"\nResolved aiResults path: {container}.aiResults")
        if container not in raw:
            print("  WARNING: neither key was found -- check the collection.")

        print("\nSample document (truncated):")
        print(json.dumps(raw, indent=2, default=str)[:4000])
        return 0


def command_run(args: argparse.Namespace) -> int:
    filters = InspectionFilter(
        start=args.start,
        end=args.end,
        gbm=args.gbm,
        process=args.process,
        location=args.location,
        equipment_id=args.equipment_id,
        product_id=args.product_id,
        mode=args.mode,
        local_timezone=args.local_timezone,
        ai_model=args.ai_model,
        model_name=args.model_name,
        task=args.task,
        conclusion=args.conclusion,
        upload_status=args.upload_status,
        include_deleted=args.include_deleted,
    )
    granularities = [Granularity(value) for value in dict.fromkeys(args.granularity)]

    source, target = open_connections()
    with source:
        if not source.ping():
            return 1

        inspections = InspectionRepository(source)
        statistics = StatisticsRepository(target)

        print("\nFilters:")
        print(filters.describe())
        print("\nResolved query (window varies per period):")
        print(json.dumps(filters.to_mongo_query(), indent=2, default=str))

        print("\nWindows actually read, snapped outward to complete periods:")
        total_matched = 0
        for granularity in granularities:
            snapped_start, snapped_end = snap_window(filters.start, filters.end, granularity)
            matched = inspections.count(filters.to_mongo_query(snapped_start, snapped_end))
            total_matched = max(total_matched, matched)
            print(f"  {granularity.value:<7}: [{snapped_start.isoformat()}, {snapped_end.isoformat()})"
                  f"  ->  {matched} inspection documents")

        print(f"\nSource: {source.target}   ->   Target: {target.target}")
        print(f"Mode  : {'DRY RUN (no writes)' if args.dry_run else 'WRITE'}")

        partial = filters.partial_filters()
        if partial:
            print(f"\nWARNING: {', '.join(partial)} are not part of the document key. The documents written "
                  f"by this run cover ONLY the selected subset and will be replaced wholesale by the next run "
                  f"over the same periods.")
        if args.limit:
            print(f"\nWARNING: --limit {args.limit} stops reading mid-period, so the documents written are partial.")
        print()

        if total_matched == 0:
            print("Nothing to do.")
            return 0

        if not args.dry_run and not args.yes and not confirm():
            print("Aborted.")
            return 1

        if not args.dry_run:
            statistics.ensure_indexes(config.RETENTION_SECONDS)

        pipeline = StatisticsPipeline(
            inspections=inspections,
            statistics=statistics,
            bins=bins_from_config(),
            round_digits=config.ROUND_DIGITS,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
        report = pipeline.run(filters, granularities, limit=args.limit)

        print("\n" + report.render(args.dry_run))

        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as handle:
                json.dump(report.as_dict(), handle, indent=2, default=str)
            print(f"\nReport written to {args.report_json}")

        return 0


def confirm() -> bool:
    try:
        return input("Proceed and write to the statistics collection? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.command == "verify":
        return command_verify()
    return command_run(args)


if __name__ == "__main__":
    sys.exit(main())
