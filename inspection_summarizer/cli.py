"""Command line entry point. Every filter and knob is explicit."""
import argparse
import json
import logging
import sys
from datetime import datetime

from inspection_summarizer import config
from inspection_summarizer.common.constants import ModelTasks, InspectionMode, UploadStatus
from inspection_summarizer.db import MongoConnection
from inspection_summarizer.filters import InspectionFilter
from inspection_summarizer.pipeline import SummaryPipeline
from inspection_summarizer.repositories import InspectionRepository, SummaryRepository
from inspection_summarizer.summarizer import InspectionSummarizer

logger = logging.getLogger("inspection_summarizer")


def parse_datetime(value: str) -> datetime:
    """Accept either a date (2026-06-20) or a full ISO timestamp."""
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO date/datetime: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspection-summarizer",
        description="Fold inspection documents from server A into inspection summary documents on server B.",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="Connect to both servers and dump one raw inspection document. Read-only.")

    run = subparsers.add_parser("run", help="Build or update summary documents.")

    window = run.add_argument_group("time window (required)")
    window.add_argument("--start", required=True, type=parse_datetime,
                        help="Inclusive UTC start of metadata.createdAt, e.g. 2026-06-20")
    window.add_argument("--end", required=True, type=parse_datetime,
                        help="Exclusive UTC end of metadata.createdAt, e.g. 2026-06-21")

    metadata = run.add_argument_group("metadata filters (repeat a flag for OR)")
    metadata.add_argument("--gbm", nargs="+")
    metadata.add_argument("--process", nargs="+")
    metadata.add_argument("--location", nargs="+")
    metadata.add_argument("--equipment-id", nargs="+")
    metadata.add_argument("--product-id", nargs="+")
    metadata.add_argument("--mode", nargs="+", choices=[item.value for item in InspectionMode])
    metadata.add_argument("--local-timezone", nargs="+")

    result = run.add_argument_group("result filters")
    result.add_argument("--ai-model", nargs="+", help='Full aiModel string, e.g. "sidebottom/sidebottom_250429"')
    result.add_argument("--model-name", nargs="+", help="Model name only (the part before the last slash)")
    result.add_argument("--task", nargs="+", choices=[item.value for item in ModelTasks])
    result.add_argument("--conclusion", nargs="+")
    result.add_argument("--upload-status", nargs="+", choices=[item.value for item in UploadStatus])
    result.add_argument("--include-deleted", action="store_true",
                        help="By default documents with isDeleted=true are excluded.")

    knobs = run.add_argument_group("execution")
    knobs.add_argument("--max-samples-per-class", type=int, default=config.MAX_SAMPLES_PER_CLASS)
    knobs.add_argument("--batch-size", type=int, default=config.READ_BATCH_SIZE)
    knobs.add_argument("--limit", type=int, help="Stop after this many inspection documents.")
    knobs.add_argument("--dry-run", action="store_true", help="Compute everything, write nothing.")
    knobs.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    knobs.add_argument("--report-json", help="Write the run report to this path.")

    return parser


def open_connections() -> tuple[MongoConnection, MongoConnection]:
    missing = config.validate()
    if missing:
        logger.error("Unfilled constants in config.py: " + ", ".join(missing))
        sys.exit(2)

    source = MongoConnection(config.source_uri(), config.SOURCE_DB_NAME, config.SOURCE_COLLECTION, label="A/source")
    target = MongoConnection(config.target_uri(), config.TARGET_DB_NAME, config.TARGET_COLLECTION, label="B/target")
    return source, target


def command_verify() -> int:
    source, target = open_connections()
    with source, target:
        if not (source.ping() and target.ping()):
            return 1

        inspections = InspectionRepository(source)
        total = inspections.count({})
        print(f"\nsource {source.target}: {total} documents")
        print(f"target {target.target}: {target.collection.count_documents({})} documents\n")

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

    source, target = open_connections()
    with source, target:
        if not (source.ping() and target.ping()):
            return 1

        inspections = InspectionRepository(source)
        summaries = SummaryRepository(target)

        query = filters.to_mongo_query()
        matched = inspections.count(query)

        print("\nFilters:")
        print(filters.describe())
        print("\nResolved query:")
        print(json.dumps(query, indent=2, default=str))
        print(f"\nMatching inspection documents: {matched}")
        print(f"Source: {source.target}   ->   Target: {target.target}")
        print(f"Mode  : {'DRY RUN (no writes)' if args.dry_run else 'WRITE'}\n")

        if matched == 0:
            print("Nothing to do.")
            return 0

        if not args.dry_run and not args.yes and not confirm():
            print("Aborted.")
            return 1

        if not args.dry_run:
            summaries.ensure_index()

        pipeline = SummaryPipeline(
            inspections=inspections,
            summaries=summaries,
            summarizer=InspectionSummarizer(max_samples_per_class=args.max_samples_per_class),
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
        report = pipeline.run(filters, limit=args.limit)

        print("\n" + report.render(args.dry_run))

        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as handle:
                json.dump(report.as_dict(), handle, indent=2, default=str)
            print(f"\nReport written to {args.report_json}")

        return 0


def confirm() -> bool:
    try:
        return input("Proceed and write to the target collection? [y/N] ").strip().lower() in {"y", "yes"}
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
