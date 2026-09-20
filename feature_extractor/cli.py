"""Command line entry point. Every filter and knob is explicit."""
import argparse
import json
import logging
import sys
from datetime import datetime

from feature_extractor import config
from feature_extractor.common.constants import ModelTasks, InspectionMode, UploadStatus
from feature_extractor.db import MongoConnection
from feature_extractor.extractor import FeatureExtractor, filename_of
from feature_extractor.feature_api import FeatureApiClient
from feature_extractor.filters import InspectionFilter
from feature_extractor.imaging import image_size, normalize_bbox
from feature_extractor.milvus_store import MilvusStore
from feature_extractor.object_storage import ObjectStorage
from feature_extractor.pipeline import FeaturePipeline
from feature_extractor.repositories import InspectionRepository
from feature_extractor.schemas.inspections import InspectionDocumentV1

logger = logging.getLogger("feature_extractor")


def parse_datetime(value: str) -> datetime:
    """Accept either a date (2026-06-20) or a full ISO timestamp."""
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO date/datetime: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feature-extractor",
        description="Extract feature vectors for every prediction in the inspection documents and store them in Milvus.",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="Check MongoDB, object storage, the feature API and Milvus. Writes nothing.")

    run = subparsers.add_parser("run", help="Extract feature vectors and insert them into Milvus.")

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
    knobs.add_argument("--batch-size", type=int, default=config.READ_BATCH_SIZE, help="Cursor batch size.")
    knobs.add_argument("--limit", type=int, help="Stop after this many inspection documents.")
    knobs.add_argument("--force", action="store_true",
                       help="Insert even when a row with the same data_uri already exists in the collection.")
    knobs.add_argument("--dry-run", action="store_true",
                       help="Read and count only: no downloads, no API calls, no Milvus writes.")
    knobs.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    knobs.add_argument("--report-json", help="Write the run report to this path.")

    return parser


def open_clients() -> tuple[MongoConnection, ObjectStorage, FeatureApiClient, MilvusStore]:
    missing = config.validate()
    if missing:
        logger.error("Unfilled constants in feature_extractor/config.py: " + ", ".join(missing))
        sys.exit(2)

    source = MongoConnection(config.source_uri(), config.SOURCE_DB_NAME, config.SOURCE_COLLECTION, label="mongo/source")
    storage = ObjectStorage(
        config.OBJECT_STORAGE_ENDPOINT, config.OBJECT_STORAGE_BUCKET,
        config.OBJECT_STORAGE_ACCESS_KEY, config.OBJECT_STORAGE_SECRET_KEY,
    )
    api = FeatureApiClient(config.extractor_url(), config.EXTRACTOR_TIMEOUT_SECONDS, config.FEATURE_DIM)
    store = MilvusStore(
        uri=config.milvus_uri(),
        user=config.MILVUS_USER,
        password=config.MILVUS_PASSWORD,
        db_name=config.MILVUS_DB_NAME,
        dim=config.FEATURE_DIM,
        metric_type=config.MILVUS_METRIC_TYPE,
        index_type=config.MILVUS_INDEX_TYPE,
        varchar_max_length=config.VARCHAR_MAX_LENGTH,
        collection_prefix=config.COLLECTION_PREFIX,
        object_suffix=config.OBJECT_COLLECTION_SUFFIX,
    )
    return source, storage, api, store


def command_verify() -> int:
    source, storage, api, store = open_clients()
    with source, storage, api, store:
        ok = source.ping()
        ok = storage.ping() and ok
        ok = store.ping() and ok
        print(f"\nfeature API : {api.url} (checked below with a real image)")
        if not ok:
            return 1

        inspections = InspectionRepository(source)
        print(f"source {source.target}: {inspections.count({})} documents")
        print(f"milvus {store.target}: {len(store.client.list_collections())} collections\n")

        raw = inspections.find_one({"inspectionResults.aiResults.task": {"$in": ["cls", "det"]}})
        if raw is None:
            print("No inspection with a cls/det aiResult found -- nothing more to check.")
            return 0

        print("Top-level field names as stored:")
        for name in raw:
            print(f"  {name}")
        container = "inspectionResults" if "inspectionResults" in raw else "inspectionResult"
        print(f"\nResolved aiResults path: {container}.aiResults")
        if container not in raw:
            print("  WARNING: neither key was found -- check the collection.")

        try:
            inspection = InspectionDocumentV1.model_validate(raw)
        except Exception as error:
            print(f"\nSample document does not parse: {error}")
            return 1

        return _probe_sample(inspection, storage, api, store)


def _probe_sample(inspection: InspectionDocumentV1, storage: ObjectStorage, api: FeatureApiClient,
                  store: MilvusStore) -> int:
    """Download one real image, run it through the API, and show what run would do with it."""
    for ai_result in inspection.inspection_result.ai_results:
        if ai_result.task == ModelTasks.SEGMENTATION:
            continue
        prediction = next((item for item in ai_result.predictions if item.data_path), None)
        if prediction is None:
            continue

        model_name, _, model_version = ai_result.ai_model.rpartition("/")
        process = inspection.metadata.process
        print(f"\nSample aiResult: {ai_result.ai_model} ({ai_result.task.value})")
        print(f"  main collection   : {store.collection_name(process, model_name, model_version)}")
        if ai_result.task == ModelTasks.DETECTION:
            print(f"  object collection : {store.collection_name(process, model_name, model_version, objects=True)}")
        print(f"  dataPath          : {prediction.data_path}")
        print(f"  filename          : {filename_of(prediction.data_path)}")

        try:
            image = storage.get_bytes(prediction.data_path)
        except Exception as error:
            print(f"  download          : FAILED -- {error}")
            return 1
        width, height = image_size(image)
        print(f"  download          : OK, {len(image)} bytes, {width}x{height} px")

        if ai_result.task == ModelTasks.DETECTION and prediction.detections:
            bbox = prediction.detections[0].bbox
            print(f"  first bbox        : {bbox} -> pixel box {normalize_bbox(bbox, width, height)}")

        try:
            vector = api.extract(image, filename_of(prediction.data_path))
        except Exception as error:
            print(f"  feature API       : FAILED -- {error}")
            return 1
        print(f"  feature API       : OK, vector length {len(vector)} (FEATURE_DIM={config.FEATURE_DIM})")
        return 0

    print("\nSample document has no cls/det prediction with a dataPath.")
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

    source, storage, api, store = open_clients()
    with source, storage, api, store:
        if not source.ping():
            return 1
        if not args.dry_run and not (storage.ping() and store.ping()):
            return 1

        inspections = InspectionRepository(source)

        query = filters.to_mongo_query()
        matched = inspections.count(query)

        mode = "DRY RUN (no downloads, API calls or writes)" if args.dry_run else "WRITE"
        if args.force:
            mode += " (force: ignore existing data_uri)"

        print("\nFilters:")
        print(filters.describe())
        print("\nResolved query:")
        print(json.dumps(query, indent=2, default=str))
        print(f"\nMatching inspection documents: {matched}")
        print(f"Source : {source.target}")
        print(f"Images : {storage.target}")
        print(f"API    : {api.url}")
        print(f"Milvus : {store.target}")
        print(f"Mode   : {mode}\n")

        if matched == 0:
            print("Nothing to do.")
            return 0

        if not args.dry_run and not args.yes and not confirm():
            print("Aborted.")
            return 1

        extractor = FeatureExtractor(
            storage=storage,
            api=api,
            store=store,
            dry_run=args.dry_run,
            force=args.force,
            crop_format=config.CROP_IMAGE_FORMAT,
        )
        pipeline = FeaturePipeline(inspections=inspections, extractor=extractor, batch_size=args.batch_size)
        report = pipeline.run(filters, limit=args.limit)

        print("\n" + report.render(args.dry_run))

        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as handle:
                json.dump(report.as_dict(), handle, indent=2, default=str)
            print(f"\nReport written to {args.report_json}")

        return 0


def confirm() -> bool:
    try:
        return input("Proceed and write to Milvus? [y/N] ").strip().lower() in {"y", "yes"}
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
