"""The per-aiResult step.

    seg  -> skipped
    cls  -> one row per prediction in VFM_{process}_{model}_{version}
    det  -> one row per prediction in VFM_{process}_{model}_{version}
            plus one row per valid detection (cropped to its bbox) in ..._object

For each row: the object behind ``prediction.dataPath`` is downloaded, sent to
the feature API, and the returned vector is inserted into Milvus together with
the inspection metadata.
"""
import logging
from datetime import datetime

from feature_extractor.common.constants import ModelTasks
from feature_extractor.feature_api import FeatureApiClient, FeatureApiError
from feature_extractor.imaging import (
    FORMAT_CONTENT_TYPES, FORMAT_EXTENSIONS, crop, image_size, normalize_bbox,
)
from feature_extractor.milvus_store import MilvusStore
from feature_extractor.object_storage import ObjectStorage, ObjectStorageError
from feature_extractor.report import RunReport
from feature_extractor.schemas.inspections import (
    InspectionDocumentV1, ClassificationData, ObjectDetectionData, SegmentationData,
    ClassificationResult, ObjectDetectionResult,
)

logger = logging.getLogger(__name__)

AIResultData = ClassificationData | ObjectDetectionData | SegmentationData


def max_confidence(value) -> float:
    """confidence is stored as a float or a list of floats; a list collapses to its max."""
    if isinstance(value, (list, tuple)):
        return float(max(value)) if value else 0.0
    return float(value)


def filename_of(data_path: str) -> str:
    return data_path.rsplit("/", 1)[-1]


class FeatureExtractor:
    def __init__(
            self,
            storage: ObjectStorage,
            api: FeatureApiClient,
            store: MilvusStore,
            dry_run: bool = False,
            force: bool = False,
            crop_format: str = "PNG",
    ):
        self._storage = storage
        self._api = api
        self._store = store
        self._dry_run = dry_run
        self._force = force
        self._crop_format = crop_format.upper()

    # ------------------------------------------------------------------ entry
    def process(self, inspection: InspectionDocumentV1, ai_result: AIResultData, report: RunReport):
        if ai_result.task == ModelTasks.SEGMENTATION:
            report.ai_results_seg_skipped += 1
            return

        is_detection = ai_result.task == ModelTasks.DETECTION
        model_name, _, model_version = ai_result.ai_model.rpartition("/")
        process = inspection.metadata.process

        main_collection = self._store.collection_name(process, model_name, model_version)
        object_collection = (
            self._store.collection_name(process, model_name, model_version, objects=True) if is_detection else None
        )

        if not self._dry_run:
            report.collections_created += int(self._store.ensure_collection(main_collection))
            if object_collection:
                report.collections_created += int(self._store.ensure_collection(object_collection, with_bbox=True))

        base = self._base_row(inspection, ai_result)

        for prediction in ai_result.predictions:
            report.predictions_seen += 1

            data_path = prediction.data_path
            if not data_path:
                report.predictions_no_data_path += 1
                continue

            if not self._force and self._store.has_data_uri(main_collection, data_path):
                report.predictions_already_present += 1
                continue

            if is_detection:
                self._process_detection(prediction, base, main_collection, object_collection, report)
            else:
                self._process_classification(prediction, base, main_collection, report)

    # ------------------------------------------------------------------ cls
    def _process_classification(
            self,
            prediction: ClassificationResult,
            base: dict,
            collection: str,
            report: RunReport,
    ):
        row = self._prediction_row(base, prediction)
        row["prediction"] = prediction.prediction
        row["confidence"] = max_confidence(prediction.confidence)

        if self._dry_run:
            report.rows_main_inserted += 1
            logger.debug(f"would insert into {collection}: {self._describe(row)}")
            return

        image = self._download(prediction.data_path, report)
        if image is None:
            return

        vector = self._extract(image, row["filename"], None, report, "prediction")
        if vector is None:
            return

        row["feature_vector"] = vector
        self._insert(collection, [row], report, "main")

    # ------------------------------------------------------------------ det
    def _process_detection(
            self,
            prediction: ObjectDetectionResult,
            base: dict,
            main_collection: str,
            object_collection: str,
            report: RunReport,
    ):
        detections = prediction.detections

        row = self._prediction_row(base, prediction)
        row["prediction"] = "_".join(detection.prediction for detection in detections) if detections else None
        row["confidence"] = (
            sum(max_confidence(detection.confidence) for detection in detections) / len(detections)
            if detections else 0.0
        )

        if self._dry_run:
            report.rows_main_inserted += 1
            report.detections_seen += len(detections)
            report.rows_object_inserted += len(detections)   # bbox validity needs the image; assume all valid
            logger.debug(f"would insert into {main_collection}: {self._describe(row)} "
                         f"and {len(detections)} rows into {object_collection}")
            return

        image = self._download(prediction.data_path, report)
        if image is None:
            return

        vector = self._extract(image, row["filename"], None, report, "prediction")
        if vector is None:
            return

        row["feature_vector"] = vector
        if not self._insert(main_collection, [row], report, "main"):
            return

        if not detections:
            return

        try:
            width, height = image_size(image)
        except Exception as error:
            logger.error(f"{prediction.data_path}: cannot read image size, skipping detections: {error}")
            report.detections_seen += len(detections)
            report.detections_bad_bbox += len(detections)
            return

        stem = row["filename"].rsplit(".", 1)[0]
        extension = FORMAT_EXTENSIONS.get(self._crop_format, self._crop_format.lower())
        content_type = FORMAT_CONTENT_TYPES.get(self._crop_format)

        object_rows = []
        for index, detection in enumerate(detections):
            report.detections_seen += 1

            box = normalize_bbox(detection.bbox, width, height)
            if box is None:
                report.detections_bad_bbox += 1
                logger.debug(f"{prediction.data_path}: detection {index} has invalid bbox {detection.bbox}")
                continue

            crop_name = f"{stem}_bbox{index}.{extension}"
            crop_vector = self._extract(crop(image, box, self._crop_format), crop_name, content_type,
                                        report, "detection")
            if crop_vector is None:
                continue

            object_row = self._prediction_row(base, prediction)
            object_row["prediction"] = detection.prediction
            object_row["confidence"] = max_confidence(detection.confidence)
            object_row["bbox"] = [float(value) for value in detection.bbox]
            object_row["feature_vector"] = crop_vector
            object_rows.append(object_row)

        self._insert(object_collection, object_rows, report, "object")

    # ------------------------------------------------------------------ rows
    @staticmethod
    def _base_row(inspection: InspectionDocumentV1, ai_result: AIResultData) -> dict:
        metadata = inspection.metadata
        created_at: datetime = metadata.created_at
        return {
            "gbm": metadata.gbm,
            "process": metadata.process,
            "location": metadata.location,
            "equipment_id": metadata.equipment_id,
            "product_id": metadata.product_id,
            "created_at": created_at.isoformat(),
            "task": ai_result.task.value,
        }

    @staticmethod
    def _prediction_row(base: dict, prediction) -> dict:
        return {
            **base,
            "filename": filename_of(prediction.data_path),
            "data_uri": prediction.data_path,
            "elapsed_time": float(prediction.elapsed_time),
        }

    @staticmethod
    def _describe(row: dict) -> str:
        return (f"{row['data_uri']} prediction={row['prediction']!r} "
                f"confidence={row['confidence']:.4f} elapsed={row['elapsed_time']}")

    # ------------------------------------------------------------------ I/O
    def _download(self, data_path: str, report: RunReport) -> bytes | None:
        try:
            return self._storage.get_bytes(data_path)
        except ObjectStorageError as error:
            report.predictions_download_failed += 1
            logger.error(f"download failed: {error}")
            return None

    def _extract(self, image: bytes, filename: str, content_type: str | None,
                 report: RunReport, kind: str) -> list[float] | None:
        try:
            return self._api.extract(image, filename, content_type)
        except FeatureApiError as error:
            if kind == "prediction":
                report.predictions_api_failed += 1
            else:
                report.detections_api_failed += 1
            logger.error(f"feature API failed: {error}")
            return None

    def _insert(self, collection: str, rows: list[dict], report: RunReport, kind: str) -> bool:
        if not rows:
            return True
        try:
            count = self._store.insert(collection, rows)
        except Exception as error:
            if kind == "main":
                report.rows_main_failed += len(rows)
            else:
                report.rows_object_failed += len(rows)
            logger.error(f"insert into {collection} failed ({len(rows)} rows): {error}")
            return False

        if kind == "main":
            report.rows_main_inserted += count
        else:
            report.rows_object_inserted += count
        logger.info(f"inserted {count} {kind} row(s) into {collection}: {rows[0]['data_uri']}")
        return True
