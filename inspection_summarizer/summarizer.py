"""Verbatim port of the reference InspectionSummarizer.

The only changes from the original are the import paths. The behaviour of this
class is a contract with anything else that reads the summary collection, so do
not "fix" anything in here without regenerating the affected summaries.
"""
import logging
from typing import Any

from inspection_summarizer.common.constants import ModelTasks
from inspection_summarizer.common.datetime_util import truncate_to_day_start
from inspection_summarizer.schemas.inspections import (
    InspectionDocumentV1, ClassificationData, ObjectDetectionData, SegmentationData,
    ClassificationResult, ObjectDetectionResult, SegmentationResult
)
from inspection_summarizer.schemas.summary import (
    BaseSummaryInfo, InspectionSummaryDocumentV1, Statistics, BaseDataSampleInfo, DataSamples, ValueStats,
    ClassificationSample, DetectionSample, SegmentationSample,
)

logger = logging.getLogger(__name__)

AIResultData = ClassificationData | ObjectDetectionData | SegmentationData


class InspectionSummarizer:
    def __init__(self, max_samples_per_class: int = 3):
        self._max_samples_per_class = max_samples_per_class
        self._default_class = "unknown"

    def get_summary_base(self, inspection_document: InspectionDocumentV1) -> list[tuple[BaseSummaryInfo, AIResultData]]:
        date = truncate_to_day_start(inspection_document.metadata.created_at.replace(tzinfo=None))

        summary_base = []
        for ai_result in inspection_document.inspection_result.ai_results:
            model_name, _, model_version = ai_result.ai_model.rpartition("/")
            summary_base.append((
                BaseSummaryInfo(
                    model_name=model_name,
                    model_version=model_version,
                    gbm=inspection_document.metadata.gbm,
                    process=inspection_document.metadata.process,
                    date=date,
                    mode=inspection_document.metadata.mode
                ),
                ai_result
            ))

        return summary_base

    def register(
            self,
            summary_base: BaseSummaryInfo,
            summary_document: InspectionSummaryDocumentV1 | None,
            inspection_document: InspectionDocumentV1,
            ai_result: AIResultData
    ) -> InspectionSummaryDocumentV1 | None:
        if not inspection_document.inspection_id:
            logger.warning(f"Skipping summary of an inspection without an id: {ai_result.ai_model}")
            return None

        inspection_id = str(inspection_document.inspection_id)

        if summary_document is None:
            summary_document = self._create_summary_document(summary_base, inspection_document, ai_result)
        elif inspection_id in summary_document.inspection_ids:
            logger.info(f"Inspection {inspection_id} is already merged into the summary of {ai_result.ai_model}")
            return None

        summary_document.inspection_ids.append(inspection_id)
        data_samples = self._get_data_samples(summary_document, inspection_document, ai_result)
        self._update_avg_stats(summary_document)
        self._update_data_samples(summary_document, data_samples)

        return summary_document

    def _create_summary_document(
            self,
            summary_base: BaseSummaryInfo,
            inspection_document: InspectionDocumentV1,
            ai_result: AIResultData
    ) -> InspectionSummaryDocumentV1:
        return InspectionSummaryDocumentV1(
            **summary_base.model_dump(),
            location=inspection_document.metadata.location,
            equipment_id=inspection_document.metadata.equipment_id,
            product_id=inspection_document.metadata.product_id,
            local_timezone=inspection_document.metadata.local_timezone,
            task=ai_result.task,
            classes=ai_result.classes,
            threshold=self._get_threshold(ai_result),
            statistics=Statistics(),
            samples={}
        )

    def _get_data_samples(
            self,
            summary_document: InspectionSummaryDocumentV1,
            inspection_document: InspectionDocumentV1,
            ai_result: AIResultData
    ) -> dict[str, list[DataSamples]]:
        summary_stats = summary_document.statistics
        data_count = summary_stats.data_count
        data_samples = {}

        for class_name in ai_result.classes:
            if class_name not in summary_document.classes:
                summary_document.classes.append(class_name)

        for prediction in ai_result.predictions:
            data_count["totalCount"] = data_count.get("totalCount", 0) + 1
            self._calculate_avg_min_max(summary_stats.elapsed_time, prediction.elapsed_time, data_count["totalCount"])

            if ai_result.task == ModelTasks.CLASSIFICATION:
                self._get_cls_data_samples(summary_document, inspection_document, prediction, data_samples)
            elif ai_result.task == ModelTasks.DETECTION:
                self._get_det_data_samples(summary_document, inspection_document, prediction, data_samples)
            elif ai_result.task == ModelTasks.SEGMENTATION:
                self._get_seg_data_samples(inspection_document, prediction, data_samples)
            else:
                logger.warning(f"Not supported task: {ai_result.task}")

        return data_samples

    def _get_cls_data_samples(
            self,
            summary_document: InspectionSummaryDocumentV1,
            inspection_document: InspectionDocumentV1,
            inference_result: ClassificationResult,
            data_samples: dict
    ):
        class_name = inference_result.prediction
        confidence_score = self._get_confidence_score(inference_result.confidence)
        self._calculate_class_stats(summary_document, class_name, confidence_score)

        if not self._has_valid_file_index(inference_result, inspection_document):
            return

        data_samples.setdefault(class_name, []).append(
            ClassificationSample(
                **self._get_base_sample_info(inference_result, inspection_document),
                prediction=inference_result.prediction,
                decision=inference_result.decision,
                confidence=inference_result.confidence,
                feedbacks=inference_result.feedbacks,
                threshold=inference_result.threshold
            )
        )

    def _get_det_data_samples(
            self,
            summary_document: InspectionSummaryDocumentV1,
            inspection_document: InspectionDocumentV1,
            inference_result: ObjectDetectionResult,
            data_samples: dict
    ):
        for detection in inference_result.detections:
            confidence_score = self._get_confidence_score(detection.confidence)
            self._calculate_class_stats(summary_document, detection.prediction, confidence_score)

        if not self._has_valid_file_index(inference_result, inspection_document):
            return

        sample = DetectionSample(
            **self._get_base_sample_info(inference_result, inspection_document),
            detections=inference_result.detections
        )
        for class_name in {detection.prediction for detection in inference_result.detections}:
            data_samples.setdefault(class_name, []).append(sample)

    def _get_seg_data_samples(
            self,
            inspection_document: InspectionDocumentV1,
            inference_result: SegmentationResult,
            data_samples: dict
    ):
        if not self._has_valid_file_index(inference_result, inspection_document):
            return

        data_samples.setdefault(self._default_class, []).append(
            SegmentationSample(
                **self._get_base_sample_info(inference_result, inspection_document),
                mask=inference_result.mask,
                mask_shape=inference_result.mask_shape
            )
        )

    def _calculate_class_stats(
            self,
            summary_document: InspectionSummaryDocumentV1,
            class_name: str,
            confidence_score: float
    ):
        summary_stats = summary_document.statistics
        summary_stats.data_count[class_name] = summary_stats.data_count.get(class_name, 0) + 1

        if class_name not in summary_stats.confidence:
            summary_stats.confidence[class_name] = ValueStats()

        self._calculate_avg_min_max(
            summary_stats.confidence[class_name],
            confidence_score,
            summary_stats.data_count[class_name]
        )

    def _calculate_avg_min_max(self, stats: ValueStats, value: float, count: int):
        stats.sum += value
        stats.min = value if count <= 1 else min(stats.min, value)
        stats.max = value if count <= 1 else max(stats.max, value)

    def _update_avg_stats(self, summary_document: InspectionSummaryDocumentV1):
        result_summary = summary_document.statistics
        total_count = result_summary.data_count.get("totalCount", 0)

        if total_count:
            result_summary.elapsed_time.avg = round(result_summary.elapsed_time.sum / total_count, 10)

        for class_name, stats in result_summary.confidence.items():
            class_count = result_summary.data_count.get(class_name, 0)
            if class_count:
                stats.avg = round(stats.sum / class_count, 10)

    def _update_data_samples(
            self,
            summary_document: InspectionSummaryDocumentV1,
            data_samples: dict[str, list[DataSamples]]
    ):
        for class_name, new_samples in data_samples.items():
            data_pool = summary_document.samples.get(class_name, []) + new_samples
            stats = summary_document.statistics.confidence.get(class_name)
            summary_document.samples[class_name] = self._select_data_samples(
                data_pool,
                class_name,
                stats.avg if stats else None
            )

    def _select_data_samples(
            self,
            data_pool: list[DataSamples],
            class_name: str,
            avg_stat: float | None
    ) -> list[DataSamples]:
        unique_samples = {}
        for sample in data_pool:
            unique_samples.setdefault(
                (sample.inspection_id, sample.prediction_id, sample.inference_started_at, sample.data_path),
                sample
            )

        if len(unique_samples) <= self._max_samples_per_class:
            return list(unique_samples.values())

        selected_data = {key: sample for key, sample in unique_samples.items() if getattr(sample, "feedbacks", None)}
        if len(selected_data) >= self._max_samples_per_class:
            latest_samples = sorted(selected_data.values(), key=lambda item: item.inference_started_at, reverse=True)
            return latest_samples[:self._max_samples_per_class]

        rest_samples = [sample for key, sample in unique_samples.items() if key not in selected_data]
        for sample in self._rank_data_samples(rest_samples, class_name, avg_stat):
            if len(selected_data) >= self._max_samples_per_class:
                break
            selected_data.setdefault(
                (sample.inspection_id, sample.prediction_id, sample.inference_started_at, sample.data_path),
                sample
            )

        return list(selected_data.values())

    def _rank_data_samples(
            self,
            data_samples: list[DataSamples],
            class_name: str,
            avg_stat: float | None
    ) -> list[DataSamples]:
        if not data_samples:
            return []

        if avg_stat is None:
            return sorted(data_samples, key=lambda item: item.inference_started_at, reverse=True)

        sorted_scores = sorted(data_samples, key=lambda item: self._get_avg_score_by_prediction(item, class_name))
        ranked_samples = [sorted_scores[0], sorted_scores[-1]]
        ranked_samples.extend(
            sorted(sorted_scores, key=lambda item: abs(self._get_avg_score_by_prediction(item, class_name) - avg_stat))
        )

        return ranked_samples

    def _get_avg_score_by_prediction(self, sample: DataSamples, class_name: str) -> float:
        if isinstance(sample, DetectionSample):
            scores = [self._get_confidence_score(detection.confidence)
                      for detection in sample.detections if detection.prediction == class_name]
            return sum(scores) / len(scores) if scores else 0.0

        return self._get_confidence_score(sample.confidence)

    def _get_confidence_score(self, confidence: list[float] | float) -> float:
        if not isinstance(confidence, list):
            return float(confidence)

        return float(max(confidence))

    def _get_threshold(self, ai_result: AIResultData) -> float | None:
        for prediction in ai_result.predictions:
            if getattr(prediction, "threshold", None) is not None:
                return prediction.threshold

            for detection in getattr(prediction, "detections", []):
                if detection.threshold is not None:
                    return detection.threshold

        return None

    def _has_valid_file_index(self, prediction, inspection_document: InspectionDocumentV1) -> bool:
        file_index = prediction.file_index
        if file_index is None:
            return False

        if file_index >= len(inspection_document.file_list):
            return False

        return True

    def _get_base_sample_info(self, prediction, inspection_document: InspectionDocumentV1) -> dict[str, Any]:
        data_path = prediction.data_path or inspection_document.file_list[prediction.file_index]

        return BaseDataSampleInfo(
            inspection_id=str(inspection_document.inspection_id),
            prediction_id=prediction.prediction_id,
            inference_started_at=prediction.inference_started_at,
            is_patch=prediction.is_patch,
            patch_spec=prediction.patch_spec,
            data_path=data_path,
            elapsed_time=prediction.elapsed_time,
            additional_info=prediction.additional_info,
            tags=prediction.tags
        ).model_dump(by_alias=True)
