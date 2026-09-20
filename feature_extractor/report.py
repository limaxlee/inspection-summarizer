"""Counters for one run. Every skip and failure has its own line."""
from dataclasses import dataclass


@dataclass
class RunReport:
    days_processed: int = 0
    inspections_read: int = 0
    inspections_invalid: int = 0
    inspections_no_created_at: int = 0
    inspections_no_id: int = 0

    ai_results_seen: int = 0
    ai_results_filtered_out: int = 0
    ai_results_seg_skipped: int = 0
    ai_results_errors: int = 0

    collections_created: int = 0

    predictions_seen: int = 0
    predictions_no_data_path: int = 0
    predictions_already_present: int = 0
    predictions_download_failed: int = 0
    predictions_api_failed: int = 0
    rows_main_inserted: int = 0
    rows_main_failed: int = 0

    detections_seen: int = 0
    detections_bad_bbox: int = 0
    detections_api_failed: int = 0
    rows_object_inserted: int = 0
    rows_object_failed: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self, dry_run: bool) -> str:
        mode = "DRY RUN -- nothing was downloaded, extracted or written" if dry_run else "COMPLETED"
        inserted = "would insert" if dry_run else "inserted"
        return "\n".join([
            f"Run {mode}",
            f"  days processed            : {self.days_processed}",
            f"  inspections read          : {self.inspections_read}",
            f"    invalid (skipped)       : {self.inspections_invalid}",
            f"    no createdAt (skipped)  : {self.inspections_no_created_at}",
            f"    no _id (skipped)        : {self.inspections_no_id}",
            f"  aiResults seen            : {self.ai_results_seen}",
            f"    filtered out            : {self.ai_results_filtered_out}",
            f"    seg (skipped)           : {self.ai_results_seg_skipped}",
            f"    errors                  : {self.ai_results_errors}",
            f"  collections created       : {self.collections_created}",
            f"  predictions seen          : {self.predictions_seen}",
            f"    no dataPath (skipped)   : {self.predictions_no_data_path}",
            f"    already present         : {self.predictions_already_present}",
            f"    download failed         : {self.predictions_download_failed}",
            f"    API failed              : {self.predictions_api_failed}",
            f"  main rows {inserted:<15} : {self.rows_main_inserted}",
            f"    insert failed           : {self.rows_main_failed}",
            f"  detections seen           : {self.detections_seen}",
            f"    bad bbox (skipped)      : {self.detections_bad_bbox}",
            f"    API failed              : {self.detections_api_failed}",
            f"  object rows {inserted:<13} : {self.rows_object_inserted}",
            f"    insert failed           : {self.rows_object_failed}",
        ])
