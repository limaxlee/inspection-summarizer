"""Counters for one run. Every skip has its own line."""
from dataclasses import dataclass, field


@dataclass
class RunReport:
    granularities_processed: int = 0
    periods_processed: int = 0
    periods_empty: int = 0

    inspections_read: int = 0
    inspections_invalid: int = 0
    inspections_no_created_at: int = 0

    ai_results_seen: int = 0
    ai_results_filtered_out: int = 0
    ai_results_seg_skipped: int = 0

    predictions_counted: int = 0
    boxes_counted: int = 0
    confidence_missing: int = 0
    confidence_parse_errors: int = 0

    documents_created: int = 0
    documents_updated: int = 0
    documents_written: int = 0
    documents_failed: int = 0

    snapped_windows: dict[str, str] = field(default_factory=dict)
    metadata_conflicts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def render(self, dry_run: bool) -> str:
        mode = "DRY RUN -- nothing was written" if dry_run else "COMPLETED"
        written = "would write" if dry_run else "written"
        lines = [
            f"Run {mode}",
            f"  granularities processed   : {self.granularities_processed}",
            f"  periods processed         : {self.periods_processed}",
            f"    empty (no document)     : {self.periods_empty}",
            f"  inspections read          : {self.inspections_read}",
            f"    invalid (skipped)       : {self.inspections_invalid}",
            f"    no createdAt (skipped)  : {self.inspections_no_created_at}",
            f"  aiResults seen            : {self.ai_results_seen}",
            f"    filtered out            : {self.ai_results_filtered_out}",
            f"    seg (skipped)           : {self.ai_results_seg_skipped}",
            f"  predictions counted       : {self.predictions_counted}",
            f"    boxes (det)             : {self.boxes_counted}",
            f"    missing confidence      : {self.confidence_missing}",
            f"    parse errors            : {self.confidence_parse_errors}",
            f"  documents created         : {self.documents_created}",
            f"  documents updated         : {self.documents_updated}",
            f"  documents {written:<15} : {self.documents_written}",
            f"    failed                  : {self.documents_failed}",
        ]
        if self.snapped_windows:
            lines.append("  windows actually read (snapped to full periods):")
            lines.extend(f"      {name:<7}: {window}" for name, window in self.snapped_windows.items())
        if self.metadata_conflicts:
            lines.append(f"  metadata conflicts        : {len(self.metadata_conflicts)} (first-writer-wins)")
            lines.extend(f"      {conflict}" for conflict in self.metadata_conflicts[:10])
            if len(self.metadata_conflicts) > 10:
                lines.append(f"      ... and {len(self.metadata_conflicts) - 10} more")
        return "\n".join(lines)
