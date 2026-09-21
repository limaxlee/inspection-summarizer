# inspection_statistics

Reads inspection documents from MongoDB and writes **pre-aggregated statistics** of the
AI results into the `inspectionStatistics` collection on the same server: one document
per model version, task, mode, site and period (`hourly`, `shift`, `daily`, `weekly`),
with fixed-bin confidence histograms, quantiles, class counts, threshold relation and
elapsed-time stats, per equipment and in total.

Fully self-contained: it shares no code with `inspection_summarizer` or `feature_extractor`,
only the same source collection. Full documentation: [DOCUMENTATION.md](DOCUMENTATION.md).

## Quick start

```bash
pip install -r inspection_statistics/requirements.txt
```

Fill in `inspection_statistics/config.py` (`MONGO_*`, `DB_NAME`; the two collection names
are preset). This is the only file you edit.

```bash
python stats.py verify                                                            # read-only
python stats.py run --start 2026-09-21 --end 2026-09-22 --gbm SEHC --process Side --dry-run
python stats.py run --start 2026-09-21 --end 2026-09-22 --gbm SEHC --process Side
python stats.py run --start 2026-09-21T04:30 --end 2026-09-21T06:10 --granularity hourly shift
```

The filters are identical to the summarizer's `run` command. Statistics-specific:
`--granularity` (default: all four). The window is **snapped outward to complete
periods** per granularity, because every document is recomputed from scratch and then
replaces the stored one.

## In one table

| | cls | det |
|---|---|---|
| `predictionCount` | images | images |
| `boxCount` | - | boxes |
| `confidence`, `classCounts`, `perClass` | over predictions | over boxes |
| `threshold` | `perClass[c].threshold`, per predicted class (last seen) | `perClass[c].threshold`, per box class (last seen) |
| `images` | - | no-box count, boxes per image histogram, boxes per image by class |
| seg | no document | |

## Tests

```bash
python -m pytest inspection_statistics/tests -q
```

Synthetic data only; MongoDB is not contacted.
