# inspection-summarizer

Reads inspection documents from MongoDB on **server A**, folds them into inspection
summary documents, and creates or updates those summaries on MongoDB on **server B**.

## Setup

```bash
pip install -r requirements.txt
```

Fill in the connection constants in `inspection_summarizer/config.py` (`SOURCE_*`
for server A, `TARGET_*` for server B). Nothing else is read from the environment.

## Usage

Check both connections and see how the documents are actually stored — read-only,
writes nothing:

```bash
python main.py verify
```

Build summaries. Every filter is explicit; an omitted flag adds no constraint, and
a repeated value becomes an `$in`:

```bash
python main.py run --start 2026-06-20 --end 2026-06-21 --gbm SEHC --process Side --dry-run
```

```bash
python main.py run \
    --start 2026-06-20 --end 2026-06-23 \
    --gbm SEHC --process Side \
    --mode production \
    --equipment-id SEHC_Side_VM07 SEHC_Side_VM08 \
    --model-name sidebottom \
    --report-json run-report.json
```

`--start`/`--end` are a half-open UTC window on `metadata.createdAt`. `--dry-run`
computes everything and writes nothing. Without `--yes` you get the resolved query,
the matched document count, and a confirmation prompt before any write.

Run `python main.py run --help` for the full flag list.

## How the conversion works

One summary document exists per:

```
modelName + modelVersion + gbm + process + mode + date
```

where `date` is the UTC day start of the source inspection's `metadata.createdAt`.

A single inspection **fans out to one summary per entry in
`inspectionResults.aiResults`** — an inspection that ran two models contributes to
two different summaries. Conversely, every inspection sharing a key folds into the
same summary, accumulating `statistics.dataCount`, `statistics.confidence` and
`statistics.elapsedTime`.

Merging is **idempotent**: each summary records the ids it has already absorbed in
`inspectionIds`, so re-running an overlapping window changes nothing. That makes the
same command usable both for a historical backfill and for a scheduled incremental
job.

The run walks the window one UTC day at a time. Since `date` is part of the key, a
day boundary partitions the key space cleanly, so each summary is read once and
written once per run rather than once per inspection.

### Things worth knowing

- **`location`, `equipmentId`, `productId` and `localTimezone` are not part of the
  key.** They are copied from whichever inspection creates the summary and are never
  updated. When a later inspection disagrees, the run logs a warning and reports the
  conflict, but the stored value stands.
- **Samples require `fileIndex`.** A prediction whose `fileIndex` is null, or which
  points past the end of `fileList`, still counts toward every statistic but cannot
  become a sample. Where inspections carry neither, `samples` stays `{}` and the
  summary is purely statistical. The run reports how many predictions this affected.
- **For detection models**, `totalCount` counts predictions (images) while the
  per-class counts count bounding boxes, so they do not sum to each other.
- **Segmentation** produces no per-class statistics; its samples land under
  `unknown`.
- `inspectionIds` grows with every absorbed inspection. The run warns when a summary
  approaches the 16MB BSON limit (around 500k ids).

## Layout

```
inspection_summarizer/
  common/constants.py     ModelTasks, InspectionMode, UploadStatus
  common/datetime_util.py day truncation and day-window iteration
  schemas/base.py         DataModel (snake_case fields <-> camelCase storage)
  schemas/inspections.py  InspectionDocumentV1 and its tree
  schemas/summary.py      InspectionSummaryDocumentV1 and its tree
  summarizer.py           the conversion itself
  config.py               connection constants
  db.py                   per-server connection
  filters.py              InspectionFilter -> MongoDB query
  keys.py                 the summary key
  repositories.py         read from A, upsert into B
  pipeline.py             day loop, bucket folding, run report
  cli.py                  argument parsing
```

`summarizer.py` is a verbatim port of the reference implementation. Its behaviour is
a contract with anything else reading the summary collection — changing it means
regenerating the affected summaries.

## Other tools in this repository

Each package is self-contained and only shares the source `inspections` collection.

| Package | Entry point | Output |
|---|---|---|
| `inspection_summarizer/` | `python main.py` | per-day summary documents with samples (this README) |
| `feature_extractor/` | `python extract.py` | feature vectors in Milvus, see [feature_extractor/README.md](feature_extractor/README.md) |
| `inspection_statistics/` | `python stats.py` | per-period `inspectionStatistics` documents (histograms, quantiles, class counts), see [inspection_statistics/README.md](inspection_statistics/README.md) |
