# feature_extractor

Reads inspection documents from MongoDB, downloads the image behind every
prediction from object storage, sends it to the feature extraction API, and
stores the returned feature vector in Milvus, one collection per process and model.

Fully self-contained: it shares no code with `inspection_summarizer`, only the same
source collection. Full documentation: [DOCUMENTATION.md](DOCUMENTATION.md).

## Quick start

```bash
pip install -r feature_extractor/requirements.txt
```

Fill in `feature_extractor/config.py`: the `SOURCE_*` MongoDB constants and
`EXTRACTOR_HOST`. Object storage and Milvus values are already there. This is the
only file you edit.

```bash
python extract.py verify        # pings everything, pushes one real image through the chain
python extract.py run --start 2026-06-20 --end 2026-06-21 --gbm SEHC --process Side --dry-run
python extract.py run --start 2026-06-20 --end 2026-06-21 --gbm SEHC --process Side
```

The filters are identical to the summarizer's `run` command. Extractor-specific
flags: `--force` inserts even when a row with the same `data_uri` exists, and
`--dry-run` reads and counts without downloading, calling the API, or writing.

## In one table

| task | `VFM_{process}_{modelName}_{modelVersion}` | `..._object` |
|---|---|---|
| seg | skipped | - |
| cls | one row per prediction | - |
| det | one row per prediction; labels joined with `_`, mean confidence | one row per valid detection, image cropped to its bbox |
