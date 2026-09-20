"""Connection constants and tunables for the feature extractor.

Four external systems are involved:

  SOURCE_*          -> MongoDB, where the inspection documents are read from
  OBJECT_STORAGE_*  -> S3-compatible object storage, where the image data lives
  EXTRACTOR_*       -> the feature extraction HTTP API
  MILVUS_*          -> Milvus, where the feature vectors are written to

This file is independent of inspection_summarizer/config.py on purpose.
"""
from urllib.parse import quote_plus

# --------------------------------------------------------------------------- #
# MongoDB -- source (inspection documents, read-only)
# --------------------------------------------------------------------------- #
SOURCE_MONGO_HOST = ""
SOURCE_MONGO_PORT = 27017
SOURCE_MONGO_USERNAME = ""
SOURCE_MONGO_PASSWORD = ""
SOURCE_MONGO_AUTH_SOURCE = "admin"
SOURCE_MONGO_REPLICA_SET = ""
SOURCE_DB_NAME = ""
SOURCE_COLLECTION = ""

# --------------------------------------------------------------------------- #
# Object storage (S3-compatible) -- where prediction.dataPath objects live
# --------------------------------------------------------------------------- #
OBJECT_STORAGE_ENDPOINT = "https://gage.ap-northeast-2.samsungspc.com"
OBJECT_STORAGE_BUCKET = "cosmo-storage-stage"
OBJECT_STORAGE_ACCESS_KEY = "AKIA5PMO9S8UVFAFT68P"
OBJECT_STORAGE_SECRET_KEY = "gCBynEY51EVVuV901dEybGl8JnwoPkQy9YK7AbjW"

# --------------------------------------------------------------------------- #
# Feature extraction API -- POST http://HOST:PORT/features
# --------------------------------------------------------------------------- #
EXTRACTOR_HOST = ""            # <-- fill in
EXTRACTOR_PORT = 24500
EXTRACTOR_PATH = "/features"
EXTRACTOR_TIMEOUT_SECONDS = 120

# --------------------------------------------------------------------------- #
# Milvus -- target (feature vectors, read + write)
# --------------------------------------------------------------------------- #
MILVUS_HOST = "10.240.35.180"
MILVUS_PORT = 19530
MILVUS_USER = ""
MILVUS_PASSWORD = ""
MILVUS_DB_NAME = "cosmoData"

# --------------------------------------------------------------------------- #
# Feature / collection layout
# --------------------------------------------------------------------------- #
FEATURE_DIM = 2048                     # length of feature_vector; must match the API output
MILVUS_METRIC_TYPE = "COSINE"          # metric of the vector index created on new collections
MILVUS_INDEX_TYPE = "AUTOINDEX"
COLLECTION_PREFIX = "VFM"              # VFM_{process}_{modelName}_{modelVersion}
OBJECT_COLLECTION_SUFFIX = "_object"   # ... plus this for per-detection rows of det models
VARCHAR_MAX_LENGTH = 512               # max_length of every VARCHAR field
CROP_IMAGE_FORMAT = "PNG"              # encoding of bounding-box crops sent to the API

# --------------------------------------------------------------------------- #
# Behaviour defaults (overridable from the CLI)
# --------------------------------------------------------------------------- #
READ_BATCH_SIZE = 500


def build_mongo_uri(
        host: str,
        port: int,
        username: str = "",
        password: str = "",
        auth_source: str = "admin",
        replica_set: str = "",
) -> str:
    credentials = f"{quote_plus(username)}:{quote_plus(password)}@" if username else ""
    uri = f"mongodb://{credentials}{host}:{port}"

    options = []
    if username:
        options.append(f"authSource={auth_source}")
    if replica_set:
        options.extend([f"replicaSet={replica_set}", "retryWrites=true", "w=majority"])

    return f"{uri}/?{'&'.join(options)}" if options else uri


def source_uri() -> str:
    return build_mongo_uri(
        SOURCE_MONGO_HOST, SOURCE_MONGO_PORT, SOURCE_MONGO_USERNAME,
        SOURCE_MONGO_PASSWORD, SOURCE_MONGO_AUTH_SOURCE, SOURCE_MONGO_REPLICA_SET,
    )


def extractor_url() -> str:
    return f"http://{EXTRACTOR_HOST}:{EXTRACTOR_PORT}{EXTRACTOR_PATH}"


def milvus_uri() -> str:
    return f"http://{MILVUS_HOST}:{MILVUS_PORT}"


def validate() -> list[str]:
    """Return a list of unfilled constants, so the CLI can fail early and clearly."""
    required = {
        "SOURCE_MONGO_HOST": SOURCE_MONGO_HOST,
        "SOURCE_DB_NAME": SOURCE_DB_NAME,
        "SOURCE_COLLECTION": SOURCE_COLLECTION,
        "OBJECT_STORAGE_ENDPOINT": OBJECT_STORAGE_ENDPOINT,
        "OBJECT_STORAGE_BUCKET": OBJECT_STORAGE_BUCKET,
        "OBJECT_STORAGE_ACCESS_KEY": OBJECT_STORAGE_ACCESS_KEY,
        "OBJECT_STORAGE_SECRET_KEY": OBJECT_STORAGE_SECRET_KEY,
        "EXTRACTOR_HOST": EXTRACTOR_HOST,
        "MILVUS_HOST": MILVUS_HOST,
        "MILVUS_DB_NAME": MILVUS_DB_NAME,
    }
    return [name for name, value in required.items() if not value]
