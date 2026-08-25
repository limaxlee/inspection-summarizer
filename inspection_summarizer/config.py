"""Connection constants.

Fill in the values below for your two servers.

  SOURCE_*  -> server A, where the inspection documents are read from
  TARGET_*  -> server B, where the inspection summary documents are written to

Leave USERNAME/PASSWORD empty if the server does not require authentication,
and REPLICA_SET empty if it is a standalone instance.
"""
from urllib.parse import quote_plus

# --------------------------------------------------------------------------- #
# Server A -- source (inspection documents, read-only)
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
# Server B -- target (inspection summary documents, read + write)
# --------------------------------------------------------------------------- #
TARGET_MONGO_HOST = ""
TARGET_MONGO_PORT = 27017
TARGET_MONGO_USERNAME = ""
TARGET_MONGO_PASSWORD = ""
TARGET_MONGO_AUTH_SOURCE = "admin"
TARGET_MONGO_REPLICA_SET = ""
TARGET_DB_NAME = ""
TARGET_COLLECTION = ""

# --------------------------------------------------------------------------- #
# Behaviour defaults (all overridable from the CLI)
# --------------------------------------------------------------------------- #
MAX_SAMPLES_PER_CLASS = 3
READ_BATCH_SIZE = 500

# Warn when a bucket's inspectionIds array approaches the 16MB BSON limit.
# ~31 bytes per stored ObjectId string, so this trips well before the ceiling.
INSPECTION_IDS_WARN_THRESHOLD = 300_000


def build_uri(
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
    return build_uri(
        SOURCE_MONGO_HOST, SOURCE_MONGO_PORT, SOURCE_MONGO_USERNAME,
        SOURCE_MONGO_PASSWORD, SOURCE_MONGO_AUTH_SOURCE, SOURCE_MONGO_REPLICA_SET,
    )


def target_uri() -> str:
    return build_uri(
        TARGET_MONGO_HOST, TARGET_MONGO_PORT, TARGET_MONGO_USERNAME,
        TARGET_MONGO_PASSWORD, TARGET_MONGO_AUTH_SOURCE, TARGET_MONGO_REPLICA_SET,
    )


def validate() -> list[str]:
    """Return a list of unfilled constants, so the CLI can fail early and clearly."""
    required = {
        "SOURCE_MONGO_HOST": SOURCE_MONGO_HOST,
        "SOURCE_DB_NAME": SOURCE_DB_NAME,
        "SOURCE_COLLECTION": SOURCE_COLLECTION,
        "TARGET_MONGO_HOST": TARGET_MONGO_HOST,
        "TARGET_DB_NAME": TARGET_DB_NAME,
        "TARGET_COLLECTION": TARGET_COLLECTION,
    }
    return [name for name, value in required.items() if not value]
