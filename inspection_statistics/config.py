"""Connection constants and statistics tunables.

One MongoDB server holds both collections:

  INSPECTIONS_COLLECTION  -> read-only source of inspection documents
  STATISTICS_COLLECTION   -> where the inspectionStatistics documents are written

Leave USERNAME/PASSWORD empty if the server does not require authentication,
and REPLICA_SET empty if it is a standalone instance.

This file is independent of the other packages' config.py on purpose.
"""
from urllib.parse import quote_plus

# --------------------------------------------------------------------------- #
# MongoDB -- one server, two collections
# --------------------------------------------------------------------------- #
MONGO_HOST = ""
MONGO_PORT = 27017
MONGO_USERNAME = ""
MONGO_PASSWORD = ""
MONGO_AUTH_SOURCE = "admin"
MONGO_REPLICA_SET = ""
DB_NAME = ""
INSPECTIONS_COLLECTION = "inspections"
STATISTICS_COLLECTION = "inspectionStatistics"

# --------------------------------------------------------------------------- #
# Bin definitions. Stored in every document under ``bins`` so that a reader
# never has to guess them. Changing them changes the meaning of every
# histogram, so regenerate the collection if you do.
# --------------------------------------------------------------------------- #
CONFIDENCE_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]   # 10 bins, last one closed
NEAR_THRESHOLD_MARGIN = 0.05                                                 # |confidence - threshold| < margin
BOXES_PER_IMAGE_MAX = 5                                                      # bins 0,1,2,3,4,5+

# --------------------------------------------------------------------------- #
# Behaviour defaults
# --------------------------------------------------------------------------- #
READ_BATCH_SIZE = 500
ROUND_DIGITS = 6                       # every stored float is rounded to this many decimals
RETENTION_SECONDS = 365 * 24 * 3600    # TTL index on startDate: about one year


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


def mongo_uri() -> str:
    return build_uri(
        MONGO_HOST, MONGO_PORT, MONGO_USERNAME, MONGO_PASSWORD, MONGO_AUTH_SOURCE, MONGO_REPLICA_SET,
    )


def validate() -> list[str]:
    """Return a list of unfilled constants, so the CLI can fail early and clearly."""
    required = {
        "MONGO_HOST": MONGO_HOST,
        "DB_NAME": DB_NAME,
        "INSPECTIONS_COLLECTION": INSPECTIONS_COLLECTION,
        "STATISTICS_COLLECTION": STATISTICS_COLLECTION,
    }
    return [name for name, value in required.items() if not value]
