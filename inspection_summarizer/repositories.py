"""Read side (server A) and write side (server B)."""
import logging
from typing import Iterator

from pymongo import ASCENDING
from pymongo.errors import PyMongoError

from inspection_summarizer.db import MongoConnection
from inspection_summarizer.keys import SummaryKey, KEY_FIELDS
from inspection_summarizer.schemas.base import to_mongo
from inspection_summarizer.schemas.summary import InspectionSummaryDocumentV1

logger = logging.getLogger(__name__)

CREATED_AT = "metadata.createdAt"


class InspectionRepository:
    """Reads inspection documents from server A. Never writes."""

    def __init__(self, connection: MongoConnection):
        self._connection = connection

    def count(self, query: dict) -> int:
        return self._connection.collection.count_documents(query)

    def iter_documents(self, query: dict, batch_size: int, limit: int | None = None) -> Iterator[dict]:
        """Yield raw documents oldest-first.

        Sorting by createdAt makes a run deterministic: merge order does not
        affect the statistics, but it does affect which samples survive eviction.
        """
        cursor = self._connection.collection.find(query).sort(CREATED_AT, ASCENDING).batch_size(batch_size)
        if limit:
            cursor = cursor.limit(limit)

        try:
            yield from cursor
        finally:
            cursor.close()

    def find_one(self, query: dict | None = None) -> dict | None:
        return self._connection.collection.find_one(query or {})


class SummaryRepository:
    """Reads and upserts inspection summary documents on server B."""

    def __init__(self, connection: MongoConnection):
        self._connection = connection

    def ensure_index(self):
        """One summary document per key -- enforce it at the database level."""
        try:
            name = self._connection.collection.create_index(
                [(field, ASCENDING) for field in KEY_FIELDS],
                unique=True,
                name="summary_key_unique",
            )
            logger.info(f"[{self._connection.label}] unique index ready: {name}")
        except PyMongoError as error:
            logger.warning(f"[{self._connection.label}] could not create the unique index: {error}")

    def find_by_key(self, key: SummaryKey) -> InspectionSummaryDocumentV1 | None:
        raw = self._connection.collection.find_one(key.to_query())
        if raw is None:
            return None

        # InspectionSummaryDocumentV1 has no _id field.
        raw.pop("_id", None)
        return InspectionSummaryDocumentV1.model_validate(raw)

    def upsert(self, key: SummaryKey, document: InspectionSummaryDocumentV1) -> bool:
        payload = to_mongo(document.model_dump(by_alias=True))
        result = self._connection.collection.replace_one(key.to_query(), payload, upsert=True)

        if not result.acknowledged:
            logger.error(f"Failed to upsert summary {key}")
            return False

        return True
