"""Read side: inspection documents from MongoDB. Never writes."""
import logging
from typing import Iterator

from pymongo import ASCENDING

from feature_extractor.db import MongoConnection

logger = logging.getLogger(__name__)

CREATED_AT = "metadata.createdAt"


class InspectionRepository:
    def __init__(self, connection: MongoConnection):
        self._connection = connection

    def count(self, query: dict) -> int:
        return self._connection.collection.count_documents(query)

    def iter_documents(self, query: dict, batch_size: int, limit: int | None = None) -> Iterator[dict]:
        """Yield raw documents oldest-first, so a run is deterministic."""
        cursor = self._connection.collection.find(query).sort(CREATED_AT, ASCENDING).batch_size(batch_size)
        if limit:
            cursor = cursor.limit(limit)

        try:
            yield from cursor
        finally:
            cursor.close()

    def find_one(self, query: dict | None = None) -> dict | None:
        return self._connection.collection.find_one(query or {})
