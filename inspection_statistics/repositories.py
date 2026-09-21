"""Read side (inspections) and write side (inspectionStatistics), both on the same server."""
import logging
from typing import Iterator

from pymongo import ASCENDING
from pymongo.errors import PyMongoError

from inspection_statistics.db import MongoConnection
from inspection_statistics.keys import StatisticsKey, KEY_FIELDS
from inspection_statistics.schemas.base import to_mongo
from inspection_statistics.schemas.statistics import InspectionStatisticsDocument

logger = logging.getLogger(__name__)

CREATED_AT = "metadata.createdAt"


class InspectionRepository:
    """Reads inspection documents. Never writes."""

    def __init__(self, connection: MongoConnection):
        self._connection = connection

    def count(self, query: dict) -> int:
        return self._connection.collection.count_documents(query)

    def iter_documents(self, query: dict, batch_size: int, limit: int | None = None) -> Iterator[dict]:
        """Yield raw documents oldest-first, so "last value seen" fields are deterministic."""
        cursor = self._connection.collection.find(query).sort(CREATED_AT, ASCENDING).batch_size(batch_size)
        if limit:
            cursor = cursor.limit(limit)

        try:
            yield from cursor
        finally:
            cursor.close()

    def find_one(self, query: dict | None = None) -> dict | None:
        return self._connection.collection.find_one(query or {})


class StatisticsRepository:
    """Writes inspectionStatistics documents: one full replacement per key."""

    def __init__(self, connection: MongoConnection):
        self._connection = connection

    def ensure_indexes(self, retention_seconds: int):
        collection = self._connection.collection
        label = self._connection.label
        specs = [
            # identity, one document per key; prefix serves the drift tool's main query
            dict(keys=[(field, ASCENDING) for field in KEY_FIELDS], unique=True, name="statistics_key_unique"),
            # site-level browsing: "which models ran on this line last week"
            dict(keys=[("gbm", ASCENDING), ("process", ASCENDING), ("granularity", ASCENDING),
                       ("startDate", ASCENDING)], name="site_granularity_startDate"),
            # equipment lookup across models (multikey on the array)
            dict(keys=[("equipments.equipmentId", ASCENDING), ("granularity", ASCENDING),
                       ("startDate", ASCENDING)], name="equipment_granularity_startDate"),
            # retention: expire one year after the period start, and a plain time scan
            dict(keys=[("startDate", ASCENDING)], name="startDate_ttl", expireAfterSeconds=retention_seconds),
        ]
        for spec in specs:
            keys = spec.pop("keys")
            try:
                name = collection.create_index(keys, **spec)
                logger.info(f"[{label}] index ready: {name}")
            except PyMongoError as error:
                logger.warning(f"[{label}] could not create index {spec.get('name')}: {error}")

    def index_names(self) -> list[str]:
        try:
            return sorted(self._connection.collection.index_information())
        except PyMongoError as error:
            logger.warning(f"[{self._connection.label}] could not list indexes: {error}")
            return []

    def count(self, query: dict | None = None) -> int:
        return self._connection.collection.count_documents(query or {})

    def exists(self, key: StatisticsKey) -> bool:
        return self._connection.collection.count_documents(key.to_query(), limit=1) > 0

    def upsert(self, key: StatisticsKey, document: InspectionStatisticsDocument) -> str | None:
        """Replace the document for ``key`` wholesale. Returns "created", "updated" or None on failure."""
        payload = to_mongo(document.model_dump(by_alias=True))
        result = self._connection.collection.replace_one(key.to_query(), payload, upsert=True)

        if not result.acknowledged:
            logger.error(f"Failed to upsert statistics {key}")
            return None

        return "created" if result.upserted_id is not None else "updated"
