"""Thin MongoDB connection wrapper.

Two independent servers are involved, so this deliberately is NOT a singleton --
each connection carries its own URI, database and collection.

pymongo (synchronous) is used rather than motor: this is a batch job that walks
one cursor at a time, so an event loop buys nothing and costs complexity.
"""
import logging

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


class MongoConnection:
    def __init__(self, uri: str, db_name: str, collection_name: str, label: str = ""):
        self._uri = uri
        self._db_name = db_name
        self._collection_name = collection_name
        self._label = label or db_name
        self._client = MongoClient(uri, serverSelectionTimeoutMS=10_000, tz_aware=False)

    @property
    def collection(self) -> Collection:
        return self._client[self._db_name][self._collection_name]

    @property
    def label(self) -> str:
        return self._label

    @property
    def target(self) -> str:
        return f"{self._db_name}.{self._collection_name}"

    def ping(self) -> bool:
        try:
            self._client.admin.command("ping")
            logger.info(f"[{self._label}] connected to {self.target}")
            return True
        except PyMongoError as error:
            logger.error(f"[{self._label}] connection failed: {error}")
            return False

    def close(self):
        self._client.close()

    def __enter__(self) -> "MongoConnection":
        return self

    def __exit__(self, *_):
        self.close()
