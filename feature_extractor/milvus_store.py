"""Milvus: one collection per (process, model), rows of feature vectors.

Collection naming:

    VFM_{process}_{modelName}_{modelVersion}            one row per prediction
    VFM_{process}_{modelName}_{modelVersion}_object     det models only: one row per detection

Milvus only allows letters, digits and underscores in a collection name, so any
other character in ``process`` or ``aiModel`` is replaced by an underscore.
"""
import logging
import re

from pymilvus import MilvusClient, DataType
from pymilvus.exceptions import MilvusException

logger = logging.getLogger(__name__)

# Scalar VARCHAR fields shared by both collections, in schema order.
VARCHAR_FIELDS = ["filename", "data_uri", "gbm", "process", "location", "equipment_id", "task", "created_at"]
NULLABLE_VARCHAR_FIELDS = ["prediction", "product_id"]
FLOAT_FIELDS = ["confidence", "elapsed_time"]
VECTOR_FIELD = "feature_vector"
BBOX_FIELD = "bbox"   # object collection only


class MilvusStore:
    def __init__(
            self,
            uri: str,
            user: str,
            password: str,
            db_name: str,
            dim: int,
            metric_type: str,
            index_type: str,
            varchar_max_length: int,
            collection_prefix: str,
            object_suffix: str,
    ):
        self._uri = uri
        self._token = f"{user}:{password}" if user else ""
        self._db_name = db_name
        self._dim = dim
        self._metric_type = metric_type
        self._index_type = index_type
        self._varchar_max_length = varchar_max_length
        self._prefix = collection_prefix
        self._object_suffix = object_suffix
        self._client: MilvusClient | None = None
        self._ready: set[str] = set()          # collections known to exist and be loaded
        self._renamed: set[str] = set()        # raw names already warned about

    # ------------------------------------------------------------------ setup
    @property
    def target(self) -> str:
        return f"{self._uri}/{self._db_name}"

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._uri, token=self._token, db_name=self._db_name)
        return self._client

    def ping(self) -> bool:
        try:
            names = self.client.list_collections()
            logger.info(f"[milvus] connected to {self.target}: {len(names)} collections")
            return True
        except MilvusException as error:
            logger.error(f"[milvus] connection to {self.target} failed: {error}")
            return False

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "MilvusStore":
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------ naming
    def collection_name(self, process: str, model_name: str, model_version: str, objects: bool = False) -> str:
        raw = f"{self._prefix}_{process}_{model_name}_{model_version}"
        if objects:
            raw += self._object_suffix
        return self._sanitize(raw)

    def _sanitize(self, raw: str) -> str:
        name = re.sub(r"[^0-9A-Za-z_]", "_", raw)
        if name and name[0].isdigit():
            name = "_" + name
        if name != raw and raw not in self._renamed:
            self._renamed.add(raw)
            logger.warning(f"[milvus] collection name {raw!r} contains invalid characters, using {name!r}")
        return name

    # ------------------------------------------------------------------ schema
    def has_collection(self, name: str) -> bool:
        if name in self._ready:
            return True
        return bool(self.client.has_collection(name))

    def ensure_collection(self, name: str, with_bbox: bool = False) -> bool:
        """Create the collection if missing and make sure it is loaded. Returns True when created."""
        if name in self._ready:
            return False

        if self.client.has_collection(name):
            self.client.load_collection(name)
            self._ready.add(name)
            return False

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        for field in VARCHAR_FIELDS:
            schema.add_field(field, DataType.VARCHAR, max_length=self._varchar_max_length)
        for field in NULLABLE_VARCHAR_FIELDS:
            schema.add_field(field, DataType.VARCHAR, max_length=self._varchar_max_length, nullable=True)
        for field in FLOAT_FIELDS:
            schema.add_field(field, DataType.FLOAT)
        if with_bbox:
            schema.add_field(BBOX_FIELD, DataType.ARRAY, element_type=DataType.FLOAT, max_capacity=4)
        schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=self._dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name=VECTOR_FIELD, index_type=self._index_type, metric_type=self._metric_type)

        self.client.create_collection(name, schema=schema, index_params=index_params)
        self._ready.add(name)
        logger.info(f"[milvus] created collection {name} (dim={self._dim}, {self._index_type}/{self._metric_type})")
        return True

    # ------------------------------------------------------------------ rows
    def has_data_uri(self, name: str, data_uri: str) -> bool:
        """Whether any row in ``name`` already carries this data_uri."""
        if not self.has_collection(name):
            return False

        if name not in self._ready:
            self.client.load_collection(name)
            self._ready.add(name)

        escaped = data_uri.replace("\\", "\\\\").replace('"', '\\"')
        rows = self.client.query(
            collection_name=name,
            filter=f'data_uri == "{escaped}"',
            output_fields=["pk"],
            limit=1,
        )
        return bool(rows)

    def insert(self, name: str, rows: list[dict]) -> int:
        if not rows:
            return 0
        result = self.client.insert(collection_name=name, data=rows)
        return int(result.get("insert_count", len(rows)))
