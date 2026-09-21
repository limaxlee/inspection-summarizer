from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class DataModel(BaseModel):
    """Base model matching how the documents are actually stored in MongoDB.

    Fields are declared snake_case and stored camelCase. ``populate_by_name``
    lets a model be built from either form.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
        arbitrary_types_allowed=True,
    )


def to_mongo(value: Any) -> Any:
    """Recursively convert a dumped model into BSON-encodable primitives.

    ``model_dump()`` leaves Enum members in place and BSON cannot encode them,
    while ``model_dump(mode="json")`` would turn datetimes into strings, which
    breaks range queries on ``startDate``. So convert only the enums.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: to_mongo(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_mongo(item) for item in value]
    return value
