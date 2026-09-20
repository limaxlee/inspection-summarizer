"""S3-compatible object storage: fetch the bytes behind ``prediction.dataPath``.

Same endpoint, bucket, credentials and v4 signature as the reference
aiobotocore client. This pipeline is synchronous and walks one cursor at a
time, so plain boto3 is used instead of aiobotocore.
"""
import logging

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class ObjectStorageError(Exception):
    pass


class ObjectStorage:
    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str):
        self._endpoint = endpoint
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
        )

    @property
    def target(self) -> str:
        return f"{self._endpoint}/{self._bucket}"

    def ping(self) -> bool:
        try:
            self._client.head_bucket(Bucket=self._bucket)
            logger.info(f"[object storage] connected to {self.target}")
            return True
        except (BotoCoreError, ClientError) as error:
            logger.error(f"[object storage] connection to {self.target} failed: {error}")
            return False

    def get_bytes(self, key: str) -> bytes:
        """Download one object. ``key`` is the dataPath exactly as stored."""
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read()
        except (BotoCoreError, ClientError) as error:
            raise ObjectStorageError(f"{key}: {error}") from error

    def close(self):
        self._client.close()

    def __enter__(self) -> "ObjectStorage":
        return self

    def __exit__(self, *_):
        self.close()
