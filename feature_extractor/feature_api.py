"""Client for the feature extraction API.

    POST {url}?include_feature_map=false   multipart: file=<image>

Only ``feature_vector.data`` from the response is used. Feature maps are never
requested.
"""
import logging
import mimetypes

import requests

logger = logging.getLogger(__name__)

# Content types the API accepts. Anything else falls back to DEFAULT_CONTENT_TYPE.
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"}
DEFAULT_CONTENT_TYPE = "image/jpeg"


class FeatureApiError(Exception):
    pass


def guess_content_type(filename: str) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    if guessed in SUPPORTED_CONTENT_TYPES:
        return guessed
    logger.debug(f"{filename}: content type {guessed!r} not supported, sending as {DEFAULT_CONTENT_TYPE}")
    return DEFAULT_CONTENT_TYPE


class FeatureApiClient:
    def __init__(self, url: str, timeout_seconds: float, expected_dim: int):
        self._url = url
        self._timeout = timeout_seconds
        self._expected_dim = expected_dim
        self._session = requests.Session()

    @property
    def url(self) -> str:
        return self._url

    def extract(self, image: bytes, filename: str, content_type: str | None = None) -> list[float]:
        """Return the feature vector for one image. Raises FeatureApiError on any failure."""
        content_type = content_type or guess_content_type(filename)

        try:
            response = self._session.post(
                self._url,
                params={"include_feature_map": "false"},
                files={"file": (filename, image, content_type)},
                timeout=self._timeout,
            )
        except requests.RequestException as error:
            raise FeatureApiError(f"{filename}: request failed: {error}") from error

        if response.status_code != 200:
            raise FeatureApiError(f"{filename}: HTTP {response.status_code}: {response.text[:300]}")

        try:
            vector = response.json()["feature_vector"]["data"]
        except (ValueError, KeyError, TypeError) as error:
            raise FeatureApiError(f"{filename}: unexpected response body: {error}") from error

        if not isinstance(vector, list) or len(vector) != self._expected_dim:
            size = len(vector) if isinstance(vector, list) else type(vector).__name__
            raise FeatureApiError(f"{filename}: feature vector has {size} values, expected {self._expected_dim}")

        return [float(value) for value in vector]

    def close(self):
        self._session.close()

    def __enter__(self) -> "FeatureApiClient":
        return self

    def __exit__(self, *_):
        self.close()
