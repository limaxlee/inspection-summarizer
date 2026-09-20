"""Image helpers: size lookup, bounding-box validation and cropping."""
import io
import math

from PIL import Image

FORMAT_CONTENT_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "BMP": "image/bmp", "WEBP": "image/webp"}
FORMAT_EXTENSIONS = {"PNG": "png", "JPEG": "jpg", "BMP": "bmp", "WEBP": "webp"}

PixelBox = tuple[int, int, int, int]


def image_size(data: bytes) -> tuple[int, int]:
    """(width, height) of an encoded image."""
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def normalize_bbox(bbox, width: int, height: int) -> PixelBox | None:
    """Turn a stored ``[x1, y1, x2, y2]`` into integer pixel bounds inside the image.

    Coordinates are taken as pixels. When all four values lie in [0, 1] they are
    taken as fractions of the image size instead. Returns None when the box is
    missing, malformed, non-finite, or has no area after clamping to the image.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    if any(not math.isfinite(value) for value in values):
        return None

    if all(0.0 <= value <= 1.0 for value in values):
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]

    x1, y1, x2, y2 = values
    left = min(max(int(math.floor(x1)), 0), width)
    top = min(max(int(math.floor(y1)), 0), height)
    right = min(max(int(math.ceil(x2)), 0), width)
    bottom = min(max(int(math.ceil(y2)), 0), height)

    if right - left < 1 or bottom - top < 1:
        return None

    return left, top, right, bottom


def crop(data: bytes, box: PixelBox, image_format: str = "PNG") -> bytes:
    """Crop ``box`` out of an encoded image and re-encode it as ``image_format``."""
    image_format = image_format.upper()
    with Image.open(io.BytesIO(data)) as image:
        region = image.crop(box)
        if image_format == "JPEG" and region.mode not in ("RGB", "L"):
            region = region.convert("RGB")
        buffer = io.BytesIO()
        region.save(buffer, format=image_format)
        return buffer.getvalue()
