"""Does the detection path crop the right region of the image?

Two ways to use it:

    # 1. Automatic checks on a synthetic image (also runs under pytest)
    python feature_extractor/tests/test_cropping.py
    python feature_extractor/tests/test_cropping.py --save crops_out    # also writes the images to look at

    # 2. Crop a real image from disk with a real bbox and save the result to look at
    python feature_extractor/tests/test_cropping.py --image photo.jpg --bbox 412 88 530 197 --save crops_out

The synthetic image is black with solid-colour rectangles drawn at known places.
Each detection's bbox is one of those rectangles, so a correct crop must be exactly
the rectangle's size and contain only its colour. A crop that is shifted or sized
wrongly, even by one pixel, picks up black pixels and fails.

No MongoDB, object storage, feature API or Milvus is contacted.
"""
import argparse
import io
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extractor.extractor import FeatureExtractor          # noqa: E402
from feature_extractor.imaging import crop, image_size, normalize_bbox   # noqa: E402
from feature_extractor.report import RunReport                      # noqa: E402
from feature_extractor.schemas.inspections import InspectionDocumentV1  # noqa: E402

WIDTH, HEIGHT = 200, 150
BLACK = (0, 0, 0)
RED, GREEN, BLUE, YELLOW = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)

# (name, stored bbox, expected pixel box, colour)
#   pixel      -> used as is
#   normalized -> all values in [0, 1], scaled by image size
#   spill      -> extends past the image edge, clamped
#   fractional -> non-integer pixel coordinates, floor/ceil to cover the whole region
REGIONS = [
    ("pixel",      [20, 30, 80, 90],          (20, 30, 80, 90),     RED),
    ("normalized", [0.5, 0.2, 0.75, 0.6],     (100, 30, 150, 90),   GREEN),
    ("spill",      [170, 120, 400, 400],      (170, 120, 200, 150), BLUE),
    ("fractional", [10.6, 100.2, 60.4, 140.9], (10, 100, 61, 141),  YELLOW),
]
INVALID_BBOXES = [
    ("null", None),
    ("zero-width", [30, 30, 30, 60]),
    ("three-values", [1, 2, 3]),
    ("nan", [float("nan"), 0, 50, 50]),
    ("outside", [300, 300, 400, 400]),
]


# ------------------------------------------------------------------ helpers
def synthetic_image() -> bytes:
    image = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
    draw = ImageDraw.Draw(image)
    for _, _, (x1, y1, x2, y2), colour in REGIONS:
        # PIL's rectangle is inclusive on both ends; the pixel box is half-open.
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=colour)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def unique_colours(data: bytes) -> set:
    with Image.open(io.BytesIO(data)) as image:
        return {colour for _, colour in image.convert("RGB").getcolors(maxcolors=1 << 20)}


def detection_inspection() -> InspectionDocumentV1:
    started = datetime(2026, 6, 20, 8)
    detections = [
        {"prediction": name, "confidence": 0.9, "bbox": bbox} for name, bbox, _, _ in REGIONS
    ] + [
        {"prediction": name, "confidence": 0.5, "bbox": bbox} for name, bbox in INVALID_BBOXES
    ]
    return InspectionDocumentV1.model_validate({
        "_id": "insp-crop-test",
        "metadata": {"gbm": "SEHC", "process": "Side", "location": "Line_01", "equipmentId": "VM07",
                     "createdAt": started, "mode": "production"},
        "inspectionResults": {"aiResults": [{
            "aiModel": "sidedet/sidedet_v1", "task": "det", "backend": "x", "classes": ["a"],
            "predictions": [{
                "dataPath": "images/2026/06/20/AA_2_data.png",
                "inferenceStartedAt": started, "elapsedTime": 1.0,
                "detections": detections,
            }],
        }]},
    })


class FakeStorage:
    def __init__(self, image: bytes):
        self.image = image

    def get_bytes(self, key):
        return self.image


class CapturingApi:
    """Records every image the extractor would send to the feature API."""

    def __init__(self):
        self.calls: list[tuple[str, str | None, bytes]] = []

    def extract(self, image, filename, content_type=None):
        self.calls.append((filename, content_type, image))
        return [0.0] * 4


class FakeStore:
    def __init__(self):
        self.rows: dict[str, list[dict]] = {}

    def collection_name(self, process, name, version, objects=False):
        return f"VFM_{process}_{name}_{version}" + ("_object" if objects else "")

    def ensure_collection(self, name, with_bbox=False):
        self.rows.setdefault(name, [])
        return True

    def has_data_uri(self, name, uri):
        return False

    def insert(self, name, rows):
        self.rows[name].extend(rows)
        return len(rows)


def run_detection(save_dir: Path | None = None):
    image = synthetic_image()
    api = CapturingApi()
    store = FakeStore()
    extractor = FeatureExtractor(FakeStorage(image), api, store)
    report = RunReport()

    inspection = detection_inspection()
    extractor.process(inspection, inspection.inspection_result.ai_results[0], report)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "original.png").write_bytes(image)
        for filename, _, data in api.calls:
            (save_dir / filename).write_bytes(data)
        print(f"wrote original.png and {len(api.calls) - 1} crops to {save_dir}")

    return api, store, report


# ------------------------------------------------------------------ tests
def test_normalize_bbox_valid_regions():
    for name, bbox, expected, _ in REGIONS:
        assert normalize_bbox(bbox, WIDTH, HEIGHT) == expected, name


def test_normalize_bbox_invalid_regions():
    for name, bbox in INVALID_BBOXES:
        assert normalize_bbox(bbox, WIDTH, HEIGHT) is None, name


def test_crop_function_matches_drawn_rectangle():
    image = synthetic_image()
    for name, _, box, colour in REGIONS:
        data = crop(image, box, "PNG")
        assert image_size(data) == (box[2] - box[0], box[3] - box[1]), name
        assert unique_colours(data) == {colour}, f"{name}: crop contains pixels outside the rectangle"


def test_detection_path_sends_full_image_then_one_exact_crop_per_valid_bbox():
    api, store, report = run_detection()

    # First call: the whole image for the main row.
    filename, content_type, data = api.calls[0]
    assert filename == "AA_2_data.png" and image_size(data) == (WIDTH, HEIGHT)

    # Then one crop per valid region, in detection order, named after the detection index.
    crops = api.calls[1:]
    assert len(crops) == len(REGIONS), f"expected {len(REGIONS)} crops, got {len(crops)}"
    for index, ((name, _, box, colour), (filename, content_type, data)) in enumerate(zip(REGIONS, crops)):
        assert filename == f"AA_2_data_bbox{index}.png", name
        assert content_type == "image/png", name
        assert image_size(data) == (box[2] - box[0], box[3] - box[1]), f"{name}: wrong crop size"
        assert unique_colours(data) == {colour}, f"{name}: crop is misaligned"

    # Every invalid bbox was skipped and counted; nothing was sent for it.
    assert report.detections_seen == len(REGIONS) + len(INVALID_BBOXES)
    assert report.detections_bad_bbox == len(INVALID_BBOXES)
    assert report.detections_api_failed == 0


def test_object_rows_carry_the_detection_and_its_bbox():
    _, store, report = run_detection()
    rows = store.rows["VFM_Side_sidedet_sidedet_v1_object"]
    assert len(rows) == len(REGIONS)
    for (name, bbox, _, _), row in zip(REGIONS, rows):
        assert row["prediction"] == name
        assert row["confidence"] == 0.9
        assert row["bbox"] == [float(v) for v in bbox]
        assert row["data_uri"] == "images/2026/06/20/AA_2_data.png"
        assert row["filename"] == "AA_2_data.png"
        assert row["task"] == "det"
    assert report.rows_object_inserted == len(REGIONS)

    main = store.rows["VFM_Side_sidedet_sidedet_v1"]
    assert len(main) == 1
    assert main[0]["prediction"] == "_".join([n for n, *_ in REGIONS] + [n for n, _ in INVALID_BBOXES])


# ------------------------------------------------------------------ manual mode
def crop_real_image(path: Path, bbox: list[float], save_dir: Path):
    data = path.read_bytes()
    width, height = image_size(data)
    box = normalize_bbox(bbox, width, height)
    print(f"image     : {path} ({width}x{height} px)")
    print(f"bbox      : {bbox}")
    print(f"pixel box : {box}")
    if box is None:
        print("bbox is invalid -- the extractor would skip this detection")
        return 1
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{path.stem}_crop.png"
    out.write_bytes(crop(data, box, "PNG"))
    print(f"crop size : {box[2] - box[0]}x{box[3] - box[1]} px")
    print(f"saved     : {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, help="Crop this real image instead of running the checks")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Bounding box for --image, as stored in the inspection document")
    parser.add_argument("--save", type=Path, help="Directory to write the original image and crops into")
    args = parser.parse_args(argv)

    if args.image:
        if not args.bbox:
            parser.error("--image needs --bbox X1 Y1 X2 Y2")
        return crop_real_image(args.image, args.bbox, args.save or Path("crops_out"))

    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")

    if args.save:
        run_detection(args.save)

    print(f"\n{len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
