"""Thin wrapper around the trained YOLO strawberry model.

Keeps all the ultralytics-specific handling in one place so the UI (and any
other caller) just gets an annotated image plus a list of detections back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# perception/ui/detector.py -> perception/models/
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "strawberry_v1.pt"


def list_models() -> list[Path]:
    """Every .pt checkpoint sitting in perception/models/, newest name last.

    Sorted so strawberry_v1, strawberry_v2, ... come out in order; the UI picks
    the last one as the default.
    """
    if not MODELS_DIR.is_dir():
        return []
    return sorted(MODELS_DIR.glob("*.pt"))


@dataclass
class Detection:
    confidence: float
    # xyxy pixel coordinates in the source image
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class DetectionResult:
    image: np.ndarray          # RGB, with boxes/masks drawn on
    detections: list[Detection]

    @property
    def count(self) -> int:
        return len(self.detections)


class StrawberryDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.model_path}. "
                "Expected the trained checkpoint from the Setup notebooks."
            )
        self._model = None

    @property
    def model(self):
        """Load the model lazily so importing this module stays cheap."""
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - env hint only
                raise ImportError(
                    "ultralytics is not installed. Run `pip install -r requirements.txt` "
                    "inside the harvestai environment."
                ) from exc
            self._model = YOLO(str(self.model_path))
        return self._model

    def detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> DetectionResult:
        """Run detection on a single RGB image.

        Args:
            image: HxWx3 RGB uint8 array (what Gradio hands us).
            conf: confidence threshold.
            iou: NMS IoU threshold.
        """
        if image is None:
            raise ValueError("No image provided")

        # ultralytics expects BGR when given a raw numpy array
        bgr = image[:, :, ::-1]
        results = self.model.predict(source=bgr, conf=conf, iou=iou, verbose=False)
        result = results[0]

        detections: list[Detection] = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                detections.append(
                    Detection(float(c), float(x1), float(y1), float(x2), float(y2))
                )

        # result.plot() returns a BGR image with boxes + masks already drawn
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1].copy()

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return DetectionResult(image=annotated_rgb, detections=detections)


_DETECTOR_CACHE: dict[str, StrawberryDetector] = {}


def get_detector(model_path: str | Path | None = None) -> StrawberryDetector:
    """Return a detector for the given weights, reusing loaded ones.

    Keeps one StrawberryDetector per checkpoint path so switching models in the
    UI doesn't re-read weights you've already loaded this session.
    """
    key = str(Path(model_path).resolve()) if model_path else str(DEFAULT_MODEL_PATH)
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = StrawberryDetector(key)
    return _DETECTOR_CACHE[key]
