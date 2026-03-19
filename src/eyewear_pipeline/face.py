from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(slots=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class HaarFaceDetector:
    def __init__(self, cascade_path: str | None = None) -> None:
        if cascade_path:
            self.classifier = cv2.CascadeClassifier(cascade_path)
        else:
            self.classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections: Sequence[tuple[int, int, int, int]] = self.classifier.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return [FaceBox(x=x, y=y, w=w, h=h, score=1.0) for (x, y, w, h) in detections]

