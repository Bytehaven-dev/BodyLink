from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np


class Landmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
)


@dataclass(slots=True)
class PoseSnapshot:
    image_points: np.ndarray
    world_points: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray
    timestamp_s: float

    def __post_init__(self) -> None:
        if self.image_points.shape != (33, 3):
            raise ValueError("image_points must have shape (33, 3)")
        if self.world_points.shape != (33, 3):
            raise ValueError("world_points must have shape (33, 3)")
        if self.visibility.shape != (33,) or self.presence.shape != (33,):
            raise ValueError("confidence arrays must have shape (33,)")

    @property
    def confidence(self) -> np.ndarray:
        return np.minimum(self.visibility, self.presence)

    def score(self, indices: Iterable[int]) -> float:
        selected = np.asarray(list(indices), dtype=np.intp)
        if selected.size == 0:
            return 0.0
        return float(np.clip(np.mean(self.confidence[selected]), 0.0, 1.0))


def median_snapshot(samples: list[PoseSnapshot]) -> PoseSnapshot:
    if not samples:
        raise ValueError("at least one pose sample is required")

    return PoseSnapshot(
        image_points=np.median(np.stack([sample.image_points for sample in samples]), axis=0),
        world_points=np.median(np.stack([sample.world_points for sample in samples]), axis=0),
        visibility=np.median(np.stack([sample.visibility for sample in samples]), axis=0),
        presence=np.median(np.stack([sample.presence for sample in samples]), axis=0),
        timestamp_s=samples[-1].timestamp_s,
    )
