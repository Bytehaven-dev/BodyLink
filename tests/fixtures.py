from __future__ import annotations

import numpy as np

from bodylink.geometry import CameraIntrinsics
from bodylink.pose import Landmark, PoseSnapshot


def synthetic_world() -> np.ndarray:
    points = np.zeros((33, 3), dtype=np.float64)
    points[Landmark.NOSE] = [0.00, -0.76, -0.03]
    points[Landmark.LEFT_EYE] = [0.035, -0.78, -0.025]
    points[Landmark.RIGHT_EYE] = [-0.035, -0.78, -0.025]
    points[Landmark.LEFT_EAR] = [0.075, -0.75, 0.00]
    points[Landmark.RIGHT_EAR] = [-0.075, -0.75, 0.00]
    points[Landmark.LEFT_SHOULDER] = [0.22, -0.53, 0.00]
    points[Landmark.RIGHT_SHOULDER] = [-0.22, -0.53, 0.00]
    points[Landmark.LEFT_ELBOW] = [0.43, -0.51, -0.01]
    points[Landmark.RIGHT_ELBOW] = [-0.43, -0.51, -0.01]
    points[Landmark.LEFT_WRIST] = [0.66, -0.49, -0.02]
    points[Landmark.RIGHT_WRIST] = [-0.66, -0.49, -0.02]
    points[Landmark.LEFT_HIP] = [0.105, 0.00, 0.00]
    points[Landmark.RIGHT_HIP] = [-0.105, 0.00, 0.00]
    points[Landmark.LEFT_KNEE] = [0.11, 0.47, -0.015]
    points[Landmark.RIGHT_KNEE] = [-0.11, 0.47, -0.015]
    points[Landmark.LEFT_ANKLE] = [0.12, 0.88, 0.00]
    points[Landmark.RIGHT_ANKLE] = [-0.12, 0.88, 0.00]
    points[Landmark.LEFT_HEEL] = [0.12, 0.94, 0.04]
    points[Landmark.RIGHT_HEEL] = [-0.12, 0.94, 0.04]
    points[Landmark.LEFT_FOOT_INDEX] = [0.12, 0.94, -0.18]
    points[Landmark.RIGHT_FOOT_INDEX] = [-0.12, 0.94, -0.18]
    return points


def synthetic_snapshot(
    translation: np.ndarray | None = None,
    timestamp_s: float = 1.0,
    intrinsics: CameraIntrinsics | None = None,
) -> PoseSnapshot:
    intrinsics = intrinsics or CameraIntrinsics(1280, 720, 60.0)
    translation = (
        np.asarray(translation, dtype=np.float64)
        if translation is not None
        else np.array([0.0, 0.0, 3.5], dtype=np.float64)
    )
    world = synthetic_world()
    camera = world + translation
    pixel_x = intrinsics.focal_x * camera[:, 0] / camera[:, 2] + intrinsics.center_x
    pixel_y = intrinsics.focal_y * camera[:, 1] / camera[:, 2] + intrinsics.center_y
    image = np.column_stack(
        [pixel_x / intrinsics.width, pixel_y / intrinsics.height, world[:, 2]]
    )
    return PoseSnapshot(
        image_points=image,
        world_points=world,
        visibility=np.ones(33, dtype=np.float64),
        presence=np.ones(33, dtype=np.float64),
        timestamp_s=timestamp_s,
    )

