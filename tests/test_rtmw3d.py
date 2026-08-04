from __future__ import annotations

import unittest

import numpy as np

from bodylink.geometry import CameraIntrinsics, reconstruct_camera_points
from bodylink.pose import Landmark
from bodylink.rtmw3d import (
    Rtmw3dTracker,
    Rtmw3dRuntimeError,
    require_cuda_provider,
    snapshot_from_rtmw3d,
)
from tests.fixtures import synthetic_snapshot


class Rtmw3dMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(1280, 720, 60.0)
        source = synthetic_snapshot(intrinsics=self.intrinsics)
        self.keypoints_3d = np.zeros((1, 133, 3), dtype=np.float64)
        self.keypoints_2d = np.tile(
            source.image_points[Landmark.LEFT_HIP, :2]
            * np.array([self.intrinsics.width, self.intrinsics.height]),
            (1, 133, 1),
        )
        self.scores = np.ones((1, 133), dtype=np.float64)

        direct = {
            0: Landmark.NOSE,
            1: Landmark.LEFT_EYE,
            2: Landmark.RIGHT_EYE,
            3: Landmark.LEFT_EAR,
            4: Landmark.RIGHT_EAR,
            5: Landmark.LEFT_SHOULDER,
            6: Landmark.RIGHT_SHOULDER,
            7: Landmark.LEFT_ELBOW,
            8: Landmark.RIGHT_ELBOW,
            9: Landmark.LEFT_WRIST,
            10: Landmark.RIGHT_WRIST,
            11: Landmark.LEFT_HIP,
            12: Landmark.RIGHT_HIP,
            13: Landmark.LEFT_KNEE,
            14: Landmark.RIGHT_KNEE,
            15: Landmark.LEFT_ANKLE,
            16: Landmark.RIGHT_ANKLE,
            17: Landmark.LEFT_FOOT_INDEX,
            18: Landmark.LEFT_FOOT_INDEX,
            19: Landmark.LEFT_HEEL,
            20: Landmark.RIGHT_FOOT_INDEX,
            21: Landmark.RIGHT_FOOT_INDEX,
            22: Landmark.RIGHT_HEEL,
            71: Landmark.MOUTH_LEFT,
            77: Landmark.MOUTH_RIGHT,
            95: Landmark.LEFT_THUMB,
            99: Landmark.LEFT_INDEX,
            111: Landmark.LEFT_PINKY,
            116: Landmark.RIGHT_THUMB,
            120: Landmark.RIGHT_INDEX,
            132: Landmark.RIGHT_PINKY,
        }
        pixel_scale = np.array([self.intrinsics.width, self.intrinsics.height])
        for rtm_index, landmark in direct.items():
            self.keypoints_2d[0, rtm_index] = source.image_points[landmark, :2] * pixel_scale
            self.keypoints_3d[0, rtm_index, 2] = source.world_points[landmark, 2]

    def _snapshot(self):
        return snapshot_from_rtmw3d(
            self.keypoints_3d,
            self.keypoints_2d,
            self.scores,
            (self.intrinsics.height, self.intrinsics.width, 3),
            2.5,
            self.intrinsics,
        )

    def test_maps_wholebody_points_into_existing_pose_contract(self) -> None:
        snapshot = self._snapshot()
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.image_points.shape, (33, 3))
        self.assertEqual(snapshot.world_points.shape, (33, 3))
        self.assertEqual(snapshot.timestamp_s, 2.5)
        np.testing.assert_allclose(
            snapshot.image_points[Landmark.LEFT_FOOT_INDEX, :2],
            self.keypoints_2d[0, [17, 18]].mean(axis=0)
            / np.array([self.intrinsics.width, self.intrinsics.height]),
        )
        pelvis = snapshot.world_points[[Landmark.LEFT_HIP, Landmark.RIGHT_HIP]].mean(axis=0)
        np.testing.assert_allclose(pelvis, np.zeros(3), atol=1e-9)

    def test_converted_local_skeleton_remains_projection_consistent(self) -> None:
        snapshot = self._snapshot()
        assert snapshot is not None
        _, info = reconstruct_camera_points(snapshot, self.intrinsics, 1.0)
        self.assertLess(info.median_error_px, 1e-6)
        self.assertGreater(info.translation[2], 1.0)

    def test_foot_confidence_is_average_of_big_and_small_toe(self) -> None:
        self.scores[0, 17] = 0.2
        self.scores[0, 18] = 0.8
        snapshot = self._snapshot()
        assert snapshot is not None
        self.assertAlmostEqual(snapshot.confidence[Landmark.LEFT_FOOT_INDEX], 0.5)

    def test_invalid_or_empty_people_return_none(self) -> None:
        result = snapshot_from_rtmw3d(
            np.empty((0, 133, 3)),
            np.empty((0, 133, 2)),
            np.empty((0, 133)),
            (720, 1280, 3),
            1.0,
            self.intrinsics,
        )
        self.assertIsNone(result)


class CudaProviderTests(unittest.TestCase):
    def test_cuda_provider_is_required(self) -> None:
        require_cuda_provider(["CUDAExecutionProvider", "CPUExecutionProvider"])
        with self.assertRaises(Rtmw3dRuntimeError):
            require_cuda_provider(["CPUExecutionProvider"])


class DetectionBoxTests(unittest.TestCase):
    def test_empty_detection_does_not_force_full_frame_pose(self) -> None:
        self.assertIsNone(Rtmw3dTracker._largest_detection([], (720, 1280, 3)))

    def test_largest_person_detection_is_selected_and_clipped(self) -> None:
        result = Rtmw3dTracker._largest_detection(
            np.array(
                [
                    [10.0, 20.0, 60.0, 100.0],
                    [-20.0, 30.0, 900.0, 800.0],
                ]
            ),
            (720, 1280, 3),
        )
        self.assertIsNotNone(result)
        assert result is not None
        np.testing.assert_allclose(result, [[0.0, 30.0, 900.0, 720.0]])


if __name__ == "__main__":
    unittest.main()
