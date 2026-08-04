from __future__ import annotations

import unittest

import numpy as np

from bodylink.vr_runtime import TrackedPose, VrPoseSnapshot, tracked_pose_from_openvr_matrix


class FakeMatrix34:
    def __init__(self, values: list[list[float]]) -> None:
        self.m = values


class VrRuntimeTests(unittest.TestCase):
    def test_openvr_pose_is_converted_to_vrchat_coordinates(self) -> None:
        pose = tracked_pose_from_openvr_matrix(
            FakeMatrix34(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                ]
            ),
            4.0,
            "PICO 4 Ultra",
        )
        np.testing.assert_allclose(pose.position_m, [1.0, 2.0, -3.0])
        np.testing.assert_allclose(pose.rotation, np.eye(3))
        self.assertAlmostEqual(pose.yaw_deg, 0.0)
        self.assertEqual(pose.device_name, "PICO 4 Ultra")

    def test_snapshot_requires_hmd_and_both_controllers(self) -> None:
        identity = np.eye(3)
        tracked = TrackedPose(np.zeros(3), identity, 1.0)
        partial = VrPoseSnapshot(1.0, hmd=tracked, left_hand=tracked)
        ready = VrPoseSnapshot(
            1.0,
            hmd=tracked,
            left_hand=tracked,
            right_hand=tracked,
        )
        self.assertFalse(partial.ready)
        self.assertEqual(partial.controller_count, 1)
        self.assertTrue(ready.ready)


if __name__ == "__main__":
    unittest.main()
