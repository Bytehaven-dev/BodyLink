from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from bodylink.geometry import (
    CameraIntrinsics,
    TrackerPose,
    TrackingStabilizer,
    align_calibration_to_vr,
    calibrate_pose,
    calibration_pose_ready,
    reconstruct_camera_points,
    rotate_y,
    trackers_from_pose,
)
from bodylink.pose import Landmark
from bodylink.vr_runtime import TrackedPose, VrPoseSnapshot
from tests.fixtures import synthetic_snapshot


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(1280, 720, 60.0)
        self.samples = [synthetic_snapshot(timestamp_s=index / 30) for index in range(24)]

    def test_reconstruction_recovers_camera_translation(self) -> None:
        snapshot = synthetic_snapshot(np.array([0.18, -0.07, 3.2]))
        _, info = reconstruct_camera_points(snapshot, self.intrinsics, 1.0)
        np.testing.assert_allclose(info.translation, [0.18, -0.07, 3.2], atol=1e-8)
        self.assertLess(info.median_error_px, 1e-8)

    def test_calibration_sets_floor_and_forward_axis(self) -> None:
        calibration = calibrate_pose(self.samples, self.intrinsics, 1.70)
        self.assertGreater(calibration.scale, 0.7)
        self.assertLess(calibration.scale, 1.3)
        self.assertAlmostEqual(calibration.yaw_correction_deg, 0.0, delta=0.1)

        camera_points, _ = reconstruct_camera_points(
            self.samples[-1], self.intrinsics, calibration.scale
        )
        tracking = calibration.camera_to_tracking(camera_points)
        sole_y = tracking[[29, 30, 31, 32], 1]
        self.assertLessEqual(float(np.min(sole_y)), 0.001)
        self.assertGreater(float(np.mean(tracking[[23, 24], 1])), 0.75)

    def test_calibration_does_not_require_head_landmarks(self) -> None:
        samples = []
        head = np.asarray(
            [
                Landmark.NOSE,
                Landmark.LEFT_EYE_INNER,
                Landmark.LEFT_EYE,
                Landmark.LEFT_EYE_OUTER,
                Landmark.RIGHT_EYE_INNER,
                Landmark.RIGHT_EYE,
                Landmark.RIGHT_EYE_OUTER,
                Landmark.LEFT_EAR,
                Landmark.RIGHT_EAR,
            ],
            dtype=np.intp,
        )
        for sample in self.samples:
            sample.visibility[head] = 0.0
            sample.presence[head] = 0.0
            sample.image_points[head, :2] = 0.0
            samples.append(sample)
        self.assertTrue(calibration_pose_ready(samples[-1]))
        calibration = calibrate_pose(samples, self.intrinsics, 1.70)
        self.assertFalse(calibration.vr_assisted)

    def test_vr_alignment_recovers_yaw_and_horizontal_offset(self) -> None:
        calibration = calibrate_pose(self.samples, self.intrinsics, 1.70)
        expected_yaw = 24.0
        expected_offset = np.array([0.35, 0.0, -0.28])
        identity = np.eye(3)
        vr_samples = []
        for sample in self.samples:
            camera_points, _ = reconstruct_camera_points(
                sample, self.intrinsics, calibration.scale
            )
            tracking = calibration.camera_to_tracking(camera_points)
            left = rotate_y(tracking[Landmark.LEFT_WRIST], expected_yaw) + expected_offset
            right = rotate_y(tracking[Landmark.RIGHT_WRIST], expected_yaw) + expected_offset
            timestamp = sample.timestamp_s
            vr_samples.append(
                VrPoseSnapshot(
                    timestamp,
                    hmd=TrackedPose(np.array([0.0, 1.60, 0.0]), identity, timestamp),
                    left_hand=TrackedPose(left, identity, timestamp),
                    right_hand=TrackedPose(right, identity, timestamp),
                )
            )
        aligned, info = align_calibration_to_vr(
            calibration,
            self.samples,
            vr_samples,
            self.intrinsics,
        )
        self.assertTrue(aligned.vr_assisted)
        self.assertAlmostEqual(aligned.vr_yaw_offset_deg, expected_yaw, delta=0.1)
        np.testing.assert_allclose(aligned.tracking_offset_m, expected_offset, atol=1e-6)
        self.assertLess(info.horizontal_error_m, 1e-6)

    def test_tracker_modes_have_expected_ids(self) -> None:
        calibration = calibrate_pose(self.samples, self.intrinsics, 1.70)
        stable, _ = trackers_from_pose(
            self.samples[-1], calibration, self.intrinsics, "stable"
        )
        full, _ = trackers_from_pose(self.samples[-1], calibration, self.intrinsics, "full")
        self.assertEqual([pose.tracker_id for pose in stable], [1, 2, 3])
        self.assertEqual([pose.tracker_id for pose in full], list(range(1, 9)))
        self.assertTrue(all(pose.position_m.shape == (3,) for pose in full))
        self.assertTrue(all(pose.euler_deg.shape == (3,) for pose in full))

    def test_vr_hand_anchor_only_constrains_elbow_trackers(self) -> None:
        calibration = calibrate_pose(self.samples, self.intrinsics, 1.70)
        calibration = replace(calibration, vr_assisted=True)
        camera_points, _ = reconstruct_camera_points(
            self.samples[-1], self.intrinsics, calibration.scale
        )
        tracking = calibration.camera_to_tracking(camera_points)
        timestamp = self.samples[-1].timestamp_s
        identity = np.eye(3)
        vr_pose = VrPoseSnapshot(
            timestamp,
            hmd=TrackedPose(np.array([0.0, 1.60, 0.0]), identity, timestamp),
            left_hand=TrackedPose(
                tracking[Landmark.LEFT_WRIST] + np.array([0.0, 0.05, 0.03]),
                identity,
                timestamp,
            ),
            right_hand=TrackedPose(
                tracking[Landmark.RIGHT_WRIST] + np.array([0.0, 0.05, 0.03]),
                identity,
                timestamp,
            ),
        )
        trackers, _ = trackers_from_pose(
            self.samples[-1], calibration, self.intrinsics, "full", vr_pose
        )
        self.assertEqual([pose.tracker_id for pose in trackers], list(range(1, 9)))
        self.assertEqual({pose.role for pose in trackers}, {
            "hip", "left_foot", "right_foot", "chest",
            "left_knee", "right_knee", "left_elbow", "right_elbow",
        })

    def test_stabilizer_holds_and_then_drops_missing_tracker(self) -> None:
        stabilizer = TrackingStabilizer(hold_seconds=0.25)
        pose = TrackerPose(
            role="hip",
            tracker_id=1,
            label="腰",
            position_m=np.array([0.0, 1.0, 0.0]),
            euler_deg=np.zeros(3),
            confidence=0.9,
        )
        first = stabilizer.update([pose], 1.0, 0.5, 0.5)
        held = stabilizer.update([], 1.1, 0.5, 0.5)
        dropped = stabilizer.update([], 1.4, 0.5, 0.5)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(held), 1)
        self.assertTrue(held[0].stale)
        self.assertEqual(dropped, [])

    def test_stabilizer_rejects_single_frame_teleport(self) -> None:
        stabilizer = TrackingStabilizer()
        base = TrackerPose(
            role="hip",
            tracker_id=1,
            label="腰",
            position_m=np.array([0.0, 1.0, 0.0]),
            euler_deg=np.zeros(3),
            confidence=0.9,
        )
        stabilizer.update([base], 1.0, 0.6, 0.5)
        outlier = base.copy()
        outlier.position_m = np.array([5.0, 1.0, 0.0])
        result = stabilizer.update([outlier], 1.03, 0.6, 0.5)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].stale)
        np.testing.assert_allclose(result[0].position_m, base.position_m)


if __name__ == "__main__":
    unittest.main()
