from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2

from bodylink.config import AppConfig
from bodylink.vision_worker import (
    CameraProbeThread,
    _open_capture,
    fourcc_name,
    requested_fourcc_name,
)


class FakeCapture:
    def __init__(self) -> None:
        self.settings: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> bool:
        self.settings.append((prop, value))
        return True

    def release(self) -> None:
        pass


class CameraCaptureTests(unittest.TestCase):
    def test_named_devices_are_listed_without_opening_busy_cameras(self) -> None:
        thread = CameraProbeThread(max_index=1)
        emissions: list[list[object]] = []
        thread.cameras_found.connect(emissions.append)

        with (
            patch(
                "bodylink.vision_worker.directshow_camera_names",
                return_value=("UHD 4K AF Camera", "Virtual Camera"),
            ),
            patch("bodylink.vision_worker.cv2.VideoCapture") as video_capture,
        ):
            thread.run()

        video_capture.assert_not_called()
        self.assertEqual(len(emissions), 1)
        self.assertEqual(
            [(device.index, device.name) for device in emissions[0]],
            [(0, "UHD 4K AF Camera"), (1, "Virtual Camera")],
        )

    def test_mjpg_is_applied_after_resolution_and_fps(self) -> None:
        capture = FakeCapture()
        config = AppConfig(camera_format="mjpg", camera_fps=30)
        with patch("bodylink.vision_worker.cv2.VideoCapture", return_value=capture):
            actual = _open_capture(config)

        self.assertIs(actual, capture)
        properties = [prop for prop, _ in capture.settings]
        fourcc_index = properties.index(cv2.CAP_PROP_FOURCC)
        self.assertGreater(fourcc_index, properties.index(cv2.CAP_PROP_FRAME_WIDTH))
        self.assertGreater(fourcc_index, properties.index(cv2.CAP_PROP_FRAME_HEIGHT))
        self.assertGreater(fourcc_index, properties.index(cv2.CAP_PROP_FPS))
        self.assertEqual(fourcc_name(capture.settings[fourcc_index][1]), "MJPG")
        self.assertIn((cv2.CAP_PROP_FPS, 30), capture.settings)

    def test_auto_leaves_fourcc_to_the_driver(self) -> None:
        capture = FakeCapture()
        with patch("bodylink.vision_worker.cv2.VideoCapture", return_value=capture):
            _open_capture(AppConfig(camera_format="auto"))

        properties = [prop for prop, _ in capture.settings]
        self.assertNotIn(cv2.CAP_PROP_FOURCC, properties)

    def test_fourcc_name_handles_empty_value(self) -> None:
        self.assertEqual(fourcc_name(0), "AUTO")

    def test_requested_fourcc_name_maps_config_values(self) -> None:
        self.assertEqual(requested_fourcc_name("mjpg"), "MJPG")
        self.assertEqual(requested_fourcc_name("yuy2"), "YUY2")
        self.assertEqual(requested_fourcc_name("auto"), "AUTO")


if __name__ == "__main__":
    unittest.main()
