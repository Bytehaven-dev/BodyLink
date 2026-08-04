from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from bodylink import __version__
from bodylink.config import AppConfig
from bodylink.ui import MainWindow
from bodylink.vision_worker import CameraDevice


class UiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_renders_expected_initial_state(self) -> None:
        window = MainWindow(AppConfig())
        window.resize(1120, 700)
        window.show()
        self.app.processEvents()
        try:
            self.assertEqual(
                window.windowTitle(),
                f"BodyLink v{__version__} - VRChat OSC 全身追踪",
            )
            self.assertEqual(window.version_label.text(), f"v{__version__}")
            self.assertEqual(window.camera_button.text(), "开启摄像头")
            self.assertTrue(window.stable_mode_button.isChecked())
            self.assertFalse(window.calibrate_button.isEnabled())
            self.assertFalse(window.send_button.isEnabled())
            self.assertGreater(window.preview.width(), 600)
            self.assertEqual(window.tabs.count(), 3)
            self.assertEqual(window.tabs.tabText(2), "面捕")
            self.assertFalse(window.face_enable_check.isChecked())
            self.assertTrue(window.face_native_eyes_check.isChecked())
            self.assertFalse(window.face_native_eyes_check.isEnabled())
            self.assertEqual(window.face_fps_combo.currentData(), 20)
            self.assertEqual(window.capture_format_combo.currentData(), "mjpg")
            self.assertIn("低 USB 带宽", window.capture_format_hint.text())
            self.assertTrue(window.vr_assist_check.isChecked())
            self.assertEqual(window.vr_badge.text(), "等待 SteamVR / Pico")

            window._cameras_found(
                [
                    CameraDevice(0, "UHD 4K AF Camera"),
                    CameraDevice(2, "OBS Virtual Camera"),
                ]
            )
            self.assertEqual(window.camera_combo.itemText(0), "UHD 4K AF Camera  [0]")
            self.assertEqual(window.camera_combo.itemText(1), "OBS Virtual Camera  [2]")
            window._camera_ready(
                {
                    "width": 1280,
                    "height": 720,
                    "fps": 30,
                    "format": "MJPG",
                    "requested_format": "MJPG",
                    "camera_index": 0,
                    "backend": "RTMW3D / CUDA",
                }
            )
            self.assertEqual(
                window.header_status.text(),
                "UHD 4K AF Camera · RTMW3D / CUDA",
            )
            self.assertEqual(
                window.resolution_label.text(),
                "1280 x 720 · MJPG · 30 FPS",
            )
        finally:
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
