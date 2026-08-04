from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bodylink.config import AppConfig, load_config, save_config


class ConfigTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            expected = AppConfig(
                camera_index=2,
                camera_format="yuy2",
                target_host="localhost",
                tracker_mode="full",
                user_height_m=1.83,
                face_enabled=True,
                face_native_eyes=False,
                face_fps=30,
                face_smoothing=0.42,
                vr_assist_enabled=False,
            )
            save_config(expected, path)
            actual = load_config(path)
            self.assertEqual(actual.camera_index, 2)
            self.assertEqual(actual.camera_format, "yuy2")
            self.assertEqual(actual.target_host, "localhost")
            self.assertEqual(actual.tracker_mode, "full")
            self.assertAlmostEqual(actual.user_height_m, 1.83)
            self.assertTrue(actual.face_enabled)
            self.assertFalse(actual.face_native_eyes)
            self.assertEqual(actual.face_fps, 30)
            self.assertAlmostEqual(actual.face_smoothing, 0.42)
            self.assertFalse(actual.vr_assist_enabled)

    def test_invalid_values_are_clamped(self) -> None:
        config = AppConfig(
            target_port=99999,
            smoothing=4.0,
            tracker_mode="unknown",
            face_fps=90,
            face_smoothing=-2.0,
            camera_format="invalid",
        )
        config.normalized()
        self.assertEqual(config.target_port, 65535)
        self.assertEqual(config.smoothing, 0.95)
        self.assertEqual(config.tracker_mode, "stable")
        self.assertEqual(config.face_fps, 30)
        self.assertEqual(config.face_smoothing, 0.0)
        self.assertEqual(config.camera_format, "mjpg")


if __name__ == "__main__":
    unittest.main()
