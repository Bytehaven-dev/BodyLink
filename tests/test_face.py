from __future__ import annotations

import math
import unittest

from bodylink.face import (
    BlendshapeSmoother,
    clamp_coefficient,
    face_output,
    gaze_axes,
    neutral_face_output,
)


class FaceMappingTests(unittest.TestCase):
    def test_coefficients_are_clamped_and_non_finite_values_are_zero(self) -> None:
        self.assertEqual(clamp_coefficient(-1.0), 0.0)
        self.assertEqual(clamp_coefficient(2.0), 1.0)
        self.assertEqual(clamp_coefficient(math.nan), 0.0)
        self.assertEqual(clamp_coefficient(math.inf), 0.0)

    def test_gaze_uses_mediapipe_opposite_side_categories(self) -> None:
        left_x, left_y, right_x, right_y = gaze_axes(
            {
                "eyeLookOutRight": 0.8,
                "eyeLookInRight": 0.1,
                "eyeLookUpRight": 0.6,
                "eyeLookDownRight": 0.2,
                "eyeLookInLeft": 0.7,
                "eyeLookOutLeft": 0.3,
                "eyeLookDownLeft": 0.9,
                "eyeLookUpLeft": 0.1,
            }
        )
        self.assertAlmostEqual(left_x, 0.7)
        self.assertAlmostEqual(left_y, 0.4)
        self.assertAlmostEqual(right_x, 0.4)
        self.assertAlmostEqual(right_y, -0.8)

    def test_vrcft_eyelids_combine_blink_openness_and_eye_wide(self) -> None:
        output = face_output(
            {
                "eyeBlinkLeft": 1.0,
                "eyeBlinkRight": 0.0,
                "eyeWideLeft": 0.0,
                "eyeWideRight": 1.0,
            },
            native_eyes=False,
        )
        self.assertEqual(output.parameters["v2/EyeLidLeft"], 0.0)
        self.assertEqual(output.parameters["v2/EyeOpenLeft"], 0.0)
        self.assertEqual(output.parameters["v2/EyeClosedLeft"], 1.0)
        self.assertEqual(output.parameters["v2/EyeLidRight"], 1.0)

    def test_combined_jaw_and_mouth_axes_are_signed(self) -> None:
        output = face_output(
            {
                "jawRight": 0.75,
                "jawLeft": 0.20,
                "jawForward": 0.45,
                "mouthRight": 0.10,
                "mouthLeft": 0.60,
            },
            native_eyes=True,
        )
        self.assertAlmostEqual(output.parameters["v2/JawX"], 0.55)
        self.assertAlmostEqual(output.parameters["v2/JawZ"], 0.45)
        self.assertAlmostEqual(output.parameters["v2/MouthX"], -0.50)

    def test_smoothing_retains_configured_fraction_of_previous_frame(self) -> None:
        smoother = BlendshapeSmoother()
        self.assertEqual(smoother.update({"jawOpen": 0.0}, 0.75)["jawOpen"], 0.0)
        self.assertAlmostEqual(smoother.update({"jawOpen": 1.0}, 0.75)["jawOpen"], 0.25)
        smoother.reset()
        self.assertEqual(smoother.update({"jawOpen": 1.0}, 0.75)["jawOpen"], 1.0)

    def test_neutral_output_opens_eyes_and_disables_tracking_flags(self) -> None:
        output = neutral_face_output(native_eyes=False)
        self.assertEqual(output.parameters["v2/EyeLidLeft"], 0.75)
        self.assertEqual(output.parameters["v2/EyeOpenLeft"], 1.0)
        self.assertEqual(output.parameters["v2/EyeClosedLeft"], 0.0)
        self.assertIs(output.parameters["EyeTrackingActive"], False)
        self.assertIs(output.parameters["ExpressionTrackingActive"], False)
        self.assertIs(output.parameters["LipTrackingActive"], False)

    def test_native_eye_output_converts_normalized_gaze_to_degrees(self) -> None:
        output = face_output(
            {
                "eyeLookOutRight": 0.5,
                "eyeLookUpRight": 0.4,
                "eyeBlinkLeft": 0.8,
                "eyeBlinkRight": 0.2,
            },
            native_eyes=True,
        )
        self.assertIsNotNone(output.native_eye)
        assert output.native_eye is not None
        self.assertAlmostEqual(output.native_eye.left_yaw_deg, 15.0)
        self.assertAlmostEqual(output.native_eye.left_pitch_deg, -10.0)
        self.assertAlmostEqual(output.native_eye.closed_amount, 0.5)


if __name__ == "__main__":
    unittest.main()
