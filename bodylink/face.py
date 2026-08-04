from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np


FaceParameter = float | bool


@dataclass(frozen=True, slots=True)
class FaceSnapshot:
    landmarks: np.ndarray
    blendshapes: dict[str, float]
    timestamp_s: float


@dataclass(frozen=True, slots=True)
class NativeEyeState:
    left_pitch_deg: float
    left_yaw_deg: float
    right_pitch_deg: float
    right_yaw_deg: float
    closed_amount: float


@dataclass(frozen=True, slots=True)
class FaceOutput:
    parameters: Mapping[str, FaceParameter]
    native_eye: NativeEyeState | None


def clamp_coefficient(value: float) -> float:
    value = float(value)
    if not isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def snapshot_from_mediapipe(result: object, timestamp_s: float) -> FaceSnapshot | None:
    faces = getattr(result, "face_landmarks", None)
    face_blendshapes = getattr(result, "face_blendshapes", None)
    if not faces or not face_blendshapes:
        return None

    landmarks = np.asarray(
        [[point.x, point.y, point.z] for point in faces[0]],
        dtype=np.float32,
    )
    blendshapes = {
        category.category_name: clamp_coefficient(category.score)
        for category in face_blendshapes[0]
        if category.category_name
    }
    if landmarks.shape[0] < 468 or not blendshapes:
        return None
    return FaceSnapshot(landmarks=landmarks, blendshapes=blendshapes, timestamp_s=timestamp_s)


class BlendshapeSmoother:
    def __init__(self) -> None:
        self._state: dict[str, float] = {}

    def reset(self) -> None:
        self._state.clear()

    def update(self, blendshapes: Mapping[str, float], smoothing: float) -> dict[str, float]:
        amount = max(0.0, min(0.95, float(smoothing)))
        current = {name: clamp_coefficient(value) for name, value in blendshapes.items()}
        if not self._state:
            self._state = current
            return dict(current)

        filtered: dict[str, float] = {}
        for name in self._state.keys() | current.keys():
            previous = self._state.get(name, 0.0)
            value = current.get(name, 0.0)
            filtered[name] = clamp_coefficient(previous * amount + value * (1.0 - amount))
        self._state = filtered
        return dict(filtered)


def _shape(blendshapes: Mapping[str, float], name: str) -> float:
    return clamp_coefficient(blendshapes.get(name, 0.0))


def gaze_axes(blendshapes: Mapping[str, float]) -> tuple[float, float, float, float]:
    # MediaPipe's directional category side is opposite the VRCFT eye side.
    left_x = _shape(blendshapes, "eyeLookOutRight") - _shape(
        blendshapes, "eyeLookInRight"
    )
    right_x = _shape(blendshapes, "eyeLookInLeft") - _shape(
        blendshapes, "eyeLookOutLeft"
    )
    left_y = _shape(blendshapes, "eyeLookUpRight") - _shape(
        blendshapes, "eyeLookDownRight"
    )
    right_y = _shape(blendshapes, "eyeLookUpLeft") - _shape(
        blendshapes, "eyeLookDownLeft"
    )
    return tuple(max(-1.0, min(1.0, value)) for value in (left_x, left_y, right_x, right_y))


_DIRECT_PARAMETERS = {
    "v2/EyeSquintLeft": "eyeSquintLeft",
    "v2/EyeSquintRight": "eyeSquintRight",
    "v2/EyeWideLeft": "eyeWideLeft",
    "v2/EyeWideRight": "eyeWideRight",
    "v2/BrowDownLeft": "browDownLeft",
    "v2/BrowDownRight": "browDownRight",
    "v2/BrowInnerUp": "browInnerUp",
    "v2/BrowOuterUpLeft": "browOuterUpLeft",
    "v2/BrowOuterUpRight": "browOuterUpRight",
    "v2/CheekSquintLeft": "cheekSquintLeft",
    "v2/CheekSquintRight": "cheekSquintRight",
    "v2/JawOpen": "jawOpen",
    "v2/MouthClosed": "mouthClose",
    "v2/LipFunnel": "mouthFunnel",
    "v2/LipPucker": "mouthPucker",
    "v2/LipSuckUpper": "mouthRollUpper",
    "v2/LipSuckLower": "mouthRollLower",
    "v2/MouthLowerDownLeft": "mouthLowerDownLeft",
    "v2/MouthLowerDownRight": "mouthLowerDownRight",
    "v2/MouthPressLeft": "mouthPressLeft",
    "v2/MouthPressRight": "mouthPressRight",
    "v2/MouthRaiserUpper": "mouthShrugUpper",
    "v2/MouthRaiserLower": "mouthShrugLower",
    "v2/MouthSmileLeft": "mouthSmileLeft",
    "v2/MouthSmileRight": "mouthSmileRight",
    "v2/MouthFrownLeft": "mouthFrownLeft",
    "v2/MouthFrownRight": "mouthFrownRight",
    "v2/MouthDimpleLeft": "mouthDimpleLeft",
    "v2/MouthDimpleRight": "mouthDimpleRight",
    "v2/MouthStretchLeft": "mouthStretchLeft",
    "v2/MouthStretchRight": "mouthStretchRight",
    "v2/MouthUpperUpLeft": "mouthUpperUpLeft",
    "v2/MouthUpperUpRight": "mouthUpperUpRight",
    "v2/NoseSneerLeft": "noseSneerLeft",
    "v2/NoseSneerRight": "noseSneerRight",
}


def face_output(blendshapes: Mapping[str, float], native_eyes: bool) -> FaceOutput:
    parameters: dict[str, FaceParameter] = {
        parameter: _shape(blendshapes, shape)
        for parameter, shape in _DIRECT_PARAMETERS.items()
    }
    parameters.update(
        {
            "v2/CheekPuffSuck": _shape(blendshapes, "cheekPuff"),
            "v2/JawX": _shape(blendshapes, "jawRight")
            - _shape(blendshapes, "jawLeft"),
            "v2/JawZ": _shape(blendshapes, "jawForward"),
            "v2/MouthX": _shape(blendshapes, "mouthRight")
            - _shape(blendshapes, "mouthLeft"),
            "EyeTrackingActive": True,
            "ExpressionTrackingActive": True,
            "LipTrackingActive": True,
        }
    )

    left_x, left_y, right_x, right_y = gaze_axes(blendshapes)
    blink_left = _shape(blendshapes, "eyeBlinkLeft")
    blink_right = _shape(blendshapes, "eyeBlinkRight")
    wide_left = _shape(blendshapes, "eyeWideLeft")
    wide_right = _shape(blendshapes, "eyeWideRight")
    if native_eyes:
        native_eye = NativeEyeState(
            left_pitch_deg=-left_y * 25.0,
            left_yaw_deg=left_x * 30.0,
            right_pitch_deg=-right_y * 25.0,
            right_yaw_deg=right_x * 30.0,
            closed_amount=(blink_left + blink_right) * 0.5,
        )
    else:
        openness_left = 1.0 - blink_left
        openness_right = 1.0 - blink_right
        parameters.update(
            {
                "v2/EyeLeftX": left_x,
                "v2/EyeLeftY": left_y,
                "v2/EyeRightX": right_x,
                "v2/EyeRightY": right_y,
                "v2/EyeLidLeft": openness_left * 0.75 + wide_left * 0.25,
                "v2/EyeLidRight": openness_right * 0.75 + wide_right * 0.25,
                "v2/EyeOpenLeft": openness_left,
                "v2/EyeOpenRight": openness_right,
                "v2/EyeClosedLeft": blink_left,
                "v2/EyeClosedRight": blink_right,
            }
        )
        native_eye = None
    return FaceOutput(parameters=parameters, native_eye=native_eye)


def neutral_face_output(native_eyes: bool) -> FaceOutput:
    output = face_output({}, native_eyes)
    parameters = dict(output.parameters)
    parameters["EyeTrackingActive"] = False
    parameters["ExpressionTrackingActive"] = False
    parameters["LipTrackingActive"] = False
    return FaceOutput(parameters=parameters, native_eye=output.native_eye)
