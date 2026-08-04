from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from math import atan2, degrees
from typing import Any

import numpy as np


OPENVR_TO_VRCHAT = np.diag([1.0, 1.0, -1.0])


@dataclass(frozen=True, slots=True)
class TrackedPose:
    position_m: np.ndarray
    rotation: np.ndarray
    timestamp_s: float
    device_name: str = ""

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError("position_m must have shape (3,)")
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        object.__setattr__(self, "position_m", position.copy())
        object.__setattr__(self, "rotation", rotation.copy())

    @property
    def yaw_deg(self) -> float:
        forward = self.rotation[:, 2]
        return degrees(atan2(float(forward[0]), float(forward[2])))


@dataclass(frozen=True, slots=True)
class VrPoseSnapshot:
    timestamp_s: float
    hmd: TrackedPose | None = None
    left_hand: TrackedPose | None = None
    right_hand: TrackedPose | None = None
    runtime_name: str = "SteamVR"

    @property
    def controller_count(self) -> int:
        return int(self.left_hand is not None) + int(self.right_hand is not None)

    @property
    def ready(self) -> bool:
        return self.hmd is not None and self.controller_count == 2


@dataclass(frozen=True, slots=True)
class VrRuntimeStatus:
    state: str
    detail: str
    hmd_name: str = ""
    controller_count: int = 0


def tracked_pose_from_openvr_matrix(
    matrix: Any,
    timestamp_s: float,
    device_name: str = "",
) -> TrackedPose:
    raw = np.array(
        [[float(matrix.m[row][column]) for column in range(4)] for row in range(3)],
        dtype=np.float64,
    )
    position = OPENVR_TO_VRCHAT @ raw[:, 3]
    rotation = OPENVR_TO_VRCHAT @ raw[:, :3] @ OPENVR_TO_VRCHAT
    return TrackedPose(position, rotation, timestamp_s, device_name)


class OpenVrPoseProvider:
    def __init__(self, poll_hz: float = 90.0, history_seconds: float = 2.0) -> None:
        self.poll_hz = max(30.0, min(240.0, float(poll_hz)))
        history_size = max(60, int(self.poll_hz * max(0.5, history_seconds)))
        self._history: deque[VrPoseSnapshot] = deque(maxlen=history_size)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = VrRuntimeStatus("waiting", "SteamVR is not connected")
        self._openvr: Any | None = None
        self._system: Any | None = None

    @property
    def status(self) -> VrRuntimeStatus:
        with self._lock:
            return self._status

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="BodyLink-OpenVR",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._disconnect()
        with self._lock:
            self._thread = None
            self._status = VrRuntimeStatus("stopped", "SteamVR input stopped")

    def sample(self, timestamp_s: float, max_age_s: float = 0.12) -> VrPoseSnapshot | None:
        with self._lock:
            if not self._history:
                return None
            closest = min(
                self._history,
                key=lambda snapshot: abs(snapshot.timestamp_s - timestamp_s),
            )
        if abs(closest.timestamp_s - timestamp_s) > max_age_s:
            return None
        return closest

    def _set_status(self, status: VrRuntimeStatus) -> None:
        with self._lock:
            self._status = status

    def _connect(self) -> bool:
        try:
            import openvr

            if not openvr.isRuntimeInstalled():
                self._set_status(VrRuntimeStatus("waiting", "SteamVR is not installed"))
                return False
            if not openvr.isHmdPresent():
                self._set_status(VrRuntimeStatus("waiting", "No VR headset is active"))
                return False
            system = openvr.init(openvr.VRApplication_Background)
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            self._set_status(VrRuntimeStatus("waiting", detail))
            return False
        self._openvr = openvr
        self._system = system
        self._set_status(VrRuntimeStatus("waiting", "Waiting for tracked devices"))
        return True

    def _disconnect(self) -> None:
        openvr = self._openvr
        self._system = None
        self._openvr = None
        if openvr is not None:
            try:
                openvr.shutdown()
            except Exception:
                pass

    def _device_name(self, device_index: int) -> str:
        try:
            return str(
                self._system.getStringTrackedDeviceProperty(
                    device_index,
                    self._openvr.Prop_ModelNumber_String,
                )
            ).strip()
        except Exception:
            return ""

    def _poll(self) -> VrPoseSnapshot:
        openvr = self._openvr
        system = self._system
        poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0.0,
            poses,
        )
        timestamp_s = time.perf_counter()
        hmd: TrackedPose | None = None
        left: TrackedPose | None = None
        right: TrackedPose | None = None

        for device_index, pose in enumerate(poses):
            if not pose.bDeviceIsConnected or not pose.bPoseIsValid:
                continue
            device_class = system.getTrackedDeviceClass(device_index)
            tracked = tracked_pose_from_openvr_matrix(
                pose.mDeviceToAbsoluteTracking,
                timestamp_s,
                self._device_name(device_index),
            )
            if device_class == openvr.TrackedDeviceClass_HMD:
                hmd = tracked
            elif device_class == openvr.TrackedDeviceClass_Controller:
                role = system.getControllerRoleForTrackedDeviceIndex(device_index)
                if role == openvr.TrackedControllerRole_LeftHand:
                    left = tracked
                elif role == openvr.TrackedControllerRole_RightHand:
                    right = tracked

        runtime_name = hmd.device_name if hmd is not None and hmd.device_name else "SteamVR"
        return VrPoseSnapshot(timestamp_s, hmd, left, right, runtime_name)

    def _record(self, snapshot: VrPoseSnapshot) -> None:
        with self._lock:
            self._history.append(snapshot)
            if snapshot.ready:
                state = "ready"
                detail = "Headset and both controllers are tracked"
            elif snapshot.hmd is not None or snapshot.controller_count:
                state = "partial"
                detail = "Waiting for headset or both controllers"
            else:
                state = "waiting"
                detail = "Waiting for tracked devices"
            self._status = VrRuntimeStatus(
                state,
                detail,
                snapshot.hmd.device_name if snapshot.hmd is not None else "",
                snapshot.controller_count,
            )

    def _run(self) -> None:
        interval_s = 1.0 / self.poll_hz
        while not self._stop_event.is_set():
            if self._system is None and not self._connect():
                self._stop_event.wait(1.5)
                continue
            try:
                self._record(self._poll())
            except Exception as exc:
                detail = str(exc).strip() or type(exc).__name__
                self._set_status(VrRuntimeStatus("waiting", detail))
                self._disconnect()
                self._stop_event.wait(1.0)
                continue
            self._stop_event.wait(interval_s)
