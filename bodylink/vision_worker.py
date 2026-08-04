from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from bodylink.config import AppConfig
from bodylink.face import (
    BlendshapeSmoother,
    FaceSnapshot,
    face_output,
    snapshot_from_mediapipe as face_snapshot_from_mediapipe,
)
from bodylink.geometry import (
    CALIBRATION_LANDMARKS,
    CameraIntrinsics,
    Calibration,
    CalibrationError,
    TrackingStabilizer,
    align_calibration_to_vr,
    calibrate_pose,
    calibration_pose_ready,
    trackers_from_pose,
)
from bodylink.osc_sender import VRChatOscSender
from bodylink.pose import Landmark, POSE_CONNECTIONS, PoseSnapshot
from bodylink.rtmw3d import Rtmw3dTracker, missing_model_paths
from bodylink.vr_runtime import OpenVrPoseProvider, VrPoseSnapshot


FACE_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"
CALIBRATION_SAMPLE_COUNT = 24
FACE_LOSS_TIMEOUT_S = 0.50
CAPTURE_FOURCC = {"mjpg": "MJPG", "yuy2": "YUY2"}
FACE_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for group in (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
        vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
    )
    for connection in group
)


def _capture_backend() -> int:
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def fourcc_name(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "AUTO"
    raw = bytes((code >> (8 * index)) & 0xFF for index in range(4))
    name = raw.decode("ascii", errors="replace").strip("\x00 ")
    return name or "AUTO"


def requested_fourcc_name(camera_format: str) -> str:
    return CAPTURE_FOURCC.get(camera_format, "AUTO")


def directshow_camera_names() -> tuple[str, ...]:
    if os.name != "nt":
        return ()
    try:
        from pygrabber.dshow_graph import DeviceCategories, SystemDeviceEnum

        names = SystemDeviceEnum().get_available_filters(
            DeviceCategories.VideoInputDevice
        )
    except Exception:
        return ()
    return tuple(str(name).strip() for name in names)


@dataclass(frozen=True, slots=True)
class CameraDevice:
    index: int
    name: str

    @property
    def label(self) -> str:
        name = self.name.strip() or f"摄像头 {self.index}"
        return f"{name}  [{self.index}]"


def _open_capture(config: AppConfig) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(config.camera_index, _capture_backend())
    if not capture.isOpened() and _capture_backend() != cv2.CAP_ANY:
        capture.release()
        capture = cv2.VideoCapture(config.camera_index)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开摄像头 {config.camera_index}")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    capture.set(cv2.CAP_PROP_FPS, config.camera_fps)
    requested_fourcc = requested_fourcc_name(config.camera_format)
    if requested_fourcc != "AUTO":
        # DirectShow may select a new media type when size or FPS changes.
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*requested_fourcc),
        )
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _anchor_indices(mode: str) -> tuple[int, ...]:
    stable = (Landmark.LEFT_HIP, Landmark.RIGHT_HIP, Landmark.LEFT_ANKLE, Landmark.RIGHT_ANKLE)
    if mode == "stable":
        return stable
    return stable + (
        Landmark.LEFT_SHOULDER,
        Landmark.RIGHT_SHOULDER,
        Landmark.LEFT_KNEE,
        Landmark.RIGHT_KNEE,
        Landmark.LEFT_ELBOW,
        Landmark.RIGHT_ELBOW,
    )


def annotate_frame(
    frame_bgr: np.ndarray,
    snapshot: PoseSnapshot | None,
    mirror: bool,
    mode: str,
    face_snapshot: FaceSnapshot | None = None,
) -> np.ndarray:
    frame = cv2.flip(frame_bgr, 1) if mirror else frame_bgr.copy()
    height, width = frame.shape[:2]

    if snapshot is not None:
        confidence = snapshot.confidence

        def pose_point(index: int) -> tuple[int, int]:
            x = float(snapshot.image_points[index, 0])
            if mirror:
                x = 1.0 - x
            y = float(snapshot.image_points[index, 1])
            return int(x * width), int(y * height)

        overlay = frame.copy()
        for start, end in POSE_CONNECTIONS:
            if min(confidence[start], confidence[end]) < 0.35:
                continue
            cv2.line(
                overlay, pose_point(start), pose_point(end), (125, 224, 185), 7, cv2.LINE_AA
            )
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0.0, frame)

        for start, end in POSE_CONNECTIONS:
            if min(confidence[start], confidence[end]) < 0.35:
                continue
            cv2.line(
                frame, pose_point(start), pose_point(end), (105, 220, 174), 2, cv2.LINE_AA
            )

        visible_joints = {index for connection in POSE_CONNECTIONS for index in connection}
        for index in visible_joints:
            if confidence[index] >= 0.35:
                cv2.circle(frame, pose_point(index), 4, (242, 246, 247), -1, cv2.LINE_AA)

        for index in _anchor_indices(mode):
            if confidence[index] >= 0.35:
                cv2.circle(frame, pose_point(index), 9, (67, 188, 244), 2, cv2.LINE_AA)

    if face_snapshot is not None:
        landmarks = face_snapshot.landmarks

        def face_point(index: int) -> tuple[int, int]:
            x = float(landmarks[index, 0])
            if mirror:
                x = 1.0 - x
            return int(x * width), int(float(landmarks[index, 1]) * height)

        for start, end in FACE_CONNECTIONS:
            if start >= len(landmarks) or end >= len(landmarks):
                continue
            cv2.line(frame, face_point(start), face_point(end), (244, 190, 82), 1, cv2.LINE_AA)
    return frame


def to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()


class CameraProbeThread(QThread):
    cameras_found = Signal(object)

    def __init__(self, max_index: int = 5) -> None:
        super().__init__()
        self.max_index = max_index

    def run(self) -> None:
        found: list[CameraDevice] = []
        names = directshow_camera_names()
        if names:
            for index, name in enumerate(names[: self.max_index + 1]):
                if self.isInterruptionRequested():
                    break
                found.append(CameraDevice(index=index, name=name))
            self.cameras_found.emit(found)
            return

        backend = _capture_backend()
        for index in range(self.max_index + 1):
            if self.isInterruptionRequested():
                break
            capture = cv2.VideoCapture(index, backend)
            if capture.isOpened():
                ok, _ = capture.read()
                if ok:
                    found.append(CameraDevice(index=index, name=""))
            capture.release()
        self.cameras_found.emit(found)


class TrackingWorker(QThread):
    frame_ready = Signal(QImage)
    metrics_ready = Signal(object)
    camera_ready = Signal(object)
    pose_state = Signal(str, str)
    face_state = Signal(str, str)
    vr_state = Signal(str, str)
    calibration_progress = Signal(int)
    calibration_succeeded = Signal(object)
    calibration_failed = Signal(str)
    runtime_error = Signal(str)
    osc_error = Signal(str)
    worker_stopped = Signal()

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._config = replace(config).normalized()
        self._stop_requested = False
        self._send_enabled = False
        self._calibration_requested = False
        self._invalidate_requested = False
        self._alignment_pending = False

    def update_config(self, config: AppConfig) -> None:
        with self._lock:
            previous = self._config
            self._config = replace(config).normalized()
            if (
                previous.user_height_m != self._config.user_height_m
                or previous.horizontal_fov_deg != self._config.horizontal_fov_deg
                or previous.vr_assist_enabled != self._config.vr_assist_enabled
            ):
                self._invalidate_requested = True
                self._send_enabled = False

    def request_calibration(self) -> None:
        with self._lock:
            self._calibration_requested = True
            self._send_enabled = False

    def set_sending(self, enabled: bool) -> None:
        with self._lock:
            self._send_enabled = bool(enabled)

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def _control_snapshot(self) -> tuple[AppConfig, bool, bool, bool, bool]:
        with self._lock:
            config = replace(self._config)
            result = (
                config,
                self._stop_requested,
                self._send_enabled,
                self._calibration_requested,
                self._invalidate_requested,
            )
            self._calibration_requested = False
            self._invalidate_requested = False
            return result

    def run(self) -> None:
        capture: cv2.VideoCapture | None = None
        body_tracker: Rtmw3dTracker | None = None
        face_landmarker: vision.FaceLandmarker | None = None
        vr_provider: OpenVrPoseProvider | None = None
        sender: VRChatOscSender | None = None
        face_output_active = False
        face_output_native_eyes = True
        try:
            missing = missing_model_paths()
            if missing:
                names = "、".join(path.name for path in missing)
                raise RuntimeError(f"缺少 RTMW3D 模型：{names}，请先运行 install.bat")

            config, _, _, _, _ = self._control_snapshot()
            body_tracker = Rtmw3dTracker()
            capture = _open_capture(config)
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.camera_width
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.camera_height
            actual_fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(config.camera_fps)
            actual_format = fourcc_name(capture.get(cv2.CAP_PROP_FOURCC))
            self.camera_ready.emit(
                {
                    "width": actual_width,
                    "height": actual_height,
                    "fps": actual_fps,
                    "format": actual_format,
                    "requested_format": requested_fourcc_name(config.camera_format),
                    "camera_index": config.camera_index,
                    "backend": "RTMW3D / CUDA",
                }
            )
            sender = VRChatOscSender("127.0.0.1", 9000)
            stabilizer = TrackingStabilizer()
            calibration: Calibration | None = None
            calibration_samples: list[PoseSnapshot] = []
            calibration_vr_samples: list[VrPoseSnapshot | None] = []
            collecting_calibration = False
            calibration_started_s = 0.0
            last_timestamp_ms = 0
            read_failures = 0
            fps_ema = 0.0
            previous_frame_s = time.perf_counter()
            last_metrics_s = 0.0
            last_pose_state = ""
            last_reprojection = 0.0
            active_trackers = 0
            face_smoother = BlendshapeSmoother()
            face_snapshot: FaceSnapshot | None = None
            next_face_inference_s = 0.0
            last_face_seen_s = 0.0
            face_inference_ms = 0.0
            last_face_state = ""
            previous_face_enabled = False
            face_initialization_failed = False
            last_vr_state = ""

            def update_vr_state(state: str, text: str) -> None:
                nonlocal last_vr_state
                key = f"{state}:{text}"
                if key != last_vr_state:
                    self.vr_state.emit(state, text)
                    last_vr_state = key

            def update_face_state(state: str, text: str) -> None:
                nonlocal last_face_state
                if state != last_face_state:
                    self.face_state.emit(state, text)
                    last_face_state = state

            def osc_failed(exc: Exception) -> None:
                with self._lock:
                    self._send_enabled = False
                self.osc_error.emit(str(exc))

            def reset_face_output() -> None:
                nonlocal face_output_active
                if not face_output_active:
                    return
                try:
                    sender.send_face_reset(face_output_native_eyes)
                except (OSError, ValueError) as exc:
                    osc_failed(exc)
                face_output_active = False

            while True:
                config, stop_requested, send_enabled, calibrate_requested, invalidate = (
                    self._control_snapshot()
                )
                if stop_requested:
                    reset_face_output()
                    break

                if config.vr_assist_enabled and vr_provider is None:
                    vr_provider = OpenVrPoseProvider()
                    vr_provider.start()
                    update_vr_state("loading", "等待 SteamVR / Pico")
                elif not config.vr_assist_enabled and vr_provider is not None:
                    vr_provider.stop()
                    vr_provider = None
                    update_vr_state("disabled", "VR 辅助未启用")
                elif not config.vr_assist_enabled:
                    update_vr_state("disabled", "VR 辅助未启用")

                if config.face_enabled != previous_face_enabled:
                    face_initialization_failed = False
                    if config.face_enabled:
                        update_face_state("loading", "正在加载面捕")
                    previous_face_enabled = config.face_enabled

                if not config.face_enabled:
                    reset_face_output()
                    if face_landmarker is not None:
                        face_landmarker.close()
                        face_landmarker = None
                    face_snapshot = None
                    last_face_seen_s = 0.0
                    face_smoother.reset()
                    update_face_state("disabled", "面捕未启用")
                elif face_landmarker is None and not face_initialization_failed:
                    update_face_state("loading", "正在加载面捕")
                    try:
                        if not FACE_MODEL_PATH.exists():
                            raise RuntimeError("缺少面捕模型，请重新运行 install.bat")
                        face_options = vision.FaceLandmarkerOptions(
                            base_options=mp_python.BaseOptions(
                                model_asset_path=str(FACE_MODEL_PATH),
                                delegate=mp_python.BaseOptions.Delegate.CPU,
                            ),
                            running_mode=vision.RunningMode.VIDEO,
                            num_faces=1,
                            min_face_detection_confidence=0.50,
                            min_face_presence_confidence=0.50,
                            min_tracking_confidence=0.50,
                            output_face_blendshapes=True,
                            output_facial_transformation_matrixes=False,
                        )
                        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
                    except Exception as exc:
                        face_initialization_failed = True
                        update_face_state("error", f"面捕错误：{exc}")
                    else:
                        next_face_inference_s = 0.0
                        update_face_state("lost", "等待人脸入镜")

                if face_output_active and face_output_native_eyes != config.face_native_eyes:
                    reset_face_output()
                if not send_enabled:
                    reset_face_output()
                if invalidate:
                    calibration = None
                    collecting_calibration = False
                    calibration_samples.clear()
                    calibration_vr_samples.clear()
                    stabilizer.reset()
                if calibrate_requested:
                    calibration = None
                    collecting_calibration = True
                    calibration_samples.clear()
                    calibration_vr_samples.clear()
                    stabilizer.reset()
                    calibration_started_s = time.perf_counter()
                    self.calibration_progress.emit(0)

                ok, frame = capture.read()
                if not ok:
                    read_failures += 1
                    if read_failures >= 20:
                        raise RuntimeError("摄像头连续读取失败，请关闭占用摄像头的应用")
                    self.msleep(15)
                    continue
                read_failures = 0

                frame_time_s = time.perf_counter()
                vr_pose = (
                    vr_provider.sample(frame_time_s)
                    if config.vr_assist_enabled and vr_provider is not None
                    else None
                )
                if vr_provider is not None:
                    status = vr_provider.status
                    if status.state == "ready":
                        update_vr_state("ready", "VR 头显与双手柄已连接")
                    elif status.state == "partial":
                        update_vr_state(
                            "partial",
                            f"VR 已连接 · 手柄 {status.controller_count} / 2",
                        )
                    else:
                        update_vr_state("loading", "等待 SteamVR / Pico")
                timestamp_ms = max(last_timestamp_ms + 1, int(frame_time_s * 1000))
                last_timestamp_ms = timestamp_ms
                intrinsics = CameraIntrinsics(
                    width=frame.shape[1],
                    height=frame.shape[0],
                    horizontal_fov_deg=config.horizontal_fov_deg,
                )
                inference_start = time.perf_counter()
                snapshot = body_tracker.infer(frame, frame_time_s, intrinsics)
                inference_ms = (time.perf_counter() - inference_start) * 1000.0

                if collecting_calibration and snapshot is not None:
                    if calibration_pose_ready(snapshot):
                        calibration_samples.append(snapshot)
                        calibration_vr_samples.append(vr_pose)
                        progress = min(
                            100,
                            int(len(calibration_samples) * 100 / CALIBRATION_SAMPLE_COUNT),
                        )
                        self.calibration_progress.emit(progress)
                    if len(calibration_samples) >= CALIBRATION_SAMPLE_COUNT:
                        collecting_calibration = False
                        try:
                            calibration = calibrate_pose(
                                calibration_samples,
                                intrinsics,
                                config.user_height_m,
                            )
                        except CalibrationError as exc:
                            calibration = None
                            self.calibration_failed.emit(str(exc))
                        else:
                            vr_error = ""
                            vr_alignment_error_m = 0.0
                            if config.vr_assist_enabled:
                                try:
                                    calibration, vr_info = align_calibration_to_vr(
                                        calibration,
                                        calibration_samples,
                                        calibration_vr_samples,
                                        intrinsics,
                                    )
                                except CalibrationError as exc:
                                    vr_error = str(exc)
                                else:
                                    vr_alignment_error_m = vr_info.horizontal_error_m
                            self._alignment_pending = (
                                config.align_yaw_on_calibrate and not calibration.vr_assisted
                            )
                            self.calibration_succeeded.emit(
                                {
                                    "scale": calibration.scale,
                                    "error_px": calibration.reprojection_error_px,
                                    "vr_assisted": calibration.vr_assisted,
                                    "vr_error": vr_error,
                                    "vr_alignment_error_m": vr_alignment_error_m,
                                }
                            )
                if collecting_calibration and frame_time_s - calibration_started_s > 8.0:
                    collecting_calibration = False
                    calibration_samples.clear()
                    calibration_vr_samples.clear()
                    self.calibration_failed.emit("未能收集到肩部至双脚姿态，请退后并改善光线")

                trackers = []
                if snapshot is not None and calibration is not None:
                    try:
                        raw_trackers, reconstruction = trackers_from_pose(
                            snapshot,
                            calibration,
                            intrinsics,
                            config.tracker_mode,
                            vr_pose if calibration.vr_assisted else None,
                        )
                        last_reprojection = reconstruction.median_error_px
                    except CalibrationError:
                        raw_trackers = []
                    trackers = stabilizer.update(
                        raw_trackers,
                        frame_time_s,
                        config.smoothing,
                        config.min_confidence,
                    )
                    active_trackers = len([tracker for tracker in trackers if not tracker.stale])
                elif calibration is not None:
                    trackers = stabilizer.update(
                        [], frame_time_s, config.smoothing, config.min_confidence
                    )
                    active_trackers = 0
                else:
                    active_trackers = 0

                if send_enabled and calibration is not None and trackers:
                    try:
                        sender.configure(config.target_host, config.target_port)
                        if calibration.vr_assisted and vr_pose is not None and vr_pose.hmd is not None:
                            sender.send_trackers(
                                trackers,
                                head_position_m=vr_pose.hmd.position_m,
                                head_yaw_deg=vr_pose.hmd.yaw_deg,
                            )
                        else:
                            if self._alignment_pending:
                                sender.send_yaw_alignment()
                                self._alignment_pending = False
                            sender.send_trackers(trackers)
                    except (OSError, ValueError) as exc:
                        osc_failed(exc)
                        send_enabled = False

                if snapshot is None:
                    state_key, state_text = "lost", "未检测到人体"
                    pose_score = 0.0
                else:
                    pose_score = snapshot.score(CALIBRATION_LANDMARKS)
                    if not calibration_pose_ready(snapshot):
                        state_key, state_text = "partial", "请让肩部至双脚完整入镜"
                    else:
                        state_key, state_text = "ready", "肩部至双脚已识别"
                if state_key != last_pose_state:
                    self.pose_state.emit(state_key, state_text)
                    last_pose_state = state_key

                face_period_s = 1.0 / config.face_fps
                if (
                    config.face_enabled
                    and face_landmarker is not None
                    and (
                        next_face_inference_s == 0.0
                        or frame_time_s >= next_face_inference_s
                    )
                ):
                    if next_face_inference_s == 0.0:
                        next_face_inference_s = frame_time_s
                    next_face_inference_s += face_period_s
                    if next_face_inference_s <= frame_time_s:
                        next_face_inference_s = frame_time_s + face_period_s
                    face_inference_start = time.perf_counter()
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb),
                        )
                        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                    except Exception as exc:
                        face_inference_ms = 0.0
                        reset_face_output()
                        face_landmarker.close()
                        face_landmarker = None
                        face_snapshot = None
                        last_face_seen_s = 0.0
                        face_smoother.reset()
                        face_initialization_failed = True
                        update_face_state("error", f"面捕错误：{exc}")
                    else:
                        face_inference_ms = (
                            time.perf_counter() - face_inference_start
                        ) * 1000.0
                        detected_face = face_snapshot_from_mediapipe(
                            face_result, frame_time_s
                        )
                        if detected_face is not None:
                            smoothed = face_smoother.update(
                                detected_face.blendshapes, config.face_smoothing
                            )
                            face_snapshot = replace(detected_face, blendshapes=smoothed)
                            last_face_seen_s = frame_time_s
                            update_face_state("ready", "人脸已识别")
                            if send_enabled:
                                try:
                                    sender.configure(config.target_host, config.target_port)
                                    sender.send_face(
                                        face_output(smoothed, config.face_native_eyes)
                                    )
                                except (OSError, ValueError) as exc:
                                    osc_failed(exc)
                                    send_enabled = False
                                    face_output_active = False
                                else:
                                    face_output_active = True
                                    face_output_native_eyes = config.face_native_eyes
                        elif last_face_seen_s == 0.0:
                            update_face_state("lost", "未检测到人脸")

                if (
                    config.face_enabled
                    and face_landmarker is not None
                    and last_face_seen_s > 0.0
                    and frame_time_s - last_face_seen_s > FACE_LOSS_TIMEOUT_S
                ):
                    face_snapshot = None
                    last_face_seen_s = 0.0
                    face_smoother.reset()
                    reset_face_output()
                    update_face_state("lost", "人脸已离开画面")

                annotated = annotate_frame(
                    frame,
                    snapshot,
                    config.mirror_preview,
                    config.tracker_mode,
                    face_snapshot,
                )
                self.frame_ready.emit(to_qimage(annotated))

                frame_delta = max(1e-4, frame_time_s - previous_frame_s)
                instantaneous_fps = 1.0 / frame_delta
                fps_ema = instantaneous_fps if fps_ema == 0.0 else fps_ema * 0.90 + instantaneous_fps * 0.10
                previous_frame_s = frame_time_s

                if frame_time_s - last_metrics_s >= 0.25:
                    self.metrics_ready.emit(
                        {
                            "fps": fps_ema,
                            "inference_ms": inference_ms,
                            "pose_score": pose_score,
                            "tracker_count": active_trackers,
                            "packets": sender.stats.packets_sent,
                            "calibrated": calibration is not None,
                            "sending": send_enabled and calibration is not None,
                            "reprojection_px": last_reprojection,
                            "face_inference_ms": face_inference_ms,
                            "face_enabled": config.face_enabled,
                            "face_detected": face_snapshot is not None,
                            "body_provider": body_tracker.runtime_info.provider,
                            "vr_assisted": bool(calibration is not None and calibration.vr_assisted),
                        }
                    )
                    last_metrics_s = frame_time_s

        except Exception as exc:
            self.runtime_error.emit(str(exc))
        finally:
            if sender is not None and face_output_active:
                try:
                    sender.send_face_reset(face_output_native_eyes)
                except (OSError, ValueError):
                    pass
            if face_landmarker is not None:
                face_landmarker.close()
            if body_tracker is not None:
                body_tracker.close()
            if vr_provider is not None:
                vr_provider.stop()
            if sender is not None:
                sender.close()
            if capture is not None:
                capture.release()
            self.worker_stopped.emit()
