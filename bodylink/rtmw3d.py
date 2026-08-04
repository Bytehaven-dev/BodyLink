from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from bodylink.pose import Landmark, PoseSnapshot

if TYPE_CHECKING:
    from bodylink.geometry import CameraIntrinsics


MODEL_DIRECTORY = Path(
    os.environ.get(
        "BODYLINK_MODEL_DIR",
        Path(__file__).resolve().parents[1] / "models",
    )
).resolve()
RTMW3D_MODEL_PATH = (
    MODEL_DIRECTORY / "rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7.onnx"
)
PERSON_DETECTOR_PATH = MODEL_DIRECTORY / "yolox-m-humanart.onnx"
DETECTION_INTERVAL = 7
REFERENCE_HEIGHT_M = 1.70


class Rtmw3dRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Rtmw3dRuntimeInfo:
    provider: str
    detection_interval: int
    warmup_ms: float


def missing_model_paths() -> tuple[Path, ...]:
    return tuple(
        path for path in (RTMW3D_MODEL_PATH, PERSON_DETECTOR_PATH) if not path.exists()
    )


def require_cuda_provider(providers: Sequence[str]) -> None:
    if "CUDAExecutionProvider" not in providers:
        available = ", ".join(providers) or "无"
        raise Rtmw3dRuntimeError(
            "ONNX Runtime CUDA 不可用（当前 Provider："
            f"{available}）。请重新运行 install.bat 安装 CUDA 运行库。"
        )


def _mean(values: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return np.mean(values[np.asarray(indices, dtype=np.intp)], axis=0)


def _nominal_root_depth(
    image_pixels: np.ndarray,
    relative_depth: np.ndarray,
    scores: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> float:
    sole_indices = np.array([17, 18, 19, 20, 21, 22], dtype=np.intp)
    sole_valid = sole_indices[scores[sole_indices] >= 0.15]
    if sole_valid.size == 0:
        sole_valid = np.array([15, 16], dtype=np.intp)

    ray_y = (image_pixels[:, 1] - intrinsics.center_y) / intrinsics.focal_y
    sole_ray = float(np.mean(ray_y[sole_valid]))
    sole_depth = float(np.mean(relative_depth[sole_valid]))
    estimates: list[float] = []
    anchors = (
        (np.array([1, 2], dtype=np.intp), 0.936, 0.15),
        (np.array([5, 6], dtype=np.intp), 0.818, 0.20),
        (np.array([11, 12], dtype=np.intp), 0.530, 0.25),
    )
    for indices, height_ratio, threshold in anchors:
        valid = indices[scores[indices] >= threshold]
        if valid.size == 0:
            continue
        anchor_ray = float(np.mean(ray_y[valid]))
        denominator = sole_ray - anchor_ray
        if denominator <= 0.02:
            continue
        anchor_depth = float(np.mean(relative_depth[valid]))
        depth_term = sole_ray * sole_depth - anchor_ray * anchor_depth
        estimate = (REFERENCE_HEIGHT_M * height_ratio - depth_term) / denominator
        if np.isfinite(estimate) and 0.7 <= estimate <= 10.0:
            estimates.append(float(estimate))
    return float(np.median(estimates)) if estimates else 3.0


def _estimate_height(points: np.ndarray, scores: np.ndarray) -> float | None:
    sole = _mean(points, [17, 18, 19, 20, 21, 22])
    estimates: list[float] = []
    if float(np.mean(scores[[1, 2]])) >= 0.15:
        estimates.append(float((sole[1] - _mean(points, [1, 2])[1]) / 0.936))
    if float(np.mean(scores[[5, 6]])) >= 0.15:
        estimates.append(float((sole[1] - _mean(points, [5, 6])[1]) / 0.818))
    if float(np.mean(scores[[11, 12]])) >= 0.15:
        estimates.append(float((sole[1] - _mean(points, [11, 12])[1]) / 0.530))
    usable = [value for value in estimates if np.isfinite(value) and 0.5 <= value <= 3.0]
    return float(np.median(usable)) if usable else None


def _camera_local_points(
    keypoints_3d: np.ndarray,
    keypoints_2d: np.ndarray,
    scores: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    relative_depth = np.nan_to_num(
        np.asarray(keypoints_3d[:, 2], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    pelvis_depth = float(np.mean(relative_depth[[11, 12]]))
    relative_depth -= pelvis_depth
    root_depth = _nominal_root_depth(
        keypoints_2d, relative_depth, scores, intrinsics
    )
    absolute_depth = root_depth + relative_depth
    minimum_depth = float(np.min(absolute_depth))
    if minimum_depth < 0.25:
        absolute_depth += 0.25 - minimum_depth

    ray_x = (keypoints_2d[:, 0] - intrinsics.center_x) / intrinsics.focal_x
    ray_y = (keypoints_2d[:, 1] - intrinsics.center_y) / intrinsics.focal_y
    camera_points = np.column_stack(
        (ray_x * absolute_depth, ray_y * absolute_depth, absolute_depth)
    )
    pelvis = _mean(camera_points, [11, 12])
    local = camera_points - pelvis

    estimated_height = _estimate_height(local, scores)
    if estimated_height is not None:
        local *= float(np.clip(REFERENCE_HEIGHT_M / estimated_height, 0.55, 1.80))
    return local


def _select_person(
    keypoints_3d: np.ndarray,
    keypoints_2d: np.ndarray,
    scores: np.ndarray,
    frame_shape: tuple[int, ...],
) -> int | None:
    if keypoints_3d.ndim != 3 or keypoints_2d.ndim != 3 or scores.ndim != 2:
        return None
    people = min(len(keypoints_3d), len(keypoints_2d), len(scores))
    if people == 0:
        return None

    frame_area = max(1.0, float(frame_shape[0] * frame_shape[1]))
    best_index: int | None = None
    best_rank = -1.0
    for index in range(people):
        body_scores = np.clip(scores[index, :23], 0.0, 1.0)
        confidence = float(np.mean(body_scores))
        if confidence < 0.05:
            continue
        body = keypoints_2d[index, :23]
        if not np.isfinite(body).all():
            continue
        span = np.ptp(body, axis=0)
        area = max(0.0, float(span[0] * span[1])) / frame_area
        rank = area * (0.5 + confidence)
        if rank > best_rank:
            best_rank = rank
            best_index = index
    return best_index


def snapshot_from_rtmw3d(
    keypoints_3d: np.ndarray,
    keypoints_2d: np.ndarray,
    scores: np.ndarray,
    frame_shape: tuple[int, ...],
    timestamp_s: float,
    intrinsics: CameraIntrinsics,
) -> PoseSnapshot | None:
    keypoints_3d = np.asarray(keypoints_3d, dtype=np.float64)
    keypoints_2d = np.asarray(keypoints_2d, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    person = _select_person(keypoints_3d, keypoints_2d, scores, frame_shape)
    if person is None:
        return None

    source_3d = keypoints_3d[person]
    source_2d = keypoints_2d[person]
    source_scores = np.clip(scores[person], 0.0, 1.0)
    if (
        source_3d.shape != (133, 3)
        or source_2d.shape != (133, 2)
        or source_scores.shape != (133,)
        or not np.isfinite(source_2d).all()
    ):
        return None

    source_world = _camera_local_points(
        source_3d, source_2d, source_scores, intrinsics
    )
    height, width = frame_shape[:2]
    image_points = np.zeros((33, 3), dtype=np.float64)
    world_points = np.zeros((33, 3), dtype=np.float64)
    confidence = np.zeros(33, dtype=np.float64)

    def assign(target: Landmark, source: int) -> None:
        image_points[target, :2] = source_2d[source] / np.array([width, height])
        image_points[target, 2] = source_3d[source, 2]
        world_points[target] = source_world[source]
        confidence[target] = source_scores[source]

    def assign_average(target: Landmark, sources: Sequence[int]) -> None:
        indices = np.asarray(sources, dtype=np.intp)
        image_points[target, :2] = np.mean(source_2d[indices], axis=0) / np.array(
            [width, height]
        )
        image_points[target, 2] = float(np.mean(source_3d[indices, 2]))
        world_points[target] = np.mean(source_world[indices], axis=0)
        confidence[target] = float(np.mean(source_scores[indices]))

    direct_mapping = {
        Landmark.NOSE: 0,
        Landmark.LEFT_EYE_INNER: 1,
        Landmark.LEFT_EYE: 1,
        Landmark.LEFT_EYE_OUTER: 1,
        Landmark.RIGHT_EYE_INNER: 2,
        Landmark.RIGHT_EYE: 2,
        Landmark.RIGHT_EYE_OUTER: 2,
        Landmark.LEFT_EAR: 3,
        Landmark.RIGHT_EAR: 4,
        Landmark.MOUTH_LEFT: 71,
        Landmark.MOUTH_RIGHT: 77,
        Landmark.LEFT_SHOULDER: 5,
        Landmark.RIGHT_SHOULDER: 6,
        Landmark.LEFT_ELBOW: 7,
        Landmark.RIGHT_ELBOW: 8,
        Landmark.LEFT_WRIST: 9,
        Landmark.RIGHT_WRIST: 10,
        Landmark.LEFT_PINKY: 111,
        Landmark.RIGHT_PINKY: 132,
        Landmark.LEFT_INDEX: 99,
        Landmark.RIGHT_INDEX: 120,
        Landmark.LEFT_THUMB: 95,
        Landmark.RIGHT_THUMB: 116,
        Landmark.LEFT_HIP: 11,
        Landmark.RIGHT_HIP: 12,
        Landmark.LEFT_KNEE: 13,
        Landmark.RIGHT_KNEE: 14,
        Landmark.LEFT_ANKLE: 15,
        Landmark.RIGHT_ANKLE: 16,
        Landmark.LEFT_HEEL: 19,
        Landmark.RIGHT_HEEL: 22,
    }
    for target, source in direct_mapping.items():
        assign(target, source)
    assign_average(Landmark.LEFT_FOOT_INDEX, [17, 18])
    assign_average(Landmark.RIGHT_FOOT_INDEX, [20, 21])

    if not np.isfinite(world_points).all() or not np.isfinite(image_points).all():
        return None
    return PoseSnapshot(
        image_points=image_points,
        world_points=world_points,
        visibility=confidence.copy(),
        presence=confidence.copy(),
        timestamp_s=timestamp_s,
    )


class Rtmw3dTracker:
    def __init__(
        self,
        pose_model_path: Path = RTMW3D_MODEL_PATH,
        detector_model_path: Path = PERSON_DETECTOR_PATH,
        detection_interval: int = DETECTION_INTERVAL,
    ) -> None:
        missing = tuple(
            path for path in (pose_model_path, detector_model_path) if not path.exists()
        )
        if missing:
            names = "、".join(path.name for path in missing)
            raise Rtmw3dRuntimeError(f"缺少 RTMW3D 模型：{names}，请运行 install.bat")

        try:
            import onnxruntime as ort

            ort.set_default_logger_severity(3)
            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls(directory="")
                except TypeError:
                    ort.preload_dlls()
            require_cuda_provider(ort.get_available_providers())

            from rtmlib import Wholebody3d

            model = Wholebody3d(
                det=str(detector_model_path),
                pose=str(pose_model_path),
                backend="onnxruntime",
                device="cuda",
            )
            sessions = (model.det_model.session, model.pose_model.session)
            for session in sessions:
                providers = session.get_providers()
                require_cuda_provider(providers)
                if not providers or providers[0] != "CUDAExecutionProvider":
                    raise Rtmw3dRuntimeError(
                        "RTMW3D 会话没有优先使用 CUDA，已停止以避免 CPU 回退。"
                    )

            warmup_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            warmup_started = time.perf_counter()
            model.det_model(warmup_frame)
            model.pose_model(
                warmup_frame,
                bboxes=np.array([[0.0, 0.0, 1280.0, 720.0]], dtype=np.float64),
            )
            warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
        except Rtmw3dRuntimeError:
            raise
        except Exception as exc:
            raise Rtmw3dRuntimeError(f"RTMW3D CUDA 初始化失败：{exc}") from exc

        self._model = model
        self._detection_interval = max(1, int(detection_interval))
        self._frame_count = 0
        self._detection_misses = 0
        self._bbox: np.ndarray | None = None
        self.runtime_info = Rtmw3dRuntimeInfo(
            provider="CUDAExecutionProvider",
            detection_interval=self._detection_interval,
            warmup_ms=warmup_ms,
        )

    @staticmethod
    def _largest_detection(
        bboxes: object, frame_shape: tuple[int, ...]
    ) -> np.ndarray | None:
        values = np.asarray(bboxes, dtype=np.float64)
        if values.size == 0:
            return None
        try:
            values = values.reshape(-1, 4)
        except ValueError:
            return None
        height, width = frame_shape[:2]
        values[:, [0, 2]] = np.clip(values[:, [0, 2]], 0.0, float(width))
        values[:, [1, 3]] = np.clip(values[:, [1, 3]], 0.0, float(height))
        areas = np.maximum(0.0, values[:, 2] - values[:, 0]) * np.maximum(
            0.0, values[:, 3] - values[:, 1]
        )
        index = int(np.argmax(areas))
        if areas[index] < 1000.0:
            return None
        return values[index : index + 1].copy()

    @staticmethod
    def _bbox_from_pose(
        keypoints_2d: np.ndarray,
        scores: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        points = np.asarray(keypoints_2d[:23], dtype=np.float64)
        confidence = np.asarray(scores[:23], dtype=np.float64)
        height, width = frame_shape[:2]
        valid = (
            np.isfinite(points).all(axis=1)
            & (confidence >= 0.10)
            & (points[:, 0] >= -0.1 * width)
            & (points[:, 0] <= 1.1 * width)
            & (points[:, 1] >= -0.1 * height)
            & (points[:, 1] <= 1.1 * height)
        )
        if int(np.count_nonzero(valid)) < 5:
            return None
        minimum = np.min(points[valid], axis=0)
        maximum = np.max(points[valid], axis=0)
        center = (minimum + maximum) * 0.5
        half_size = np.maximum((maximum - minimum) * 0.70, np.array([40.0, 60.0]))
        bbox = np.concatenate((center - half_size, center + half_size))
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, float(width))
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, float(height))
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 1000.0:
            return None
        return bbox[None, :]

    def infer(
        self,
        frame_bgr: np.ndarray,
        timestamp_s: float,
        intrinsics: CameraIntrinsics,
    ) -> PoseSnapshot | None:
        if self._model is None:
            raise Rtmw3dRuntimeError("RTMW3D 追踪器已经关闭")
        try:
            if self._frame_count % self._detection_interval == 0:
                detected = self._largest_detection(
                    self._model.det_model(frame_bgr), frame_bgr.shape
                )
                if detected is not None:
                    self._bbox = detected
                    self._detection_misses = 0
                elif self._bbox is not None and self._detection_misses < 1:
                    self._detection_misses += 1
                else:
                    self._bbox = None
            self._frame_count += 1
            if self._bbox is None:
                return None
            output = self._model.pose_model(frame_bgr, bboxes=self._bbox)
        except Exception as exc:
            raise Rtmw3dRuntimeError(f"RTMW3D 推理失败：{exc}") from exc
        if not isinstance(output, tuple) or len(output) != 4:
            raise Rtmw3dRuntimeError("RTMW3D 返回了无效的推理结果")
        keypoints_3d, scores, _, keypoints_2d = output
        snapshot = snapshot_from_rtmw3d(
            keypoints_3d,
            keypoints_2d,
            scores,
            frame_bgr.shape,
            timestamp_s,
            intrinsics,
        )
        if snapshot is not None and len(keypoints_2d) > 0 and len(scores) > 0:
            tracked_bbox = self._bbox_from_pose(
                np.asarray(keypoints_2d)[0], np.asarray(scores)[0], frame_bgr.shape
            )
            if tracked_bbox is not None:
                self._bbox = tracked_bbox
        return snapshot

    def close(self) -> None:
        self._model = None
        self._bbox = None
        gc.collect()
