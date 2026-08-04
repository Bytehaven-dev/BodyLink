from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import atan2, cos, degrees, exp, radians, sin, tan
from typing import Iterable

import numpy as np

from bodylink.pose import Landmark, PoseSnapshot, median_snapshot
from bodylink.vr_runtime import VrPoseSnapshot


TRACKER_LAYOUT: dict[str, tuple[int, str]] = {
    "hip": (1, "腰"),
    "left_foot": (2, "左脚"),
    "right_foot": (3, "右脚"),
    "chest": (4, "胸"),
    "left_knee": (5, "左膝"),
    "right_knee": (6, "右膝"),
    "left_elbow": (7, "左肘"),
    "right_elbow": (8, "右肘"),
}

STABLE_ROLES = ("hip", "left_foot", "right_foot")
FULL_ROLES = tuple(TRACKER_LAYOUT)

RECONSTRUCTION_LANDMARKS = np.array(
    [
        Landmark.NOSE,
        Landmark.LEFT_SHOULDER,
        Landmark.RIGHT_SHOULDER,
        Landmark.LEFT_ELBOW,
        Landmark.RIGHT_ELBOW,
        Landmark.LEFT_WRIST,
        Landmark.RIGHT_WRIST,
        Landmark.LEFT_HIP,
        Landmark.RIGHT_HIP,
        Landmark.LEFT_KNEE,
        Landmark.RIGHT_KNEE,
        Landmark.LEFT_ANKLE,
        Landmark.RIGHT_ANKLE,
        Landmark.LEFT_HEEL,
        Landmark.RIGHT_HEEL,
        Landmark.LEFT_FOOT_INDEX,
        Landmark.RIGHT_FOOT_INDEX,
    ],
    dtype=np.intp,
)

CALIBRATION_LANDMARKS = (
    Landmark.LEFT_SHOULDER,
    Landmark.RIGHT_SHOULDER,
    Landmark.LEFT_HIP,
    Landmark.RIGHT_HIP,
    Landmark.LEFT_KNEE,
    Landmark.RIGHT_KNEE,
    Landmark.LEFT_ANKLE,
    Landmark.RIGHT_ANKLE,
    Landmark.LEFT_HEEL,
    Landmark.RIGHT_HEEL,
    Landmark.LEFT_FOOT_INDEX,
    Landmark.RIGHT_FOOT_INDEX,
)

CALIBRATION_TOP_LANDMARKS = (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER)
CALIBRATION_FOOT_LANDMARKS = (
    Landmark.LEFT_ANKLE,
    Landmark.RIGHT_ANKLE,
    Landmark.LEFT_HEEL,
    Landmark.RIGHT_HEEL,
    Landmark.LEFT_FOOT_INDEX,
    Landmark.RIGHT_FOOT_INDEX,
)


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width: int
    height: int
    horizontal_fov_deg: float

    @property
    def focal_x(self) -> float:
        return self.width / (2.0 * tan(radians(self.horizontal_fov_deg) / 2.0))

    @property
    def focal_y(self) -> float:
        return self.focal_x

    @property
    def center_x(self) -> float:
        return self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.height / 2.0


@dataclass(frozen=True, slots=True)
class ReconstructionInfo:
    translation: np.ndarray
    median_error_px: float
    used_landmarks: int


@dataclass(frozen=True, slots=True)
class BodyProportions:
    left_upper_arm_m: float
    left_forearm_m: float
    right_upper_arm_m: float
    right_forearm_m: float
    left_thigh_m: float
    left_shin_m: float
    right_thigh_m: float
    right_shin_m: float


@dataclass(frozen=True, slots=True)
class Calibration:
    scale: float
    camera_origin: np.ndarray
    yaw_correction_deg: float
    user_height_m: float
    reprojection_error_px: float
    proportions: BodyProportions
    vr_yaw_offset_deg: float = 0.0
    tracking_offset_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    left_hand_offset_local_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    right_hand_offset_local_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    vr_assisted: bool = False

    def camera_to_tracking(self, camera_points: np.ndarray) -> np.ndarray:
        relative = np.asarray(camera_points, dtype=np.float64) - self.camera_origin
        tracking = relative * np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        tracking = rotate_y(tracking, self.yaw_correction_deg)
        tracking = rotate_y(tracking, self.vr_yaw_offset_deg)
        return tracking + self.tracking_offset_m


@dataclass(frozen=True, slots=True)
class VrAlignmentInfo:
    sample_count: int
    yaw_offset_deg: float
    horizontal_error_m: float


@dataclass(slots=True)
class TrackerPose:
    role: str
    tracker_id: int
    label: str
    position_m: np.ndarray
    euler_deg: np.ndarray
    confidence: float
    stale: bool = False

    def copy(self) -> "TrackerPose":
        return TrackerPose(
            role=self.role,
            tracker_id=self.tracker_id,
            label=self.label,
            position_m=self.position_m.copy(),
            euler_deg=self.euler_deg.copy(),
            confidence=self.confidence,
            stale=self.stale,
        )


@dataclass(slots=True)
class _FilterState:
    pose: TrackerPose
    last_seen_s: float
    last_update_s: float


def rotate_y(points: np.ndarray, angle_deg: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    angle = radians(angle_deg)
    c = cos(angle)
    s = sin(angle)
    matrix = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return values @ matrix.T


def _translation_from_projection(
    snapshot: PoseSnapshot,
    intrinsics: CameraIntrinsics,
    scale: float,
) -> ReconstructionInfo:
    confidence = snapshot.confidence[RECONSTRUCTION_LANDMARKS]
    valid = confidence >= 0.30
    indices = RECONSTRUCTION_LANDMARKS[valid]
    if indices.size < 6:
        raise CalibrationError("可见的身体关键点不足")

    local = snapshot.world_points[indices] * scale
    image = snapshot.image_points[indices, :2]
    xn = (image[:, 0] * intrinsics.width - intrinsics.center_x) / intrinsics.focal_x
    yn = (image[:, 1] * intrinsics.height - intrinsics.center_y) / intrinsics.focal_y

    rows = np.zeros((indices.size * 2, 3), dtype=np.float64)
    rhs = np.zeros(indices.size * 2, dtype=np.float64)
    rows[0::2, 0] = 1.0
    rows[0::2, 2] = -xn
    rhs[0::2] = xn * local[:, 2] - local[:, 0]
    rows[1::2, 1] = 1.0
    rows[1::2, 2] = -yn
    rhs[1::2] = yn * local[:, 2] - local[:, 1]

    weights = np.repeat(np.sqrt(np.clip(confidence[valid], 0.05, 1.0)), 2)
    translation, _, rank, _ = np.linalg.lstsq(rows * weights[:, None], rhs * weights, rcond=None)
    if rank < 3 or not np.isfinite(translation).all():
        raise CalibrationError("无法计算摄像头空间")

    camera = local + translation
    if np.any(camera[:, 2] <= 0.05) or not 0.35 <= translation[2] <= 10.0:
        raise CalibrationError("摄像头距离估计无效")

    projected_x = intrinsics.focal_x * camera[:, 0] / camera[:, 2] + intrinsics.center_x
    projected_y = intrinsics.focal_y * camera[:, 1] / camera[:, 2] + intrinsics.center_y
    observed_x = image[:, 0] * intrinsics.width
    observed_y = image[:, 1] * intrinsics.height
    errors = np.hypot(projected_x - observed_x, projected_y - observed_y)
    median_error = float(np.median(errors))
    if not np.isfinite(median_error) or median_error > max(120.0, intrinsics.width * 0.12):
        raise CalibrationError("姿态空间误差过大，请正对摄像头")

    return ReconstructionInfo(
        translation=translation,
        median_error_px=median_error,
        used_landmarks=int(indices.size),
    )


def reconstruct_camera_points(
    snapshot: PoseSnapshot,
    intrinsics: CameraIntrinsics,
    scale: float,
) -> tuple[np.ndarray, ReconstructionInfo]:
    info = _translation_from_projection(snapshot, intrinsics, scale)
    return snapshot.world_points * scale + info.translation, info


def _estimate_raw_height(snapshot: PoseSnapshot) -> float:
    world = snapshot.world_points
    confidence = snapshot.confidence
    sole_y = float(
        np.mean(
            world[
                [
                    Landmark.LEFT_HEEL,
                    Landmark.RIGHT_HEEL,
                    Landmark.LEFT_FOOT_INDEX,
                    Landmark.RIGHT_FOOT_INDEX,
                ],
                1,
            ]
        )
    )
    estimates: list[float] = []

    eye_ids = [Landmark.LEFT_EYE, Landmark.RIGHT_EYE]
    if float(np.mean(confidence[eye_ids])) >= 0.35:
        eye_y = float(np.mean(world[eye_ids, 1]))
        estimates.append((sole_y - eye_y) / 0.936)

    shoulder_ids = [Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER]
    if float(np.mean(confidence[shoulder_ids])) >= 0.45:
        shoulder_y = float(np.mean(world[shoulder_ids, 1]))
        estimates.append((sole_y - shoulder_y) / 0.818)

    hip_ids = [Landmark.LEFT_HIP, Landmark.RIGHT_HIP]
    if float(np.mean(confidence[hip_ids])) >= 0.50:
        hip_y = float(np.mean(world[hip_ids, 1]))
        estimates.append((sole_y - hip_y) / 0.530)

    estimates = [value for value in estimates if 0.8 <= value <= 2.5]
    if not estimates:
        raise CalibrationError("无法估算身体比例，请让肩部或髋部与双脚同时入镜")
    return float(np.median(estimates))


def calibration_pose_ready(snapshot: PoseSnapshot) -> bool:
    if snapshot.score(CALIBRATION_LANDMARKS) < 0.45:
        return False
    if snapshot.score(CALIBRATION_TOP_LANDMARKS) < 0.45:
        return False
    if snapshot.score(CALIBRATION_FOOT_LANDMARKS) < 0.45:
        return False
    image = snapshot.image_points
    important = np.asarray(CALIBRATION_LANDMARKS, dtype=np.intp)
    return bool(
        np.min(image[important, 0]) >= 0.015
        and np.max(image[important, 0]) <= 0.985
        and np.min(image[important, 1]) >= 0.01
        and np.max(image[important, 1]) <= 0.99
    )


def _distance(points: np.ndarray, start: Landmark, end: Landmark) -> float:
    return float(np.linalg.norm(points[end] - points[start]))


def _bounded_length(value: float, fallback: float) -> float:
    return float(value if np.isfinite(value) and 0.10 <= value <= 0.65 else fallback)


def _measure_proportions(points: np.ndarray, height_m: float) -> BodyProportions:
    return BodyProportions(
        left_upper_arm_m=_bounded_length(
            _distance(points, Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
            height_m * 0.186,
        ),
        left_forearm_m=_bounded_length(
            _distance(points, Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
            height_m * 0.146,
        ),
        right_upper_arm_m=_bounded_length(
            _distance(points, Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
            height_m * 0.186,
        ),
        right_forearm_m=_bounded_length(
            _distance(points, Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
            height_m * 0.146,
        ),
        left_thigh_m=_bounded_length(
            _distance(points, Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
            height_m * 0.245,
        ),
        left_shin_m=_bounded_length(
            _distance(points, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
            height_m * 0.246,
        ),
        right_thigh_m=_bounded_length(
            _distance(points, Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
            height_m * 0.245,
        ),
        right_shin_m=_bounded_length(
            _distance(points, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
            height_m * 0.246,
        ),
    )


def _body_heading(points: np.ndarray) -> float:
    shoulder_right = points[Landmark.RIGHT_SHOULDER] - points[Landmark.LEFT_SHOULDER]
    hip_right = points[Landmark.RIGHT_HIP] - points[Landmark.LEFT_HIP]
    right = shoulder_right + hip_right
    right[1] = 0.0
    length = float(np.linalg.norm(right))
    if length < 1e-5:
        return 0.0
    right /= length
    forward = np.array([-right[2], 0.0, right[0]])
    return degrees(atan2(float(forward[0]), float(forward[2])))


def calibrate_pose(
    samples: list[PoseSnapshot],
    intrinsics: CameraIntrinsics,
    user_height_m: float,
) -> Calibration:
    if len(samples) < 8:
        raise CalibrationError("校准样本不足")

    median = median_snapshot(samples)
    if not calibration_pose_ready(median):
        raise CalibrationError("身体识别置信度不足，请让肩部至双脚完整入镜")

    image = median.image_points
    important = np.asarray(CALIBRATION_LANDMARKS, dtype=np.intp)
    if (
        np.min(image[important, 0]) < 0.015
        or np.max(image[important, 0]) > 0.985
        or np.min(image[important, 1]) < 0.01
        or np.max(image[important, 1]) > 0.99
    ):
        raise CalibrationError("身体超出画面，请退后一步")

    sample_hips = np.stack(
        [
            np.mean(sample.image_points[[Landmark.LEFT_HIP, Landmark.RIGHT_HIP], :2], axis=0)
            for sample in samples
        ]
    )
    if float(np.max(np.std(sample_hips, axis=0))) > 0.035:
        raise CalibrationError("校准时移动过多，请保持站立")

    raw_height = _estimate_raw_height(median)
    scale = float(user_height_m / raw_height)
    if not 0.45 <= scale <= 2.2:
        raise CalibrationError("身体比例估计异常，请重新校准")

    camera_points, reconstruction = reconstruct_camera_points(median, intrinsics, scale)
    sole_ids = np.array(
        [
            Landmark.LEFT_HEEL,
            Landmark.RIGHT_HEEL,
            Landmark.LEFT_FOOT_INDEX,
            Landmark.RIGHT_FOOT_INDEX,
        ],
        dtype=np.intp,
    )
    origin = np.mean(camera_points[sole_ids], axis=0)
    origin[1] = float(np.max(camera_points[sole_ids, 1]))

    base_tracking = (camera_points - origin) * np.array([-1.0, -1.0, -1.0])
    yaw_correction = -_body_heading(base_tracking)
    calibrated_tracking = rotate_y(base_tracking, yaw_correction)

    return Calibration(
        scale=scale,
        camera_origin=origin,
        yaw_correction_deg=yaw_correction,
        user_height_m=user_height_m,
        reprojection_error_px=reconstruction.median_error_px,
        proportions=_measure_proportions(calibrated_tracking, user_height_m),
    )


def _mean(points: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    return np.mean(points[np.asarray(list(indices), dtype=np.intp)], axis=0)


def _yaw_from_forward(vector: np.ndarray, fallback: float) -> float:
    horizontal = np.array([vector[0], 0.0, vector[2]], dtype=np.float64)
    if float(np.linalg.norm(horizontal)) < 0.035:
        return fallback
    return degrees(atan2(float(horizontal[0]), float(horizontal[2])))


def _right_axis_angle(vector: np.ndarray) -> float:
    horizontal = np.array([vector[0], 0.0, vector[2]], dtype=np.float64)
    length = float(np.linalg.norm(horizontal))
    if length < 1e-5:
        raise CalibrationError("双手间距不足，无法对齐 VR 空间")
    horizontal /= length
    return degrees(atan2(float(-horizontal[2]), float(horizontal[0])))


def _wrap_angle(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def align_calibration_to_vr(
    calibration: Calibration,
    pose_samples: list[PoseSnapshot],
    vr_samples: list[VrPoseSnapshot | None],
    intrinsics: CameraIntrinsics,
) -> tuple[Calibration, VrAlignmentInfo]:
    records: list[tuple[np.ndarray, np.ndarray, VrPoseSnapshot]] = []
    for pose, vr_pose in zip(pose_samples, vr_samples, strict=False):
        if vr_pose is None or not vr_pose.ready:
            continue
        if pose.score((Landmark.LEFT_WRIST, Landmark.RIGHT_WRIST)) < 0.25:
            continue
        try:
            camera_points, _ = reconstruct_camera_points(
                pose,
                intrinsics,
                calibration.scale,
            )
        except CalibrationError:
            continue
        tracking = calibration.camera_to_tracking(camera_points)
        records.append(
            (
                tracking[Landmark.LEFT_WRIST].copy(),
                tracking[Landmark.RIGHT_WRIST].copy(),
                vr_pose,
            )
        )

    if len(records) < 6:
        raise CalibrationError("VR 辅助样本不足，请保持双手柄可见")

    camera_left = np.stack([record[0] for record in records])
    camera_right = np.stack([record[1] for record in records])
    vr_left = np.stack([record[2].left_hand.position_m for record in records])
    vr_right = np.stack([record[2].right_hand.position_m for record in records])

    camera_axis = np.median(camera_right - camera_left, axis=0)
    vr_axis = np.median(vr_right - vr_left, axis=0)
    if np.linalg.norm(camera_axis[[0, 2]]) < 0.20 or np.linalg.norm(vr_axis[[0, 2]]) < 0.20:
        raise CalibrationError("请将双手柄自然分开放在身体两侧")
    yaw_offset = _wrap_angle(_right_axis_angle(vr_axis) - _right_axis_angle(camera_axis))

    camera_midpoint = np.median((camera_left + camera_right) * 0.5, axis=0)
    vr_midpoint = np.median((vr_left + vr_right) * 0.5, axis=0)
    rotated_midpoint = rotate_y(camera_midpoint, yaw_offset)
    tracking_offset = vr_midpoint - rotated_midpoint
    tracking_offset[1] = 0.0
    aligned = replace(
        calibration,
        vr_yaw_offset_deg=yaw_offset,
        tracking_offset_m=tracking_offset,
        vr_assisted=True,
    )

    left_offsets: list[np.ndarray] = []
    right_offsets: list[np.ndarray] = []
    errors: list[float] = []
    for camera_left_wrist, camera_right_wrist, vr_pose in records:
        aligned_left = rotate_y(camera_left_wrist, yaw_offset) + tracking_offset
        aligned_right = rotate_y(camera_right_wrist, yaw_offset) + tracking_offset
        left_offsets.append(
            vr_pose.left_hand.rotation.T
            @ (aligned_left - vr_pose.left_hand.position_m)
        )
        right_offsets.append(
            vr_pose.right_hand.rotation.T
            @ (aligned_right - vr_pose.right_hand.position_m)
        )
        errors.extend(
            (
                float(np.linalg.norm((aligned_left - vr_pose.left_hand.position_m)[[0, 2]])),
                float(np.linalg.norm((aligned_right - vr_pose.right_hand.position_m)[[0, 2]])),
            )
        )

    left_offset = np.median(np.stack(left_offsets), axis=0)
    right_offset = np.median(np.stack(right_offsets), axis=0)
    horizontal_error = float(np.median(errors))
    if (
        horizontal_error > 0.30
        or np.linalg.norm(left_offset) > 0.30
        or np.linalg.norm(right_offset) > 0.30
    ):
        raise CalibrationError("摄像头与 VR 双手位置不一致，请重新校准")

    aligned = replace(
        aligned,
        left_hand_offset_local_m=left_offset,
        right_hand_offset_local_m=right_offset,
    )
    return aligned, VrAlignmentInfo(len(records), yaw_offset, horizontal_error)


def _solve_two_bone_joint(
    root: np.ndarray,
    joint_hint: np.ndarray,
    target: np.ndarray,
    first_length: float,
    second_length: float,
) -> np.ndarray | None:
    direction = np.asarray(target, dtype=np.float64) - np.asarray(root, dtype=np.float64)
    distance = float(np.linalg.norm(direction))
    if distance < 1e-4:
        return None
    maximum = max(1e-3, first_length + second_length - 1e-4)
    minimum = abs(first_length - second_length) + 1e-4
    solved_distance = float(np.clip(distance, minimum, maximum))
    axis = direction / distance
    along = (
        first_length * first_length
        - second_length * second_length
        + solved_distance * solved_distance
    ) / (2.0 * solved_distance)
    perpendicular_length = float(
        np.sqrt(max(0.0, first_length * first_length - along * along))
    )

    hint = np.asarray(joint_hint, dtype=np.float64) - np.asarray(root, dtype=np.float64)
    bend = hint - axis * float(np.dot(hint, axis))
    bend_length = float(np.linalg.norm(bend))
    if bend_length < 1e-4:
        bend = np.cross(np.array([0.0, 1.0, 0.0]), axis)
        bend_length = float(np.linalg.norm(bend))
    if bend_length < 1e-4:
        bend = np.cross(np.array([0.0, 0.0, 1.0]), axis)
        bend_length = float(np.linalg.norm(bend))
    if bend_length < 1e-4:
        return None
    bend /= bend_length
    return np.asarray(root, dtype=np.float64) + axis * along + bend * perpendicular_length


def _constrain_joint(
    points: np.ndarray,
    root_index: Landmark,
    joint_index: Landmark,
    target: np.ndarray,
    first_length: float,
    second_length: float,
    weight: float,
) -> bool:
    solved = _solve_two_bone_joint(
        points[root_index],
        points[joint_index],
        target,
        first_length,
        second_length,
    )
    if solved is None:
        return False
    weight = float(np.clip(weight, 0.0, 1.0))
    points[joint_index] = points[joint_index] * (1.0 - weight) + solved * weight
    return True


def _apply_skeleton_constraints(
    points: np.ndarray,
    snapshot: PoseSnapshot,
    calibration: Calibration,
    vr_pose: VrPoseSnapshot | None,
) -> tuple[np.ndarray, set[str]]:
    constrained = np.asarray(points, dtype=np.float64).copy()
    proportions = calibration.proportions
    roles: set[str] = set()

    left_knee_weight = 0.25 + 0.30 * (1.0 - snapshot.confidence[Landmark.LEFT_KNEE])
    right_knee_weight = 0.25 + 0.30 * (1.0 - snapshot.confidence[Landmark.RIGHT_KNEE])
    _constrain_joint(
        constrained,
        Landmark.LEFT_HIP,
        Landmark.LEFT_KNEE,
        constrained[Landmark.LEFT_ANKLE],
        proportions.left_thigh_m,
        proportions.left_shin_m,
        left_knee_weight,
    )
    _constrain_joint(
        constrained,
        Landmark.RIGHT_HIP,
        Landmark.RIGHT_KNEE,
        constrained[Landmark.RIGHT_ANKLE],
        proportions.right_thigh_m,
        proportions.right_shin_m,
        right_knee_weight,
    )

    if vr_pose is None or not calibration.vr_assisted:
        return constrained, roles

    if vr_pose.left_hand is not None:
        left_target = (
            vr_pose.left_hand.position_m
            + vr_pose.left_hand.rotation @ calibration.left_hand_offset_local_m
        )
        reach = float(np.linalg.norm(left_target - constrained[Landmark.LEFT_SHOULDER]))
        if reach <= (proportions.left_upper_arm_m + proportions.left_forearm_m) * 1.35:
            weight = 0.75 - 0.35 * snapshot.confidence[Landmark.LEFT_ELBOW]
            if _constrain_joint(
                constrained,
                Landmark.LEFT_SHOULDER,
                Landmark.LEFT_ELBOW,
                left_target,
                proportions.left_upper_arm_m,
                proportions.left_forearm_m,
                weight,
            ):
                roles.add("left_elbow")

    if vr_pose.right_hand is not None:
        right_target = (
            vr_pose.right_hand.position_m
            + vr_pose.right_hand.rotation @ calibration.right_hand_offset_local_m
        )
        reach = float(np.linalg.norm(right_target - constrained[Landmark.RIGHT_SHOULDER]))
        if reach <= (proportions.right_upper_arm_m + proportions.right_forearm_m) * 1.35:
            weight = 0.75 - 0.35 * snapshot.confidence[Landmark.RIGHT_ELBOW]
            if _constrain_joint(
                constrained,
                Landmark.RIGHT_SHOULDER,
                Landmark.RIGHT_ELBOW,
                right_target,
                proportions.right_upper_arm_m,
                proportions.right_forearm_m,
                weight,
            ):
                roles.add("right_elbow")
    return constrained, roles


def trackers_from_pose(
    snapshot: PoseSnapshot,
    calibration: Calibration,
    intrinsics: CameraIntrinsics,
    mode: str,
    vr_pose: VrPoseSnapshot | None = None,
) -> tuple[list[TrackerPose], ReconstructionInfo]:
    camera_points, reconstruction = reconstruct_camera_points(
        snapshot, intrinsics, calibration.scale
    )
    points = calibration.camera_to_tracking(camera_points)
    points, constrained_roles = _apply_skeleton_constraints(
        points,
        snapshot,
        calibration,
        vr_pose,
    )
    body_yaw = _body_heading(points)

    hip = _mean(points, [Landmark.LEFT_HIP, Landmark.RIGHT_HIP])
    shoulders = _mean(points, [Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER])
    left_foot = (
        points[Landmark.LEFT_ANKLE] * 0.55
        + points[Landmark.LEFT_HEEL] * 0.225
        + points[Landmark.LEFT_FOOT_INDEX] * 0.225
    )
    right_foot = (
        points[Landmark.RIGHT_ANKLE] * 0.55
        + points[Landmark.RIGHT_HEEL] * 0.225
        + points[Landmark.RIGHT_FOOT_INDEX] * 0.225
    )
    left_foot[1] = max(0.0, float(left_foot[1]))
    right_foot[1] = max(0.0, float(right_foot[1]))

    positions = {
        "hip": hip,
        "left_foot": left_foot,
        "right_foot": right_foot,
        "chest": shoulders * 0.68 + hip * 0.32,
        "left_knee": points[Landmark.LEFT_KNEE],
        "right_knee": points[Landmark.RIGHT_KNEE],
        "left_elbow": points[Landmark.LEFT_ELBOW] * 0.80
        + points[Landmark.LEFT_SHOULDER] * 0.20,
        "right_elbow": points[Landmark.RIGHT_ELBOW] * 0.80
        + points[Landmark.RIGHT_SHOULDER] * 0.20,
    }

    left_foot_yaw = _yaw_from_forward(
        points[Landmark.LEFT_FOOT_INDEX] - points[Landmark.LEFT_HEEL], body_yaw
    )
    right_foot_yaw = _yaw_from_forward(
        points[Landmark.RIGHT_FOOT_INDEX] - points[Landmark.RIGHT_HEEL], body_yaw
    )
    yaws = {role: body_yaw for role in TRACKER_LAYOUT}
    yaws["left_foot"] = left_foot_yaw
    yaws["right_foot"] = right_foot_yaw

    confidence_ids = {
        "hip": [Landmark.LEFT_HIP, Landmark.RIGHT_HIP],
        "left_foot": [Landmark.LEFT_ANKLE, Landmark.LEFT_HEEL, Landmark.LEFT_FOOT_INDEX],
        "right_foot": [Landmark.RIGHT_ANKLE, Landmark.RIGHT_HEEL, Landmark.RIGHT_FOOT_INDEX],
        "chest": [Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER],
        "left_knee": [Landmark.LEFT_HIP, Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE],
        "right_knee": [Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE],
        "left_elbow": [Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW],
        "right_elbow": [Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW],
    }

    roles = STABLE_ROLES if mode == "stable" else FULL_ROLES
    trackers: list[TrackerPose] = []
    for role in roles:
        tracker_id, label = TRACKER_LAYOUT[role]
        confidence = snapshot.score(confidence_ids[role])
        if role in constrained_roles:
            shoulder = (
                Landmark.LEFT_SHOULDER if role == "left_elbow" else Landmark.RIGHT_SHOULDER
            )
            confidence = max(confidence, min(0.90, float(snapshot.confidence[shoulder])))
        trackers.append(
            TrackerPose(
                role=role,
                tracker_id=tracker_id,
                label=label,
                position_m=np.asarray(positions[role], dtype=np.float64),
                euler_deg=np.array([0.0, yaws[role], 0.0], dtype=np.float64),
                confidence=confidence,
            )
        )
    return trackers, reconstruction


class TrackingStabilizer:
    def __init__(self, hold_seconds: float = 0.28) -> None:
        self.hold_seconds = hold_seconds
        self._states: dict[str, _FilterState] = {}

    def reset(self) -> None:
        self._states.clear()

    def update(
        self,
        raw_poses: list[TrackerPose],
        timestamp_s: float,
        smoothing: float,
        min_confidence: float,
    ) -> list[TrackerPose]:
        accepted = {pose.role: pose for pose in raw_poses if pose.confidence >= min_confidence}
        output: list[TrackerPose] = []
        tau = 0.004 + 0.38 * float(np.clip(smoothing, 0.0, 0.95)) ** 2

        for role in set(self._states) | set(accepted):
            candidate = accepted.get(role)
            state = self._states.get(role)
            if candidate is not None:
                if state is None:
                    filtered = candidate.copy()
                else:
                    dt = max(1e-3, timestamp_s - state.last_update_s)
                    jump = float(np.linalg.norm(candidate.position_m - state.pose.position_m))
                    max_jump = max(0.30, 5.5 * dt)
                    if jump > max_jump:
                        candidate = None
                    else:
                        alpha = 1.0 - exp(-dt / tau)
                        filtered = candidate.copy()
                        filtered.position_m = state.pose.position_m + alpha * (
                            candidate.position_m - state.pose.position_m
                        )
                        angle_delta = (candidate.euler_deg - state.pose.euler_deg + 180.0) % 360.0 - 180.0
                        filtered.euler_deg = (state.pose.euler_deg + alpha * angle_delta + 180.0) % 360.0 - 180.0

                if candidate is not None:
                    self._states[role] = _FilterState(
                        pose=filtered,
                        last_seen_s=timestamp_s,
                        last_update_s=timestamp_s,
                    )
                    output.append(filtered.copy())
                    continue

            if state is not None and timestamp_s - state.last_seen_s <= self.hold_seconds:
                held = state.pose.copy()
                held.stale = True
                held.confidence *= max(
                    0.0, 1.0 - (timestamp_s - state.last_seen_s) / self.hold_seconds
                )
                state.last_update_s = timestamp_s
                output.append(held)
            elif state is not None:
                del self._states[role]

        output.sort(key=lambda pose: pose.tracker_id)
        return output
