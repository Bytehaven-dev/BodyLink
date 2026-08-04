from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bodylink.config import AppConfig
from bodylink.geometry import (
    CALIBRATION_LANDMARKS,
    CameraIntrinsics,
    calibration_pose_ready,
)
from bodylink.rtmw3d import Rtmw3dTracker
from bodylink.vision_worker import _open_capture, directshow_camera_names, fourcc_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a short BodyLink camera check")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--format", choices=("auto", "mjpg", "yuy2"), default="mjpg")
    args = parser.parse_args()

    capture = _open_capture(
        AppConfig(
            camera_index=args.camera,
            camera_width=args.width,
            camera_height=args.height,
            camera_fps=args.fps,
            camera_format=args.format,
        )
    )
    names = directshow_camera_names()
    camera_name = names[args.camera] if args.camera < len(names) else f"Camera {args.camera}"
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    actual_format = fourcc_name(capture.get(cv2.CAP_PROP_FOURCC))
    tracker = Rtmw3dTracker()
    latencies: list[float] = []
    detected = 0
    pose_scores: list[float] = []
    calibration_ready_frames = 0
    read_failures = 0
    try:
        while len(latencies) < max(1, args.frames):
            ok, frame = capture.read()
            if not ok:
                read_failures += 1
                if read_failures >= 20:
                    raise RuntimeError(
                        "camera frame read failed 20 times; close any app using the camera"
                    )
                time.sleep(0.015)
                continue
            read_failures = 0
            intrinsics = CameraIntrinsics(frame.shape[1], frame.shape[0], 60.0)
            started = time.perf_counter()
            snapshot = tracker.infer(frame, started, intrinsics)
            latencies.append((time.perf_counter() - started) * 1000.0)
            detected += int(snapshot is not None)
            if snapshot is not None:
                score = snapshot.score(CALIBRATION_LANDMARKS)
                pose_scores.append(score)
                calibration_ready_frames += int(calibration_pose_ready(snapshot))
    finally:
        tracker.close()
        capture.release()

    print(
        f"Camera {args.camera} ({camera_name}): "
        f"{actual_width}x{actual_height} @ {actual_fps:.1f} FPS {actual_format}, "
        f"pose detected {detected}/{len(latencies)} frames, "
        f"calibration ready {calibration_ready_frames}/{len(latencies)} frames"
    )
    if pose_scores:
        print(f"Pose confidence: median {statistics.median(pose_scores) * 100:.1f}%")
    steady = latencies[1:] if len(latencies) > 1 else latencies
    print(
        "CUDA inference: "
        f"warm-up {latencies[0]:.1f} ms, "
        f"steady median {statistics.median(steady):.1f} ms, "
        f"steady average {statistics.fmean(steady):.1f} ms, "
        f"steady max {max(steady):.1f} ms"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Camera check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
