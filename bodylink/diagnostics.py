from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from bodylink import __version__


def collect_runtime_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "error",
        "version": __version__,
        "frozen": bool(getattr(sys, "frozen", False)),
    }
    try:
        import onnxruntime as ort
        import openvr

        from bodylink.rtmw3d import Rtmw3dTracker, missing_model_paths

        missing = missing_model_paths()
        if missing:
            raise RuntimeError(
                "Missing bundled models: " + ", ".join(path.name for path in missing)
            )

        tracker = Rtmw3dTracker()
        try:
            report["body"] = {
                "provider": tracker.runtime_info.provider,
                "detection_interval": tracker.runtime_info.detection_interval,
                "warmup_ms": round(tracker.runtime_info.warmup_ms, 1),
                "available_providers": ort.get_available_providers(),
            }
        finally:
            tracker.close()

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from bodylink.vision_worker import FACE_MODEL_PATH, directshow_camera_names

        face_options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(FACE_MODEL_PATH),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        face_landmarker.close()

        report["face"] = {
            "provider": "MediaPipe CPU",
            "model": FACE_MODEL_PATH.name,
        }
        report["openvr_version"] = openvr.__version__
        report["camera_names"] = list(directshow_camera_names())
        report["status"] = "ok"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc).strip() or type(exc).__name__
        report["traceback"] = traceback.format_exc()
    return report


def write_runtime_report(path: Path) -> int:
    report = collect_runtime_report()
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return 0 if report["status"] == "ok" else 1
