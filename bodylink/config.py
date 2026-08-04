from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AppConfig:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    camera_format: str = "mjpg"
    target_host: str = "127.0.0.1"
    target_port: int = 9000
    user_height_m: float = 1.70
    horizontal_fov_deg: float = 60.0
    smoothing: float = 0.68
    min_confidence: float = 0.55
    mirror_preview: bool = True
    tracker_mode: str = "stable"
    align_yaw_on_calibrate: bool = True
    vr_assist_enabled: bool = True
    face_enabled: bool = False
    face_native_eyes: bool = True
    face_fps: int = 20
    face_smoothing: float = 0.55

    def normalized(self) -> "AppConfig":
        self.camera_index = max(0, min(9, int(self.camera_index)))
        self.camera_width = max(320, min(3840, int(self.camera_width)))
        self.camera_height = max(240, min(2160, int(self.camera_height)))
        self.camera_fps = max(10, min(60, int(self.camera_fps)))
        self.camera_format = (
            self.camera_format
            if self.camera_format in {"auto", "mjpg", "yuy2"}
            else "mjpg"
        )
        self.target_host = str(self.target_host).strip() or "127.0.0.1"
        self.target_port = max(1, min(65535, int(self.target_port)))
        self.user_height_m = max(1.20, min(2.20, float(self.user_height_m)))
        self.horizontal_fov_deg = max(35.0, min(110.0, float(self.horizontal_fov_deg)))
        self.smoothing = max(0.0, min(0.95, float(self.smoothing)))
        self.min_confidence = max(0.20, min(0.95, float(self.min_confidence)))
        self.tracker_mode = self.tracker_mode if self.tracker_mode in {"stable", "full"} else "stable"
        self.face_fps = max(10, min(30, int(self.face_fps)))
        self.face_smoothing = max(0.0, min(0.95, float(self.face_smoothing)))
        return self


def settings_path() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "BodyLink" / "settings.json"
    return Path.home() / ".bodylink" / "settings.json"


def load_config(path: Path | None = None) -> AppConfig:
    source = path or settings_path()
    if not source.exists():
        return AppConfig()

    try:
        raw: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
        allowed = {field.name for field in fields(AppConfig)}
        return AppConfig(**{key: value for key, value in raw.items() if key in allowed}).normalized()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return AppConfig()


def save_config(config: AppConfig, path: Path | None = None) -> None:
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(asdict(config.normalized()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)
