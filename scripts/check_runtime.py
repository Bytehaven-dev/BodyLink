from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bodylink.rtmw3d import Rtmw3dTracker
import openvr


def main() -> int:
    tracker = Rtmw3dTracker()
    try:
        info = tracker.runtime_info
        print(
            "RTMW3D runtime ready: "
            f"{info.provider}, detector every {info.detection_interval} frames"
        )
        print(f"RTMW3D CUDA detector and pose warm-up: {info.warmup_ms:.1f} ms")
        print(f"OpenVR bindings ready: {openvr.__version__}")
    finally:
        tracker.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CUDA runtime check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
