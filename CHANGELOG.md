# Changelog

English | [简体中文](CHANGELOG.zh.md)

## 0.3.0 - 2026-08-05

### Features

- Add RTMW3D-X 133-point full-body tracking with YOLOX-M person detection on ONNX Runtime CUDA.
- Add stable 3-point and full 8-point VRChat OSC tracker modes.
- Add optional SteamVR/Pico head and hand alignment with camera-assisted elbow IK.
- Add optional MediaPipe face tracking with native VRChat eye tracking and VRCFaceTracking v2 expressions.
- Add named DirectShow camera discovery and selectable MJPEG, YUY2, and automatic capture formats.
- Add a Windows installer that bundles the GPU runtime and downloads a separately checksummed model pack, with an offline side-by-side fallback.
- Pin MediaPipe 0.10.33 to avoid the Clearcut network logging and shutdown delay present in newer Windows wheels.

### Documentation

- Document camera placement, calibration, Pico 4 Ultra integration, GPU usage, and current tracking limits.
- Record optional TensorRT FP16 acceleration as a future backend with CUDA fallback.
